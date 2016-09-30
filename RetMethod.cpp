/*==========================================================================
 *
 *  Original source copyright (c) 2001, Carnegie Mellon University.
 *  See copyright.cmu for details.
 *  Modifications copyright (c) 2002, University of Massachusetts.
 *  See copyright.umass for details.
 *
 *==========================================================================
 */



#include "RetMethod.h"
#include "Param.hpp"
#include "common_headers.hpp"
#include <cmath>
#include "DocUnigramCounter.hpp"
#include "RelDocUnigramCounter.hpp"
#include "IndexManager.hpp"
#include "OneStepMarkovChain.hpp"
#include <vector>
#include <set>
#include <algorithm>
#include <FreqVector.hpp>

#include <fstream>
#include <sstream>


using namespace lemur::api;
using namespace lemur::retrieval;
using namespace lemur::utility;

bool pairCompare(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem);
struct sort_pred{
    bool operator()(const std::pair<int,float> &left,const std::pair<int,float> &right){
        return left.second > right.second;
    }
};
extern int RSMethodHM; // 0--> LM , 1--> RecSys
extern int negGenModeHM;//0 --> coll , 1--> nonRel
extern int feedbackMode;
extern int updatingThresholdMode; // 0 ->no updating ,1->linear

extern map<int,vector<double> >wordEmbedding;
extern map<int,vector<double> >docIdKeyWords;
extern set<int> stopWords;
//extern map<int, map<int,double> > FEEDBACKMAP;


static int qid=1;
static string RM;
Index* myIndex = NULL;
void lemur::retrieval::QueryModel::interpolateWith(const lemur::langmod::UnigramLM &qModel,
                                                   double origModCoeff,
                                                   int howManyWord,
                                                   double prSumThresh,
                                                   double prThresh) {
    if (!qm) {
        qm = new lemur::api::IndexedRealVector();
    } else {
        qm->clear();
    }

    qModel.startIteration();
    while (qModel.hasMore()) {
        IndexedReal entry;
        qModel.nextWordProb((TERMID_T &)entry.ind,entry.val);
        qm->push_back(entry);

    }
    qm->Sort();

    double countSum = totalCount();

    // discounting the original model
    startIteration();
    while (hasMore()) {
        QueryTerm *qt = nextTerm();
        incCount(qt->id(), qt->weight()*origModCoeff/countSum);
        delete qt;
    }

    // now adding the new model
    double prSum = 0;
    int wdCount = 0;
    IndexedRealVector::iterator it;
    it = qm->begin();
    while (it != qm->end() && prSum < prSumThresh &&
           wdCount < howManyWord && (*it).val >=prThresh) {
        incCount((*it).ind, (*it).val*(1-origModCoeff));
        prSum += (*it).val;
        it++;
        wdCount++;
    }

    //Sum w in Q qtf * log(qtcf/termcount);
    colQLikelihood = 0;
    colQueryLikelihood();
    colKLComputed = false;
}

void lemur::retrieval::QueryModel::load(istream &is)
{
    // clear existing counts
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        setCount(qt->id(),0);
    }
    colQLikelihood = 0;

    int count;
    is >> count;
    char wd[500];
    double pr;
    while (count-- >0) {
        is >> wd >> pr;
        TERMID_T id = ind.term(wd);
        if (id != 0) setCount(id, pr); // don't load OOV terms
    }
    colQueryLikelihood();
    colKLComputed = false;
}

void lemur::retrieval::QueryModel::save(ostream &os)
{
    int count = 0;
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        count++;
        delete qt;
    }
    os << " " << count << endl;
    startIteration();
    while (hasMore()) {
        qt = nextTerm();
        os << ind.term(qt->id()) << " "<< qt->weight() << endl;
        delete qt;
    }
}

void lemur::retrieval::QueryModel::clarity(ostream &os)
{
    int count = 0;
    double sum=0, ln_Pr=0;
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        count++;
        // query-clarity = SUM_w{P(w|Q)*log(P(w|Q)/P(w))}
        // P(w)=cf(w)/|C|
        double pw = ((double)ind.termCount(qt->id())/(double)ind.termCount());
        // P(w|Q) is a prob computed by any model, e.g. relevance models
        double pwq = qt->weight();
        sum += pwq;
        ln_Pr += (pwq)*log(pwq/pw);
        delete qt;
    }
    // clarity should be computed with log_2, so divide by log(2).
    os << "=" << count << " " << (ln_Pr/(sum ? sum : 1.0)/log(2.0)) << endl;
    startIteration();
    while (hasMore()) {
        qt = nextTerm();
        // print clarity for each query term
        // clarity should be computed with log_2, so divide by log(2).
        os << ind.term(qt->id()) << " "
           << (qt->weight()*log(qt->weight()/
                                ((double)ind.termCount(qt->id())/
                                 (double)ind.termCount())))/log(2.0) << endl;
        delete qt;
    }
}

double lemur::retrieval::QueryModel::clarity() const
{
    int count = 0;
    double sum=0, ln_Pr=0;
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        count++;
        // query-clarity = SUM_w{P(w|Q)*log(P(w|Q)/P(w))}
        // P(w)=cf(w)/|C|
        double pw = ((double)ind.termCount(qt->id())/(double)ind.termCount());
        // P(w|Q) is a prob computed by any model, e.g. relevance models
        double pwq = qt->weight();
        sum += pwq;
        ln_Pr += (pwq)*log(pwq/pw);
        delete qt;
    }
    // normalize by sum of probabilities in the input model
    ln_Pr = ln_Pr/(sum ? sum : 1.0);
    // clarity should be computed with log_2, so divide by log(2).
    return (ln_Pr/log(2.0));
}

lemur::retrieval::RetMethod::RetMethod(const Index &dbIndex,
                                       const string &supportFileName,
                                       ScoreAccumulator &accumulator) :
    TextQueryRetMethod(dbIndex, accumulator), supportFile(supportFileName) {

    //docParam.smthMethod = RetParameter::defaultSmoothMethod;
    docParam.smthMethod = RetParameter::DIRICHLETPRIOR;
    //docParam.smthMethod = RetParameter::ABSOLUTEDISCOUNT;
    //docParam.smthMethod = RetParameter::JELINEKMERCER;

    docParam.smthStrategy= RetParameter::defaultSmoothStrategy;
    //docParam.ADDelta = RetParameter::defaultADDelta;
    docParam.JMLambda = RetParameter::defaultJMLambda;
    //docParam.JMLambda = 0.9;
    docParam.DirPrior = dbIndex.docLengthAvg();//50;//RetParameter::defaultDirPrior;

    qryParam.adjScoreMethod = RetParameter::NEGATIVEKLD;
    //qryParam.adjScoreMethod = RetParameter::QUERYLIKELIHOOD;
    //qryParam.fbMethod = RetParameter::defaultFBMethod;
    //qryParam.fbMethod = RetParameter::DIVMIN;




    qryParam.fbMethod = RetParameter::MIXTURE;
    RM="MIX";// *** Query Likelihood adjusted score method *** //
    qryParam.fbCoeff = RetParameter::defaultFBCoeff;//default = 0.5
    //qryParam.fbCoeff =0.1;
    qryParam.fbPrTh = RetParameter::defaultFBPrTh;
    qryParam.fbPrSumTh = RetParameter::defaultFBPrSumTh;
    qryParam.fbTermCount = 45;//RetParameter::defaultFBTermCount;//default = 50
    qryParam.fbMixtureNoise = RetParameter::defaultFBMixNoise;
    qryParam.emIterations = 50;//RetParameter::defaultEMIterations;

    docProbMass = NULL;
    uniqueTermCount = NULL;
    mcNorm = NULL;

    NegMu = ind.docLengthAvg();
    collectLMCounter = new lemur::langmod::DocUnigramCounter(ind);
    collectLM = new lemur::langmod::MLUnigramLM(*collectLMCounter, ind.termLexiconID());

    delta = 0.007;

    newNonRelRecieved = false;
    newRelRecieved = false;
    newNonRelRecievedCnt = 0,newRelRecievedCnt =0;

    //int W2VecSize =100;
    /*coefMatrix = new double*[W2VecSize];
    for(int i = 0 ; i< W2VecSize ; i++)
        coefMatrix[i] = new double[W2VecSize];
    for(int i = 0 ; i< W2VecSize ; i++)
        for(int j = 0 ; j < W2VecSize ; j++)
            coefMatrix[i][j]=((rand()%2000)-1000)/1000.0f;
    */
    W2VecDimSize = 100;
    /*coefMatrix.resize(W2VecDimSize);
    for(int i = 0 ;i < W2VecDimSize ; i++)
        coefMatrix[i].resize(W2VecDimSize);

    for(int i = 0 ;i < W2VecDimSize ; i++)
        for(int j = 0 ; j < W2VecDimSize ; j++)
        {
            coefMatrix[i][j] = ((rand()%2000)-1000)/1000.0f;
            //cerr<<coefMatrix[i][j]<<" ";
        }
        */
    /*alphaCoef = 0.8;
    lambdaCoef = 0.05;
    betaCoef = 0.01;
    etaCoef = 0.0000001;*/

    alphaCoef = 0.8;
    lambdaCoef = 0.05;
    betaCoef = 0.01;
    etaCoef =0.000001;
    //queryAvgVec.assign(W2VecDimSize , -10.0);
    Vq.assign(W2VecDimSize , 0.0);
    //Vbwn.assign(W2VecDimSize , 0.0);
    //Vwn.assign(W2VecDimSize , 0.0);

    numberOfPositiveSelectedTopWord = 30.0;
    numberOfNegativeSelectedTopWord = 80.0;


    /*
    switch (RSMethodHM)
    {
    case 0://lm
        // setThreshold(-4.3);
        setThreshold(-6.3);

        break;
    case 1://negGen
    {
        if(negGenModeHM==0)//col//mu=2500
            // setThreshold(2.1);
            setThreshold(1.5);
        else if(negGenModeHM == 1)
            // setThreshold(2.4);
            setThreshold(1.6);
        break;
    }
    }
*/
    //prev_distQuery = new double[ind.termCountUnique()+1];
    scFunc = new ScoreFunc();
    scFunc->setScoreMethod(qryParam.adjScoreMethod);

}
void lemur::retrieval::RetMethod::setCoeffParam(double coe)
{
    qryParam.fbCoeff = coe;
}
lemur::retrieval::RetMethod::~RetMethod()
{
    //delete [] prev_distQuery;
    delete [] docProbMass;
    delete [] uniqueTermCount;
    delete [] mcNorm;
    delete collectLM;
    delete collectLMCounter;
    delete scFunc;

    //delete [] relComputed;//FIX ME!!!!!!!
    //delete [] nonRelComputed;//FIX ME!!!!!!!

    //delete [] coefMatrix[]; //FIX ME!!!!
}

void lemur::retrieval::RetMethod::loadSupportFile() {
    ifstream ifs;
    int i;

    // Only need to load this file if smooth strategy is back off
    // or the smooth method is absolute discount. Don't reload if
    // docProbMass is not NULL.

    if (docProbMass == NULL &&
            (docParam.smthMethod == RetParameter::ABSOLUTEDISCOUNT ||
             docParam.smthStrategy == RetParameter::BACKOFF)) {
        cerr << "lemur::retrieval::SimpleKLRetMethod::loadSupportFile loading "
             << supportFile << endl;

        ifs.open(supportFile.c_str());
        if (ifs.fail()) {
            throw  Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                             "smoothing support file open failure");
        }
        COUNT_T numDocs = ind.docCount();
        docProbMass = new double[numDocs+1];
        uniqueTermCount = new COUNT_T[numDocs+1];
        for (i = 1; i <= numDocs; i++) {
            DOCID_T id;
            int uniqCount;
            double prMass;
            ifs >> id >> uniqCount >> prMass;
            if (id != i) {
                throw  Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                                 "alignment error in smooth support file, wrong id:");
            }
            docProbMass[i] = prMass;
            uniqueTermCount[i] = uniqCount;
        }
        ifs.close();
    }

    // only need to load this file if the feedback method is
    // markov chain. Don't reload if called a second time.

    if (mcNorm == NULL && qryParam.fbMethod == RetParameter::MARKOVCHAIN) {
        string mcSuppFN = supportFile + ".mc";
        cerr << "lemur::retrieval::SimpleKLRetMethod::loadSupportFile loading " << mcSuppFN << endl;

        ifs.open(mcSuppFN.c_str());
        if (ifs.fail()) {
            throw Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                            "Markov chain support file can't be opened");
        }

        mcNorm = new double[ind.termCountUnique()+1];

        for (i = 1; i <= ind.termCountUnique(); i++) {
            TERMID_T id;
            double norm;
            ifs >> id >> norm;
            if (id != i) {
                throw Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                                "alignment error in Markov chain support file, wrong id:");
            }
            mcNorm[i] = norm;
        }
    }
}

DocumentRep *lemur::retrieval::RetMethod::computeDocRep(DOCID_T docID)
{
    switch (docParam.smthMethod) {
    case RetParameter::JELINEKMERCER:
        return( new JMDocModel(docID,
                               ind.docLength(docID),
                               *collectLM,
                               docProbMass,
                               docParam.JMLambda,
                               docParam.smthStrategy));
    case RetParameter::DIRICHLETPRIOR:
        return (new DPriorDocModel(docID,
                                   ind.docLength(docID),
                                   *collectLM,
                                   docProbMass,
                                   docParam.DirPrior,
                                   docParam.smthStrategy));
    case RetParameter::ABSOLUTEDISCOUNT:
        return (new ABSDiscountDocModel(docID,
                                        ind.docLength(docID),
                                        *collectLM,
                                        docProbMass,
                                        uniqueTermCount,
                                        docParam.ADDelta,
                                        docParam.smthStrategy));
    case RetParameter::TWOSTAGE:
        return (new TStageDocModel(docID,
                                   ind.docLength(docID),
                                   *collectLM,
                                   docProbMass,
                                   docParam.DirPrior, // 1st stage mu
                                   docParam.JMLambda, // 2nd stage lambda
                                   docParam.smthStrategy));


    default:
        // this should throw, not exit.
        cerr << "Unknown document language model smoothing method\n";
        exit(1);
    }
}


void lemur::retrieval::RetMethod::updateProfile(lemur::api::TextQueryRep &origRep,
                                                vector<int> relJudgDoc ,vector<int> nonRelJudgDoc, bool isRel)
{



#if 0
    cerr<<"relJudgDoc size: "<<relJudgDoc.size()<<" nonRelJudgDoc size: "<<nonRelJudgDoc.size()<<endl;
    if(relComputed[relJudgDoc.size()] == false)
    {
        relComputed[relJudgDoc.size()] = true;
        relJudgDoc.insert(relJudgDoc.end(),initRel.begin(),initRel.end());
        computeRelNonRelDist(origRep,relJudgDoc ,nonRelJudgDoc,true,true);
    }

    if(nonRelComputed[nonRelJudgDoc.size()] == false)
    {
        nonRelComputed[nonRelJudgDoc.size()] = true;
        nonRelJudgDoc.insert(nonRelJudgDoc.end(), initNonRel.begin(),initNonRel.end());
        computeRelNonRelDist(origRep,relJudgDoc ,nonRelJudgDoc,false,true);
    }
    /**************************************************************************************************/

    map<int, vector<double> >::iterator endIt = wordEmbedding.end();
    vector<pair<double, int> >probWordVec;
    lemur::langmod::DocUnigramCounter *dCounter;
    dCounter = new lemur::langmod::DocUnigramCounter(relJudgDoc, ind);

    dCounter->startIteration();
    while(dCounter->hasMore())
    {
        int eventInd;
        double weight;
        dCounter->nextCount(eventInd,weight);

        map<int, vector<double> >::iterator tempit = wordEmbedding.find(eventInd);
        if( tempit != endIt )
        {
            vector<double> tt = tempit->second;

            float sc = cosineSim(Vq , tt);
            //float sc = softMaxFunc(Vq , tt);
            //float sc = softMaxFunc2(Vq , tt);

            probWordVec.push_back(pair<double,int>(sc,eventInd));
        }
    }
    double total_sc= 0;
    for(int i = 0 ; i < probWordVec.size() ; i++)
    {
        int cc = dCounter->count(probWordVec[i].second );
        probWordVec[i].first =  log(1+cc)* exp(probWordVec[i].first);
        total_sc += probWordVec[i].first;
    }
    for(int i = 0 ; i < probWordVec.size() ; i++)
        probWordVec[i].first /= total_sc;

    COUNT_T numTerms = ind.termCountUnique();
    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);


    std::sort(probWordVec.begin() , probWordVec.end() , pairCompare);
    int countPos = min((int)numberOfPositiveSelectedTopWord , (int)probWordVec.size());
    for (int i = 0; i < /*probWordVec.size()*/countPos; i++)
        lmCounter.incCount(probWordVec[i].second , probWordVec[i].first);


    /*origRep.startIteration();
    while(origRep.hasMore())
    {
        QueryTerm *tm = origRep.nextTerm();
        cerr<<ind.term(tm->id())<<" ";
    }
    cerr<<endl;*/

    QueryModel *qr = dynamic_cast<QueryModel *> (&origRep);
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    qr->interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);

    /*
    int ccc = 0;
    fblm->startIteration();
    cerr<<"fblm: ";

    while(fblm->hasMore())
    {
        ccc++;
        int idd;
        double tt;
        fblm->nextWordProb(idd,tt);

        if(idd == ind.term("dope") || idd == ind.term("said") || idd == ind.term("will") ||idd == ind.term("sport") ||idd == ind.term("new") ||
                idd == ind.term("us") ||idd == ind.term("sport") ||idd == ind.term("fight") ||idd == ind.term("athlet") ||idd == ind.term("legisl") )
            cerr<<ind.term(idd)<<" "<<tt<<"\n";
    }
    cerr<<"c: "<<ccc<<" prob: "<<probWordVec.size()<<endl;


    qr->interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);
    cerr<<"\nAFTER: ";
    origRep.startIteration();
    while(origRep.hasMore())
    {
        QueryTerm *tm = origRep.nextTerm();
        cerr<<ind.term(tm->id())<<" , "<<tm->weight()<<" ";

    }
    cerr<<endl;
*/

    delete dCounter;
    delete fblm;

#endif

#if 1
    /*ofstream inputfile;
    inputfile.open("outputfiles/NearestTerm2Vec.txt",ios::app);
    origRep.startIteration();//ehtemalan Fix Meee!!!!!!!!!!!!!!!!!!!!!!!!!
    while(origRep.hasMore())
    {
        int idd = origRep.nextTerm()->id();
        inputfile<<ind.term(idd)<<" ";
    }
    inputfile<<":"<<endl;*/
    //inputfile.close();
    //computeNearestTerm2Vec(Vq);
    //relJudgDoc.insert(relJudgDoc.end(),initRel.begin(),initRel.end());
    IndexedRealVector rel;
    for (int i =0 ; i<relJudgDoc.size() ; i++)
        rel.PushValue(relJudgDoc[i],0);

    PseudoFBDocs  *relDocs;
    relDocs = new PseudoFBDocs(rel,relJudgDoc.size(),true);

    //updateTextQuery(origRep, *relDocs ,*relDocs);

    QueryModel *qr = dynamic_cast<QueryModel *> (&origRep);
    computeWEFBModel(*qr, *relDocs);

    //cerr<<"size: "<<Vq.size()<<endl;
    //computeNearestTerm2Vec(Vq);
    //cerr<<"end size: "<<Vq.size()<<endl;
    return;

    /*origRep.startIteration();//ehtemalan Fix Meee!!!!!!!!!!!!!!!!!!!!!!!!!
    while(origRep.hasMore())
    {
        int idd = origRep.nextTerm()->id();
        inputfile<<ind.term(idd)<<" ";
    }
    inputfile<<":"<<endl;
    inputfile.close();
    //computeNearestTerm2Vec(Vq);
    return;
    */
#endif
}

float lemur::retrieval::RetMethod::cosineSim(vector<double> aa, vector<double> bb)
{
    double numerator = 0.0 ,denominator =0.0,sum_a = 0.0,sum_b = 0.0;
    for(int i = 0 ; i< aa.size() ; i++)
    {
        double a = aa[i];
        double b = bb[i];
        numerator += a*b;
        sum_a += a*a;
        sum_b += b*b;
    }
    denominator = sqrt(sum_a) * sqrt(sum_b);
    //cout<< numerator/denominator;

    return (numerator /denominator) ;

}
float lemur::retrieval::RetMethod::softMaxFunc(vector<double> aa, vector<double> bb)
{
    return exp(cosineSim(aa,bb));
}
float lemur::retrieval::RetMethod::softMaxFunc2(vector<double> aa, vector<double> bb)
{
    return 1/1+exp(-cosineSim(aa,bb));
}

vector<double> lemur::retrieval::RetMethod::extractKeyWord(int newDocId)
{
    double numberOfSelectedTopWord = 10.0;
    COUNT_T numTerms = ind.termCountUnique();

    lemur::langmod::DocUnigramCounter *dCounter;
    dCounter = new lemur::langmod::DocUnigramCounter(newDocId, ind);


    double *distQuery = new double[numTerms+1];
    double *distQueryEst = new double[numTerms+1];


    double meanLL=1e-40;
    double distQueryNorm=0;

    for (int i=1; i<=numTerms;i++)
    {
        distQueryEst[i] = rand()+0.001;
        distQueryNorm += distQueryEst[i];
    }

    double noisePr = 0.9; //qryParam.fbMixtureNoise;
    int itNum = qryParam.emIterations;
    do {
        // re-estimate & compute likelihood
        double ll = 0;

        for (int i=1; i<=numTerms;i++)
        {
            distQuery[i] = distQueryEst[i]/distQueryNorm;
            // cerr << "dist: "<< distQuery[i] << endl;
            distQueryEst[i] =0;
        }

        distQueryNorm = 0;

        // compute likelihood
        dCounter->startIteration();
        while (dCounter->hasMore())
        {
            int wd; //dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);
            ll += wdCt * log (noisePr*collectLM->prob(wd)  // Pc(w)
                              + (1-noisePr)*distQuery[wd]); // Pq(w)
        }
        meanLL = 0.5*meanLL + 0.5*ll;
        if (fabs((meanLL-ll)/meanLL)< 0.0001)
        {
            //cerr << "converged at "<< qryParam.emIterations - itNum+1  << " with likelihood= "<< ll << endl;
            break;
        }

        // update counts
        dCounter->startIteration();
        while (dCounter->hasMore())
        {
            int wd; // dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);

            double prTopic = (1-noisePr)*distQuery[wd]/
                    ((1-noisePr)*distQuery[wd]+noisePr*collectLM->prob(wd));

            double incVal = wdCt*prTopic;
            distQueryEst[wd] += incVal;
            distQueryNorm += incVal;
        }
    } while (itNum-- > 0);

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (int i=1; i<=numTerms; i++)
        if (distQuery[i] > 0)
            lmCounter.incCount(i, distQuery[i]);


    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    //origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
    //                      qryParam.fbPrSumTh, qryParam.fbPrTh);

    vector<pair<double, int> >probWordVec;

    fblm->startIteration();
    while(fblm->hasMore())
    {
        int wid=-10;
        double wprob=0.0;
        fblm->nextWordProb(wid,wprob);
        probWordVec.push_back(pair<double,int>(wprob,wid));
    }

    //ofstream outputfile;
    //outputfile.open("salamsalam");

    vector<double>avgDoc(W2VecDimSize ,0.0);

    std::sort(probWordVec.begin(),probWordVec.end(),pairCompare);

    double wordCount = std::min((double)probWordVec.size() , numberOfSelectedTopWord);

    const std::map<int,vector<double> >::iterator endIt = wordEmbedding.end();
    for(int i = 0 ; i < wordCount; i++)
    {
        const std::map<int,vector<double> >::iterator it = wordEmbedding.find(probWordVec[i].second);
        if( it != endIt)
        {
            vector<double> tt = it->second ;
            for(int jj = 0 ;jj < W2VecDimSize ;jj++)
                avgDoc[jj] +=  tt[jj] ;

        }
        //else: cerr<<"fix me11!\n";//Fix Me!!!!!!!!!!!!

    }

    for(int i=0 ;i < W2VecDimSize ; i++)
    {
        //cout<<avgDoc[i]<<" ";
        avgDoc[i] /= wordCount;
        //cout<<avgDoc[i]<<"\n";
    }

    delete fblm;
    delete dCounter;
    delete[] distQuery;
    delete[] distQueryEst;

    return avgDoc;

}
float lemur::retrieval::RetMethod::computeProfDocSim(lemur::api::TextQueryRep *textQR,int docID ,
                                                     vector<int> relJudgDoc ,vector<int> nonReljudgDoc , bool newNonRel , bool newRel)
{
#if 1
    double sc = scoreDoc(*textQR ,docID);
    //cerr<<sc<<" ";
    return sc;
#endif
#if 0
    vector<double> newDocAvgKeyWordsVec = docIdKeyWords[docID];//extractKeyWord(docID) ;//;
    return cosineSim(Vq,newDocAvgKeyWordsVec);

#endif
#if 0
    vector<vector<double> > docTerms;
    double counter = 0.0 ;

    TermInfoList *docTermInfoList =  ind.termInfoList(docID);
    docTermInfoList->startIteration();
    const std::map<int,vector<double> >::iterator endIt = wordEmbedding.end();
    while(docTermInfoList->hasMore())
    {

        TermInfo *ti = docTermInfoList->nextEntry();
        const std::map<int,vector<double> >::iterator itt = wordEmbedding.find(ti->termID());
        if(itt != endIt)
        {
            counter += 1;
            docTerms.push_back(itt->second);
        }
        else
            continue;

        //delete ti;
    }
    delete docTermInfoList;


    vector<double>docAvg (W2VecDimSize,0.0);
    for(int i =0 ; i< docTerms.size() ; i++)
    {
        for(int j = 0 ;j < W2VecDimSize/*docTerms[i].size()*/ ; j++)
            docAvg[j] += docTerms[i][j];
    }
    //cout<<docAvg[0]<<" "<<counter<<endl;
    for(int i = 0 ; i < docAvg.size() ;i++)
        docAvg[i] /= (double)(docTerms.size());
    //cout<<docAvg[0]<<endl;


    return cosineSim(Vq , docAvg);
#endif

#if 0
    IndexedRealVector nonRel,rel;
    for (int i =0 ; i<nonReljudgDoc.size() ; i++)
    {
        nonRel.PushValue(nonReljudgDoc[i],0);
    }
    PseudoFBDocs  *nonRelDocs;
    nonRelDocs= new PseudoFBDocs(nonRel,-1,true);

    for(int i =0 ; i< relJudgDoc.size();i++)
        rel.PushValue(relJudgDoc[i],0);
    PseudoFBDocs  *relDocs;
    relDocs= new PseudoFBDocs(rel,-1,true);

    const QueryModel *qm = dynamic_cast<const QueryModel *>(textQR);
    //cout<<"positive score"<<endl;
    double sc = 0;
    DocumentRep *dRep;
    HashFreqVector hfv(ind,docID);

    dRep = computeDocRep(docID);
    textQR->startIteration();
    while (textQR->hasMore())
    {
        QueryTerm *qTerm = textQR->nextTerm();
        if(qTerm->id()==0)
        {
            cerr<<"**********"<<endl;
            //break;
            continue;
        }

        int tf;
        hfv.find(qTerm->id(),tf);
        DocInfo *info = new DocInfo(docID,tf);


        sc += scoreFunc()->matchedTermWeight(qTerm, textQR, info, dRep);//QL = sc+=|q|*log( p_seen(w|d)/(a(d)*p(w|C)) ) [slide7-11]
        //cout<<ind.term(qTerm->id())<<": "<<scoreFunc()->matchedTermWeight(qTerm, textQR, info, dRep)<<endl;
        delete info;
        delete qTerm;
    }

    double negQueryGenerationScore=0.0;
    //cout<<"negative score:"<<endl;
    if(RSMethodHM == 1)//RecSys(neg,coll)
    {
        negQueryGenerationScore= qm->negativeQueryGeneration(dRep ,nonReljudgDoc ,relJudgDoc,negGenModeHM, newNonRel,newRel,NegMu,delta,lambda_1,lambda_2);
    }
    else if (RSMethodHM == 2 || RSMethodHM == 3)//RecSys negKLQTE(2) and negKL(3)
    {
        negQueryGenerationScore = qm->negativeKL(dRep ,nonReljudgDoc , newNonRel,NegMu);
    }
    /*else if (RSMethodHM == 4)//fang
            {
                negQueryGenerationScore = fangScore(*nonRelDocs,docID,newNonRel);
            //    cout<<"inja5"<<endl;
            }*/

    //double fangScoreTmp = fangScore(*relDocs,docID,newRel);//considering positive feedback
    //negQueryGenerationScore -= fangScore(*relDocs,docID,newRel);//considering positive feedback


    double adjustedScore = scoreFunc()->adjustedScore(sc, textQR, dRep);
    //cout<<"negqueryScore: "<<negQueryGenerationScore<<endl;

    //cout<<"fangScoreTmp: "<< fangScoreTmp<<" negqueryScore: "<<negQueryGenerationScore<<" adjusted: "<<adjustedScore<<" newRel" <<newRel<<endl;
    //negQueryGenerationScore -= fangScoreTmp;

    //cout <<"inja6"<<endl;
    delete dRep;
    delete nonRelDocs;
    delete relDocs;
    // cout<<"neg score:**********:"<<negQueryGenerationScore<<endl;
    //cout<<"pos score:**********:"<<adjustedScore<<endl;
    return (negQueryGenerationScore + adjustedScore);

#endif

}


void lemur::retrieval::RetMethod::updateTextQuery(TextQueryRep &origRep,
                                                  const DocIDSet &relDocs,const DocIDSet &nonRelDocs )
{
    //cout<<"RM: "<<RM<<endl;
    QueryModel *qr;

    qr = dynamic_cast<QueryModel *> (&origRep);

    if(RM=="RM1"){
        computeRM1FBModel(*qr, relDocs,nonRelDocs);
        return;
    }else if(RM=="RM2"){
        computeRM2FBModel(*qr, relDocs);
        return;
    }else if(RM=="RM3"){
        computeRM3FBModel(*qr, relDocs);
        return;
    }else if(RM=="RM4"){
        computeRM4FBModel(*qr, relDocs);
        return;
    }else if(RM=="MIX"){
        computeMixtureFBModel(*qr, relDocs,nonRelDocs);
        return;
    }else if(RM=="DIVMIN"){
        computeDivMinFBModel(*qr, relDocs);
        return;
    }else if(RM=="MEDMM"){
        computeMEDMMFBModel(*qr, relDocs);
        return;
    }




    switch (qryParam.fbMethod) {
    case RetParameter::MIXTURE:
        computeMixtureFBModel(*qr, relDocs,nonRelDocs);
        break;
    case RetParameter::DIVMIN:
        computeDivMinFBModel(*qr, relDocs);
        break;
    case RetParameter::MARKOVCHAIN:
        computeMarkovChainFBModel(*qr, relDocs);
        break;
    case RetParameter::RM1:
        computeRM1FBModel(*qr, relDocs,nonRelDocs);
        break;
    case RetParameter::RM2:
        computeRM2FBModel(*qr, relDocs);
        break;
    default:
        throw Exception("SimpleKLRetMethod", "unknown feedback method");
        break;
    }
}
void lemur::retrieval::RetMethod::computeNearestTerm2Vec(vector<double> vec )
{
    ofstream inputfile;
    inputfile.open("outputfiles/NearestTerm2Vec.txt",ios::app);

    vector<pair<double,int> >simTermid;

    for(int i =1 ; i< ind.termCountUnique() ; i++)
    {

        vector<double>dtemp;

        if(wordEmbedding.find(i) != wordEmbedding.end()) //FIX ME!!!!
            //if(wordEmbedding[i].size() != 0)
        {
            dtemp.assign(wordEmbedding[i].begin() ,wordEmbedding[i].end());
        }
        else
        {
            continue;
        }

        if(vec.size() != dtemp.size())
            cerr<<vec.size() << " "<<dtemp.size()<<" "<<ind.term(i)<<" "<<wordEmbedding[i].size()<<endl;

        double sim = this->cosineSim(vec,dtemp);
        simTermid.push_back(pair<double,int>(sim,i) );
    }
    //cerr<<"simterm: "<<simTermid.size()<<endl;
    std::sort(simTermid.begin(), simTermid.end(), pairCompare);


    for(int i = 0 ; i < 15 ; i++)
        inputfile <<"( "<< ind.term(simTermid[i].second)<<" , "<<simTermid[i].first<<" ) ";

    inputfile << endl << endl;
    inputfile.close();

    //cerr<<"end simterm: "<<simTermid.size()<<endl;
}

void lemur::retrieval::RetMethod::multiplyVec2Vec(vector<double> m1, vector<vector<double> >&res )
{
    //cout<< endl<<W2VecDimSize<<endl;
    for(int i = 0 ;i < W2VecDimSize ;i++)
        for(int j = 0 ; j < W2VecDimSize ; j++)
            res[i][j] += m1[i] * Vq[j];

}
void lemur::retrieval::RetMethod::multiplyMatrix2Vec(vector<double>&res  )
{
    //(N*N)^T * (1*N)^T
    for(int j = 0 ; j < W2VecDimSize ; j++)
        for(int k = 0 ; k < W2VecDimSize ; k++)
        {
            //cerr<<"j "<<j <<" k "<<k<<" m1 "<<m1[k][j]<<" m2 "<<m2[k]<<endl;
            res[j] += coefMatrix[k][j] * Vq[k];
        }
    //cout<<res[50]<<endl;
}

void lemur::retrieval::RetMethod::computeCoefMatrix()
{

    return;

    int dim = W2VecDimSize;
    double eps = 0.0000000001;
    double sum = eps;

    //qryParam.fbCoeff =fbcoef;

    //set<int> pos_terms;
    //set<int> neg_terms;

    long double W[dim][dim]; // Feedback Matrix
    long double WT[dim][dim]; // Transformation Matrix
    long double delta_W[dim][dim];
    double etha = 0.0001;

    /*double alpha = 0.8;//we_alpha;
    double lambda = 0.05;//we_lambda;
    double beta = 0.05;//we_beta;*/

    double alpha = 0.1;//we_alpha;
    double lambda = 0.1;//we_lambda;
    double beta = 0.1;//we_beta;

    int n_iter = 500;
    long double change_prev = 100.0;
    long double change_now = 1.0;
    for(int i=0;i<dim;i++)
    {
        for(int j=0;j<dim;j++)
        {
            //double f = (double) (rand()/RAND_MAX)-0.5;
            W[i][j] = 0.f;//0.f;

            WT[j][i] = 0.f;//0.f;//W[y][x];
            delta_W[i][j] = 0.f;
        }
    }

    //cerr<<"inji 1\n";
    while(abs(change_now)>0.000001 && n_iter != 0)
    {
        change_prev = change_now;
        change_now = 0.0;
        etha = 0.0001; //* (1+n_iter);
        //etha = 0.000001 * (n_iter);
        n_iter--;

        // positive samples

        //cerr<<"POOOOS: "<<pos.size()<<endl;
        for(int i=0;i<pos.size();i++)
        {
            int wn = pos[i].first;

            vector<double> w2vVec(dim,0.0);
            map<int,vector<double> >::iterator fit =  wordEmbedding.find(wn);
            if( fit == wordEmbedding.end())//not found!
                continue;
            else
                w2vVec.assign(fit->second.begin(),fit->second.end());

            //cerr<<w2vVec[10]<<" "<<wordEmbedding[wn][10]<<"\n";

            //pos_terms.insert(wn);
            //double alpha_w = pos[i].second;
            long double Wvq[dim];
            double Wvq_vwn[dim];
            for(int x=0;x<dim;x++)
            {
                //cerr<<"inji 2-0 "<<i<<"\n";
                Wvq[x]=0.0;
                for(int y=0;y<dim;y++)
                {
                    Wvq[x] += WT[x][y] * Vq[y];
                }

                Wvq_vwn[x] = Wvq[x]- w2vVec[x];

            }
            for(int x=0;x<dim;x++)
            {
                //cerr<<"inji 2-1"<<i<<"\n";
                for(int y=0;y<dim;y++)
                {
                    double mul = etha*(alpha*Wvq_vwn[x]*Vq[y]);
                    delta_W[x][y] += mul;//etha*(alpha*Wvq_vwn[x]*Vq[y]);
                    W[x][y] -= mul;//etha*(alpha*Wvq_vwn[x]*Vq[y]); //FIXME
                }
            }
        }

        // negative samples
        //cerr<<"NEEEEG: "<<neg.size()<<endl;
        for(int i=0;i<neg.size();i++)
        {
            int wn_ = neg[i].first;
            vector<double> w2vVec(dim,0.0);
            map<int,vector<double> >::iterator fit =  wordEmbedding.find(wn_);
            if( fit == wordEmbedding.end())//not found!
                continue;
            else
                w2vVec.assign(fit->second.begin(),fit->second.end());

            //neg_terms.insert(wn_);
            //double lambda_w = neg[i].second;
            long double Wvq[dim];
            double Wvq_vwn_[dim];
            for(int x=0;x<dim;x++)
            {
                Wvq[x]=0.0;
                for(int y=0;y<dim;y++)
                {
                    Wvq[x] +=WT[x][y]*Vq[y];
                }
                Wvq_vwn_[x] = Wvq[x] - w2vVec[x];
            }
            for(int x=0;x<dim;x++)
            {
                for(int y=0;y<dim;y++)
                {
                    double mul = etha*(-lambda*Wvq_vwn_[x]*Vq[y]);
                    delta_W[x][y] += mul;//etha*(-lambda*Wvq_vwn_[x]*Vq[y]);
                    W[x][y] -= mul;//etha*(-lambda*Wvq_vwn_[x]*Vq[y]);// FIXME
                }
            }
        }

        // regularization
        for (int x=0; x < dim; x++)
            for (int y = 0; y < dim; y++)
            {
                double mul = etha*(-beta*W[x][y]);
                delta_W[x][y] += mul;//delta_W[x][y] -= etha*(beta*W[x][y]);
                W[x][y] -= mul; //-etha*(beta*W[x][y]); // FIXME
            }

        // update W
        for (int x = 0; x < dim; x++)
        {
            for (int y = 0; y < dim; y++)
            {
                //W[x][y] -= delta_W[x][y];//FIXME
                change_now += delta_W[x][y]*delta_W[x][y];//pow(delta_W[x][y],2);
            }
        }
        //cerr<<"chang b: "<<change_now<<"\n";
        change_now = sqrt(change_now);//pow(change_now,1.0/4o);


        for(int x=0;x<dim;x++)
        {
            for(int y=0;y<dim;y++)
            {
                WT[x][y] = W[y][x];
                delta_W[x][y] = 0.f;
            }
        }


    }
    cerr<<"chang a: "<<change_now<<"\n";
    cerr<<"(O o)\n"<<endl;

    sum=0.0;
    vector<double> WTvq(dim,0.0);

    //long double mu[dim]; // Feedback Matrix
    //WTvq.resize(dim);

    for(int x=0;x<dim;x++)
    {
        double uv = 0;
        for(int y=0;y<dim;y++)
        {
            uv += WT[x][y]*Vq[y] ;

            //cerr << WT[x][y]<<" "<<Vq[y]<<" ";
        }
        //cerr<<"uv: "<<uv<<"\n";

        WTvq[x]= uv;
        Vq[x] = uv;
    }
    //cerr<<endl<<endl;
    //cerr<<"inji 9\n";
}

void lemur::retrieval::RetMethod::computeRelNonRelDist(TextQueryRep &origRep,
                                                       const vector<int> relDocs, const vector<int> nonRelDocs,bool isRelevant, bool computeCoeff)
{
    COUNT_T numTerms = ind.termCountUnique();

    lemur::langmod::DocUnigramCounter *dCounter;
    if(isRelevant)
    {
        //cerr<<"HERE11111 "<<relDocs.size()<<" "<<nonRelDocs.size()<<endl;
        dCounter  = new lemur::langmod::DocUnigramCounter(relDocs, ind);
    }
    else
    {
        //cerr<<"HERE22222 "<<relDocs.size()<<" "<<nonRelDocs.size()<<endl;
        dCounter = new lemur::langmod::DocUnigramCounter(nonRelDocs, ind);
    }

    double *distQuery = new double[numTerms+1];
    double *distQueryEst = new double[numTerms+1];


    double meanLL=1e-40;
    double distQueryNorm=0;

    for (int i=1; i<=numTerms;i++)
    {
        distQueryEst[i] = rand()+0.001;
        distQueryNorm += distQueryEst[i];
    }

    double noisePr = 0.9; //qryParam.fbMixtureNoise; (lambda)
    int itNum = qryParam.emIterations;
    do {
        // re-estimate & compute likelihood
        double ll = 0;

        for (int i=1; i<=numTerms;i++)
        {
            distQuery[i] = distQueryEst[i]/distQueryNorm;
            // cerr << "dist: "<< distQuery[i] << endl;
            distQueryEst[i] =0;
        }

        distQueryNorm = 0;

        // compute likelihood
        dCounter->startIteration();
        while (dCounter->hasMore())
        {
            int wd; //dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);
            ll += wdCt * log (noisePr*collectLM->prob(wd)  // P(w|C)
                              + (1-noisePr)*distQuery[wd]); // P(w|thetha_F)
        }
        meanLL = 0.5*meanLL + 0.5*ll;
        if (fabs((meanLL-ll)/meanLL)< 0.0001)
        {
            //cerr << "converged at "<< qryParam.emIterations - itNum+1  << " with likelihood= "<< ll << endl;
            break;
        }

        // update counts
        dCounter->startIteration();
        while (dCounter->hasMore())
        {
            int wd; // dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);

            double prTopic = (1-noisePr)*distQuery[wd]/
                    ((1-noisePr)*distQuery[wd]+noisePr*collectLM->prob(wd));

            double incVal = wdCt*prTopic;
            distQueryEst[wd] += incVal;
            distQueryNorm += incVal;
        }
    } while (itNum-- > 0);

    /*
    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (int i=1; i<=numTerms; i++)
        if (distQuery[i] > 0)
            lmCounter.incCount(i, distQuery[i]);
    */



    if(isRelevant)
    {
        /*
        origRep.startIteration();
        while (origRep.hasMore())
        {
            QueryTerm *qt = origRep.nextTerm();
            int id = qt->id();
            if(wordEmbedding.find(id) == wordEmbedding.end())
                continue;

            //tt.push_back(id);
            pos.push_back(make_pair(id,1.0));
            delete qt;
        }
        */
        pos.clear();

        vector<pair<int,double> >temp;
        for (int i=1; i<=numTerms; i++)
            if (distQuery[i] > 0)
                temp.push_back(make_pair(i,distQuery[i]));
        //pos.push_back(make_pair(i,distQuery[i]));
        std::sort(temp.begin(),temp.end(),sort_pred());
        pos.assign(temp.begin() , temp.begin() + numberOfPositiveSelectedTopWord );

        //cerr<<"\nPOS "<<pos.size()<<endl;

    }
    else
    {
        neg.clear();

        vector<pair<int,double> >temp;
        for (int i=1; i<=numTerms; i++)
            if (distQuery[i] > 0)
                //neg.push_back(make_pair(i,distQuery[i]));
                temp.push_back(make_pair(i,distQuery[i]));
        std::sort(temp.begin(),temp.end(),sort_pred());
        neg.assign(temp.begin() , temp.begin() + numberOfNegativeSelectedTopWord );
        //neg.assign(temp.begin() , temp.end());

        //cerr<<"\nNEG "<<neg.size()<<endl;
    }




    if(computeCoeff)
    {
        ofstream inputfile;
        inputfile.open("outputfiles/NearestTerm2Vec.txt",ios::app);

        origRep.startIteration();//ehtemalan Fix Meee!!!!!!!!!!!!!!!!!!!!!!!!!
        while(origRep.hasMore())
        {
            int idd = origRep.nextTerm()->id();
            inputfile<<ind.term(idd)<<" ";
        }
        inputfile<<":"<<endl;
        inputfile.close();

        cerr<<"111111111\n";
        computeNearestTerm2Vec(Vq);
        cerr<<"222222222\n";

        computeCoefMatrix();
        cerr<<"3333333333\n";

        computeNearestTerm2Vec(Vq);
    }


    //delete fblm;
    delete dCounter;
    delete[] distQuery;
    delete[] distQueryEst;

}

void lemur::retrieval::RetMethod::genSample(vector<pair<int,double> > &pos,vector<pair<int,double> > &neg,vector<int> &VS,map<int,int> &word_count,const DocIDSet
                                            &relDocs,QueryModel &origRep){

    double sum;
    int feedback_voc_count=0;
    COUNT_T numTerms = ind.termCountUnique();
    lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(relDocs, ind);

    double distQueryNorm=0;
    double *distQueryEst = new double[numTerms+1];
    double *distQuery = new double[numTerms+1];

    double noisePr = qryParam.fbMixtureNoise;
    //double meanLL=1e-40;
    set<int> neg_set;

    vector<pair<int,double> > scores;

    for (int i=1; i<=numTerms;i++) {
        //distQueryEst[i] = 0.0;
        distQueryEst[i] = 0.0;//rand()+0.001;
        distQueryNorm+= distQueryEst[i];
    }

    relDocs.startIteration();
    while (relDocs.hasMore())
    {
        int id;
        double pr;
        relDocs.nextIDInfo(id,pr);
        TermInfoList *tList = ind.termInfoList(id);
        DocModel *dm = dynamic_cast<DocModel *> (computeDocRep(id));
        TermInfo *info;
        tList->startIteration();
        while (tList->hasMore())
        {
            info = tList->nextEntry();
            word_count[info->termID()] += info->count();
            feedback_voc_count +=info->count();
            if(find(VS.begin(),VS.end(),info->termID())==VS.end())
            {
                VS.push_back(info->termID());
            }
            distQuery[info->termID()] += dm->seenProb(info->count(), info->termID());
        }
        delete tList;
        delete dm;
    }

    int count =0;

    // NEGATIVE
    scores.clear();
    for(map<int,int>::iterator it = word_count.begin();it!=word_count.end();it++){
        int i=it->first;
        double s =(collectLM->prob(i));
        scores.push_back(make_pair(i,s));
    }

    std::sort(scores.begin(),scores.end(),sort_pred());
    count=0;
    int count_=0;

    int we_pos_n =  numberOfPositiveSelectedTopWord;
    int we_neg_n = numberOfNegativeSelectedTopWord;//80,we_pos_n =30;/******************/

    map<int,vector<double> >::iterator hend = wordEmbedding.end();
    for(int i=0;i<scores.size();i++)
    {
        int x = scores[i].first;
        double s = scores[i].second;


        //if(wordEmbedding[x].size())
        if( wordEmbedding.find(x) != hend )
        {
            neg.push_back(make_pair(x,s));
            neg_set.insert(x);
            //cerr<<" "<<ind.term(x)<<" "<<word_count[x]<<" "<<collectLM->prob(x)<<endl;
            count++;

        }
        if(count>we_neg_n){
            break;
        }
    }

    sum=0.0;

    // POSITIVE
    scores.clear();
    for(map<int,int>::iterator it = word_count.begin();it!=word_count.end();it++){
        int i = it->first;
        distQuery[i] =word_count[i]*((1-noisePr)*distQuery[i])/((1-noisePr)*distQuery[i]+noisePr*collectLM->prob(i));
        sum+= distQuery[i];
    }

    for (int i=1; i<=numTerms;i++){
        distQuery[i] /= sum;//(1-noisePr)*fbm/((1-noisePr)*fbm+noisePr*collectLM->prob(i));
        if(distQuery[i]>0){
            scores.push_back(make_pair(i,distQuery[i]));
        }
    }
    std::sort(scores.begin(),scores.end(),sort_pred());

    /*
    origRep.startIteration();
    while (origRep.hasMore()) {
        QueryTerm *qt = origRep.nextTerm();
        int id = qt->id();
        if(wordEmbedding[id].size()==0){
            continue;
        }
        pos.push_back(make_pair(id,1.0));
        delete qt;
    }*/
    for(int i=0;i<scores.size();i++)
        if(pos.size()<we_pos_n)
            if( neg_set.find(scores[i].first)==neg_set.end() )
                if(wordEmbedding.find(scores[i].first) != wordEmbedding.end() )//if(wordEmbedding[scores[i].first].size() )
                {
                    int x = scores[i].first;
                    double s = scores[i].second;
                    pos.push_back(make_pair(x,s));
                    //cerr<<" "<<ind.term(x)<<" "<<word_count[x]<<" "<<collectLM->prob(x)<<endl;
                }

    delete[] distQuery;
    delete[] distQueryEst;
    delete dCounter;
}

void lemur::retrieval::RetMethod::computeWEFBModel(QueryModel &origRep,const DocIDSet &relDocs)
{


    //vector<double> q2v;
    double eps = 0.0000000001;
    double sum=eps;
    //map<int,double> feedback_w2v_model;
    map<int,int> word_count;
    int tot_count=0;
    //int feedback_voc_count=0;
    vector<int> VS;
    COUNT_T numTerms = ind.termCountUnique();
    //double distQueryNorm;
    //double noisePr = qryParam.fbMixtureNoise;
    double *distQuery = new double[numTerms+1];
    double *distQueryEst = new double[numTerms+1];


    lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(relDocs, ind);

    vector<pair<int,double> > scores;
    vector<pair<int,double> > pos;
    vector<pair<int,double> > neg;
    set<int> pos_terms;
    set<int> neg_terms;

    for (int i=1; i<=numTerms;i++) {
        distQueryEst[i] = 0.0;//rand()+0.001;
        distQuery[i] = 0.0;//rand()+0.001;
    }


    genSample(pos,neg,VS,word_count,relDocs,origRep);


    for(map<int,int>::iterator it = word_count.begin();it!=word_count.end();it++)
        tot_count +=it->second;

    //cerr<<tot_count<<endl;
    int dim = 100;/********************/
    //int i;

    /*
    int qcnt=0;
    q2v.resize(dim);
    for(int i=0;i<dim;i++)
        q2v[i] = 0.0;


    origRep.startIteration();
    while (origRep.hasMore())
    {
        QueryTerm *qt = origRep.nextTerm();
        int id = qt->id();
        if(wordEmbedding[id].size()==0)
            continue;

        for(int i=0;i<dim;i++)
        {
            q2v[i] += wordEmbedding[id][i];
            //cerr<<q2v[i]<<" "<<qcnt<<endl;
        }
        qcnt++;


        //cerr<<ind.term(id)<<endl;
        delete qt;
    }

    for(int i=0;i<dim;i++){
        q2v[i] /= qcnt;
        //cerr<<q2v[i]<<endl;
    }
    */
    long double W[dim][dim]; // Feedback Matrix
    long double WT[dim][dim]; // Transformation Matrix
    long double delta_W[dim][dim];


    /*double set_alpha[] = {0.8};
    double set_lambda[]={0.05};
    double set_beta[] = {0.05};
    double set_pos_p[] = {30};
    double set_neg_n[] = {100};
    //double set_coef[] = {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
    double set_coef[] = {0.9};*/

    double etha = 0.001;
    //qryParam.fbCoeff = 0.5;/*********************************/
    double alpha = 0.8;
    double lambda = 0.75;
    double beta = 0.05;

    alpha = alphaCoef;
    lambda = lambdaCoef;
    beta = betaCoef;


    /*cerr<<endl<<"NEG:\n";
    for(int i=0;i<neg.size();i++)
        cerr<< ind.term(neg[i].first)<<" "<<neg[i].second<<", ";
    cerr<<endl<<"POS:\n";
    for(int i=0;i<pos.size();i++)
        cerr<< ind.term(pos[i].first)<<" "<<pos[i].second<<", ";
    cerr<<endl;*/

    int n_iter = 100;
    long double change_prev = 100.0;
    long double change_now = 1.0;
    for(int i=0;i<dim;i++)
    {
        for(int j=0;j<dim;j++)
        {
            //double f = (double) rand()/RAND_MAX*1-0.5;
            W[i][j] = 0.f;
        }
    }

    while(abs(change_now)>0.000001 && n_iter != 0)
    {
        change_prev = change_now;
        change_now = 0.0;
        //etha = 0.001; //* (1+n_iter);
        //etha = 0.000001 * (n_iter);
        n_iter--;

        for(int x=0;x<dim;x++){
            for(int y=0;y<dim;y++){
                WT[x][y] =W[y][x];
                delta_W[x][y] = 0.f;
            }
        }

        // positive samples
        map<int, vector<double> >::iterator end_hit = wordEmbedding.end();
        for(int i=0;i<pos.size();i++)
        {
            int wn = pos[i].first;
            pos_terms.insert(wn);
            //double alpha_w = pos[i].second;
            long double Wvq[dim];
            double Wvq_vwn[dim];


            map<int, vector<double> >::iterator hit = wordEmbedding.find(wn);
            if(hit == end_hit)
                continue;
            vector<double> hvec = hit->second;
            for(int x=0;x<dim;x++)
            {
                Wvq[x]=0.0;
                for(int y=0;y<dim;y++)
                {
                    Wvq[x] +=WT[x][y]*Vq[y];
                }
                Wvq_vwn[x] = Wvq[x] - hvec[x];

            }
            for(int x=0;x<dim;x++)
            {
                for(int y=0;y<dim;y++)
                {
                    long double hval = etha*(alpha*Wvq_vwn[x]*Vq[y]);
                    delta_W[x][y] += hval;
                    W[x][y] -= hval; //FIXME
                }
            }
        }

        // negative samples
        for(int i=0;i<neg.size();i++)
        {
            int wn_ = neg[i].first;
            neg_terms.insert(wn_);
            //double lambda_w = neg[i].second;
            long double Wvq[dim];
            double Wvq_vwn_[dim];

            map<int, vector<double> >::iterator hit = wordEmbedding.find(wn_);
            if(hit == end_hit)
                continue;
            vector<double> hvec = hit->second;

            for(int x=0;x<dim;x++){
                Wvq[x]=0.0;
                for(int y=0;y<dim;y++){
                    Wvq[x] += WT[x][y] * Vq[y];
                }
                Wvq_vwn_[x] = Wvq[x] - hvec[x];
            }
            for(int x=0;x<dim;x++){
                for(int y=0;y<dim;y++)
                {
                    long double hval = etha*(-lambda * Wvq_vwn_[x] * Vq[y]);
                    delta_W[x][y] += hval;
                    W[x][y] -= hval;// FIXME
                }
            }
        }

        // regularization
        for (int x=0; x < dim; x++)
            for (int y = 0; y < dim; y++){
                delta_W[x][y] -= etha*(beta*W[x][y]);
                W[x][y] -= -etha*(beta*W[x][y]); // FIXME
            }

        // update W
        for (int x = 0; x < dim; x++){
            for (int y = 0; y < dim; y++){
                //W[x][y] -= delta_W[x][y];//FIXME
                //change_now += pow(delta_W[x][y],2);
                change_now += delta_W[x][y] * delta_W[x][y];
            }
        }
        change_now = sqrt(change_now);//pow(change_now,1.0/4o);
        //cerr<<" # "<<change_now<<" ";
    }
    //cerr<<"(O o)"<<endl;

    sum=0.0;
    vector<double> WTvq;

    //long double mu[dim]; // Feedback Matrix
    WTvq.resize(dim);
    for(int x=0;x<dim;x++){
        double uv = 0;
        for(int y=0;y<dim;y++){
            uv += WT[x][y]*Vq[y] ;
        }
        WTvq[x]= uv;

        Vq[x]=uv;/**************************/
    }

    map<int , vector<double> >::iterator hend = wordEmbedding.end();
    map<int , vector<double> >::iterator hf;
    for(int id=0;id<VS.size();id++)
    {
        hf = wordEmbedding.find(VS[id]);
        vector<double> vec;
        if( hf != hend)
            vec = hf->second;//wordEmbedding[VS[id]];//fix me
        else
        {
            //cerr<<"DARIM MAGEEEEEEE222";
            continue;
        }
        int wn = VS[id];

        /*if(vec.size()==0){
            cerr<<"DARIM MAGEEEEEEE";
            continue;
        }*/
        double sim =0.0;
        double euc =0.0;
        double norm1=0.0;
        for(int i=0;i<dim;i++){
            //euc += pow(vec[i]-WTvq[i],2);
            euc += (vec[i]-WTvq[i])*(vec[i]-WTvq[i]);
            //euc += pow(vec[i]-q2v[i],2);
        }
        for(int i=0;i<dim;i++){
            norm1+= vec[i]*vec[i];
        }

        double norm2=0.0;
        for(int i=0;i<dim;i++){
            norm2+= WTvq[i] * WTvq[i];
            //norm2+= pow(q2v[i],2);
        }

        for(int i=0;i<dim;i++){
            sim += vec[i]*WTvq[i];
            //sim += vec[i]*q2v[i];
        }

        //double prTopic =log(6.0/sqrt(2.0*M_PI)*exp(-pow((double)sim/(sqrt(norm1)*sqrt(norm2)),2)/0.1)) ;//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //double prTopic =log(1.0+-pow((double)sim/(sqrt(norm1)*sqrt(norm2)),2));//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //double prTopic = sqrt(1+word_count[wn])*exp(1/(1+exp(-sim)));//sqrt(euc);//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //double prTopic = exp(1.0/(1.0+exp(-sim)));//sqrt(euc);//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        double prTopic = word_count[wn] * exp(sim/(sqrt(norm1)*sqrt(norm2)));//sqrt(euc);//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/

        //cerr<<prTopic<<endl;
        //double prTopic = exp(sim/(sqrt(norm1)*sqrt(norm2)));//sqrt(euc);//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //((1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))+noisePr*collectLM->prob(wn));

        //double p_w_f = (double)word_count[wn]/(double)tot_count;
        //long double s = (double)sim/(sqrt(norm1)*sqrt(norm2));//- 0.1 * (collectLM->prob(wn)));//*exp(prTopic);//(log(1.0+sim/(sqrt(norm1)*sqrt(norm2))))

        //long double s =((1.0-noisePr)*p_w_f/((1-noisePr)*(p_w_f)+noisePr*collectLM->prob(wn)))*prTopic;//- 0.1 * (collectLM->prob(wn)));//*exp(prTopic);//(log(1.0+sim/(sqrt(norm1)*sqrt(norm2))))
        //cerr<<s<<endl;
        if(pos_terms.find(wn)!=pos_terms.end()){
            //if(neg_terms.find(wn)==neg_terms.end()){
            scores.push_back(make_pair(wn,prTopic));
            distQuery[wn] += prTopic;
            sum+=distQuery[wn];
        }
    }

    std::sort(scores.begin(),scores.end(),sort_pred());

    for(int i=0;i<scores.size();i++){
        int wn;
        scores[i].second /=sum;
        distQuery[wn] /=sum;
    }
    //std::sort(scores.begin(),scores.end(),sort_pred());
    /*ofstream write ("expanded-terms.txt", fstream::app);
    for (int i=0; i<qryParam.fbTermCount;i++) {
        //for (int i=0; i<scores.size();i++) {
        //for (int i=0; i<100;i++) {
        write<<ind.term(scores[i].first)<<":"<<scores[i].second<<" ";
    }
    write<<endl<<endl;
    write.close();*/


    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (int i=1; i<=numTerms; i++)
    {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    dCounter->startIteration();
    double counter =0;
    while (dCounter->hasMore()){
        int wd; //dmf FIXME
        double wdCt;
        dCounter->nextCount(wd, wdCt);
        counter+=distQuery[wd];
    }

    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
    delete dCounter;
    delete[] distQuery;
    delete[] distQueryEst;
}


void lemur::retrieval::RetMethod::updateThreshold(lemur::api::TextQueryRep &origRep,
                                                  vector<int> relJudgDoc ,vector<int> nonReljudgDoc , int mode,double relSumScores ,double nonRelSumScore)
{
    thresholdUpdatingMethod = updatingThresholdMode;
    //double alpha = 0.3,beta = 0.9;
    if(thresholdUpdatingMethod == 0)//no updating
        return;
    else if(thresholdUpdatingMethod == 1)//linear
    {
        if(mode == 0)//non rel passed
        {
            setThreshold(getThreshold()+getC1());
            //cout<<"mode 0 "<<getThreshold()<<endl;
        }
        else if(mode == 1)//not showed anything
        {
            setThreshold( getThreshold()- getC2() );
            //cout<<"mode 1 "<<getThreshold()<<endl;
        }

        //threshold = -4.5;
    }else if (thresholdUpdatingMethod == 2)//diff rel nonrel method
    {
        double alpha = getDiffThrUpdatingParam();
        double relSize = relJudgDoc.size();
        double nonRelSize = nonReljudgDoc.size();
        double val =  alpha * std::max( ((relSumScores/(relSize+1)+0.005) - (nonRelSumScore/(nonRelSize+1)+0.005)) ,-3.5) *
                (std::abs(std::log10( (nonRelSize+1) / (relSize+1) ) + 0.005 ) );


        if(mode == 0)
            setThreshold(getThreshold() + val);
        else
            setThreshold(getThreshold() - val);
        //cout<<relSumScores<<" "<<relSize<<endl;
        //cout<<alpha<<" "<<(relSumScores/(relSize+1.0))<<" "<<(nonRelSumScore/nonRelSize+1)<<" "<<(std::log10( (nonRelSize+1) / (relSize+1) ) + 0.005 )<<endl;
        cout <<"mode "<<mode<<" alpha "<<alpha <<" relSum: "<<(relSumScores/(relSize+1)+0.005)<<" nonRelSum: "<< (nonRelSumScore/(nonRelSize+1)+0.005) <<" val: "<<val<<" log: "<<std::log10( (nonRelSize+1) / (relSize+1) );
        cout<<" thr: "<<getThreshold()<<endl;
    }else if (thresholdUpdatingMethod == 3) //BAUTO
    {
        /*
        if(firstTime)
        {
            if(mode == 0)//non rel passed
            {
                this->setC1( 2*getC1() );
                setThreshold(getThreshold()+getC1());
                //cout<<"mode 0 "<<getThreshold()<<endl;
            }
            else if(mode == 1)//not showed anything
            {
                setThreshold( getThreshold()- getC1() );
                //cout<<"mode 1 "<<getThreshold()<<endl;
            }
        }else
        {
            //FIX ME
        }
*/
    }


}


void lemur::retrieval::RetMethod::computeMixtureFBModel(QueryModel &origRep,
                                                        const DocIDSet &relDocs, const DocIDSet &nonRelDocs )
{
    COUNT_T numTerms = ind.termCountUnique();

    lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    double *distQueryEst = new double[numTerms+1];

    double noisePr;

    int i;

    double meanLL=1e-40;
    double distQueryNorm=0;

    for (i=1; i<=numTerms;i++) {
        distQueryEst[i] = rand()+0.001;
        distQueryNorm+= distQueryEst[i];
    }
    noisePr = qryParam.fbMixtureNoise;

    int itNum = qryParam.emIterations;
    do {
        // re-estimate & compute likelihood
        double ll = 0;

        for (i=1; i<=numTerms;i++) {

            distQuery[i] = distQueryEst[i]/distQueryNorm;
            // cerr << "dist: "<< distQuery[i] << endl;
            distQueryEst[i] =0;
        }

        distQueryNorm = 0;

        // compute likelihood
        dCounter->startIteration();
        while (dCounter->hasMore()) {
            int wd; //dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);
            ll += wdCt * log (noisePr*collectLM->prob(wd)  // Pc(w)
                              + (1-noisePr)*distQuery[wd]); // Pq(w)
        }
        meanLL = 0.5*meanLL + 0.5*ll;
        if (fabs((meanLL-ll)/meanLL)< 0.0001) {
            //cerr << "converged at "<< qryParam.emIterations - itNum+1
            //   << " with likelihood= "<< ll << endl;
            break;
        }

        // update counts

        dCounter->startIteration();
        while (dCounter->hasMore()) {
            int wd; // dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);

            double prTopic = (1-noisePr)*distQuery[wd]/
                    ((1-noisePr)*distQuery[wd]+noisePr*collectLM->prob(wd));

            double incVal = wdCt*prTopic;
            distQueryEst[wd] += incVal;
            distQueryNorm += incVal;
        }
    } while (itNum-- > 0);

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
    delete dCounter;
    delete[] distQuery;
    delete[] distQueryEst;

}


void lemur::retrieval::RetMethod::computeDivMinFBModel(QueryModel &origRep,
                                                       const DocIDSet &relDocs)
{
    COUNT_T numTerms = ind.termCountUnique();

    double * ct = new double[numTerms+1];

    TERMID_T i;
    for (i=1; i<=numTerms; i++) ct[i]=0;

    COUNT_T actualDocCount=0;
    relDocs.startIteration();
    while (relDocs.hasMore()) {
        actualDocCount++;
        int id;
        double pr;
        relDocs.nextIDInfo(id,pr);
        DocModel *dm;
        dm = dynamic_cast<DocModel *> (computeDocRep(id));

        for (i=1; i<=numTerms; i++) { // pretend every word is unseen
            ct[i] += log(dm->unseenCoeff()*collectLM->prob(i));
        }

        TermInfoList *tList = ind.termInfoList(id);
        TermInfo *info;
        tList->startIteration();
        while (tList->hasMore()) {
            info = tList->nextEntry();
            ct[info->termID()] += log(dm->seenProb(info->count(), info->termID())/
                                      (dm->unseenCoeff()*collectLM->prob(info->termID())));
        }
        delete tList;
        delete dm;
    }
    if (actualDocCount==0) return;

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);

    double norm = 1.0/(double)actualDocCount;
    for (i=1; i<=numTerms; i++) {
        lmCounter.incCount(i,
                           exp((ct[i]*norm -
                                qryParam.fbMixtureNoise*log(collectLM->prob(i)))
                               / (1.0-qryParam.fbMixtureNoise)));
    }
    delete [] ct;


    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
}
void lemur::retrieval::RetMethod::computeMEDMMFBModel(QueryModel &origRep,
                                                      const DocIDSet &relDocs)
{
    //cout<<"MEDMM\n";
    COUNT_T numTerms = ind.termCountUnique();
    set <int> fbterms;
    double * ct = new double[numTerms+1];
    vector<pair<int,double> > scores;
    //qryParam.fbCoeff =fbcoef;

    TERMID_T i;
    for (i=1; i<=numTerms; i++) ct[i]=0;

    COUNT_T actualDocCount=0;
    int Vf = 0;
    relDocs.startIteration();
    while (relDocs.hasMore()){
        int id;
        double pr;
        relDocs.nextIDInfo(id, pr);
        TermInfoList *tList = ind.termInfoList(id);
        Vf += tList->size();
        delete tList;
    }


    relDocs.startIteration();
    double sum_alpha = 0;
    vector <double> alpha_d;
    while (relDocs.hasMore()) {
        alpha_d.push_back(1);
        actualDocCount++;
        int id;
        double pr;
        relDocs.nextIDInfo(id,pr);
        DocModel *dm;
        dm = dynamic_cast<DocModel *> (computeDocRep(id));


        origRep.startIteration();
        while (origRep.hasMore()) {
            QueryTerm *qt = origRep.nextTerm();
            TermInfo *info;
            TermInfoList *tList = ind.termInfoList(id);
            tList->startIteration();
            bool flag = true;
            while (tList->hasMore()) {
                info = tList->nextEntry();
                if (info->termID() == qt->id()){
                    alpha_d[alpha_d.size()-1] *= (info->count()+0.1)/(ind.docLength(id)+0.1*Vf);
                    flag = false;
                    break;
                }
            }
            if (flag){
                alpha_d[alpha_d.size()-1] *= (0.1)/(ind.docLength(id)+0.1*Vf);
            }
            delete tList;
            delete qt;
        }
        sum_alpha += (alpha_d[alpha_d.size()-1]);
        delete dm;
    }
    relDocs.startIteration();
    int doc_id = 0;
    while (relDocs.hasMore()){
        int id;
        double pr;
        relDocs.nextIDInfo(id, pr);
        DocModel *dm;
        dm = dynamic_cast<DocModel*> (computeDocRep(id));
        //cerr<<"++++"<<exp(alpha_d[doc_id])/sum_alpha<<endl;
        for (i=1; i<=numTerms; i++) { // pretend every word is unseen
            ct[i] += (((alpha_d[doc_id])/sum_alpha)*log(dm->unseenCoeff()*collectLM->prob(i)));
        }

        TermInfoList *tList = ind.termInfoList(id);
        TermInfo *info;
        tList->startIteration();
        while (tList->hasMore()) {
            info = tList->nextEntry();
            ct[info->termID()] += (((alpha_d[doc_id])/sum_alpha)*log(((info->count()+0.1)/(ind.docLength(id)+0.1*Vf))/
                                                                     (dm->unseenCoeff()*collectLM->prob(info->termID()))));
            fbterms.insert(info->termID());
        }
        delete tList;
        delete dm;
        doc_id++;
    }
    if (actualDocCount==0) return;
    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    double beta =1.2;
    double lambda = 0.1;
    double norm = 1.0/(double)actualDocCount;

    for (i=1; i<=numTerms; i++) {
        if(exp((1.0/beta)*(ct[i] -lambda*log(collectLM->prob(i))))>0){
            scores.push_back(make_pair(i,exp((1.0/beta)*(ct[i] - lambda*log(collectLM->prob(i)))) ));
            lmCounter.incCount(i,
                               exp((1.0/beta)*(ct[i] -
                                               lambda*log(collectLM->prob(i)))));
        }
    }
    delete [] ct;
    std::sort(scores.begin(),scores.end(),sort_pred());

    ofstream write ("expanded-terms.txt", fstream::app);
    for(int i=0; i<qryParam.fbTermCount;i++) {
        write<<ind.term(scores[i].first)<<":"<<scores[i].second<<" ";
    }
    write<<endl;
    write.close();



    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
}
void lemur::retrieval::RetMethod::computeMarkovChainFBModel(QueryModel &origRep, const DocIDSet &relDocs)
{
    int stopWordCutoff =50;

    lemur::utility::ArrayCounter<double> *counter = new lemur::utility::ArrayCounter<double>(ind.termCountUnique()+1);

    lemur::langmod::OneStepMarkovChain * mc = new lemur::langmod::OneStepMarkovChain(relDocs, ind, mcNorm,
                                                                                     1-qryParam.fbMixtureNoise);
    origRep.startIteration();
    double summ;
    while (origRep.hasMore()) {
        QueryTerm *qt;
        qt = origRep.nextTerm();
        summ =0;
        mc->startFromWordIteration(qt->id());
        // cout << " +++++++++ "<< ind.term(qt->id()) <<endl;
        TERMID_T fromWd;
        double fromWdPr;

        while (mc->hasMoreFromWord()) {
            mc->nextFromWordProb(fromWd, fromWdPr);
            if (fromWd <= stopWordCutoff) { // a stop word
                continue;
            }
            summ += qt->weight()*fromWdPr*collectLM->prob(fromWd);
            // summ += qt->weight()*fromWdPr;
        }
        if (summ==0) {
            // query term doesn't exist in the feedback documents, skip
            continue;
        }

        mc->startFromWordIteration(qt->id());
        while (mc->hasMoreFromWord()) {
            mc->nextFromWordProb(fromWd, fromWdPr);
            if (fromWd <= stopWordCutoff) { // a stop word
                continue;
            }

            counter->incCount(fromWd,
                              (qt->weight()*fromWdPr*collectLM->prob(fromWd)/summ));
            // counter->incCount(fromWd, (qt->weight()*fromWdPr/summ));

        }
        delete qt;
    }
    delete mc;

    lemur::langmod::UnigramLM *fbLM = new lemur::langmod::MLUnigramLM(*counter, ind.termLexiconID());

    origRep.interpolateWith(*fbLM, 1-qryParam.fbCoeff, qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fbLM;
    delete counter;
}

void lemur::retrieval::RetMethod::computeRM1FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs,const DocIDSet &nonRelDocs)
{
    COUNT_T numTerms = ind.termCountUnique();

    // RelDocUnigramCounter computes SUM(D){P(w|D)*P(D|Q)} for each w
    lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);
    //lemur::langmod::RelDocUnigramCounter *nCounter = new lemur::langmod::RelDocUnigramCounter(nonRelDocs, ind);

    double *distQuery = new double[numTerms+1];
    //double *negDistQuery = new double[numTerms+1];
    double expWeight = qryParam.fbCoeff;

    //double negWeight = 0.5;

    TERMID_T i;
    for (i=1; i<=numTerms;i++){
        distQuery[i] = 0.0;
        //negDistQuery[i] = 0.0;
    }

    double pSum=0.0;
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf FIXME
        double wdPr;
        dCounter->nextCount(wd, wdPr);
        distQuery[wd]=wdPr;
        pSum += wdPr;
    }
    /*double nSum=0.0;
    nCounter->startIteration();
    while (nCounter->hasMore()) {
        int wd; // dmf FIXME
        double wdPr;
        nCounter->nextCount(wd, wdPr);
        negDistQuery[wd]=wdPr;
        nSum += wdPr;
    }*/


    for (i=1; i<=numTerms;i++) {
        //REMOVE  2 *
        //if(feedbackMode == 2)
        {
            //cout<<"normalFB"<<endl;
            distQuery[i] = expWeight*distQuery[i]/pSum +
                    (1-expWeight)*ind.termCount(i)/ind.termCount();

        }/*else if(feedbackMode == 1)
        {
            cout<<"ourFB"<<endl;
            distQuery[i] =  expWeight*(getNegWeight()*(distQuery[i]/pSum)-(1-getNegWeight())*(negDistQuery[i]/nSum) )+
                    (1-expWeight)*ind.termCount(i)/ind.termCount();
        }*/

        lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
        for (i=1; i<=numTerms; i++) {
            if (distQuery[i] > 0) {
                lmCounter.incCount(i, distQuery[i]);
            }
        }

        //cout<<"sum: "<<lmCounter.sum()<<endl;

        lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
        origRep.interpolateWith(*fblm, 0.0, qryParam.fbTermCount,
                                qryParam.fbPrSumTh, 0.0);
        delete fblm;
        delete dCounter;
        //delete nCounter;
        delete[] distQuery;
        //delete[] negDistQuery;
    }
}
void lemur::retrieval::RetMethod::computeRM3FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs)
{
    COUNT_T numTerms = ind.termCountUnique();
    //qryParam.fbCoeff =fbcoef;

    // RelDocUnigramCounter computes SUM(D){P(w|D)*P(D|Q)} for each w
    lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    double expWeight = qryParam.fbCoeff;

    TERMID_T i;
    for (i=1; i<=numTerms;i++)
        distQuery[i] = 0.0;

    double pSum=0.0;
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf FIXME
        double wdPr;
        dCounter->nextCount(wd, wdPr);
        distQuery[wd]=wdPr;
        pSum += wdPr;
    }

    for (i=1; i<=numTerms;i++) {
        distQuery[i] = (expWeight*distQuery[i]/pSum +
                        (1-expWeight)*ind.termCount(i)/ind.termCount());
    }


    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);
    //origRep.interpolateWith(*fblm, 0.0, qryParam.fbTermCount,
    //		qryParam.fbPrSumTh, 0.0);
    delete fblm;
    delete dCounter;
    delete[] distQuery;
}


// out: w.weight = P(w|Q)
// P(w|Q) = k P(w) P(Q|w)
// P(Q|w) = PROD_q P(q|w)
// P(q|w) = SUM_d P(q|d) P(w|d) p(d) / p(w)
// P(w) = SUM_d P(w|d) p(d)
// Promote this to some include somewhere...
struct termProb  {
    TERMID_T id; // TERM_ID
    double prob; // a*tf(w,d)/|d| +(1-a)*tf(w,C)/|C|
};

void lemur::retrieval::RetMethod::computeRM2FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs) {
    COUNT_T numTerms = ind.termCountUnique();
    COUNT_T termCount = ind.termCount();
    double expWeight = qryParam.fbCoeff;

    // RelDocUnigramCounter computes P(w)=SUM(D){P(w|D)*P(D|Q)} for each w
    // P(w) = SUM_d P(w|d) p(d)
    lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    COUNT_T numDocs = ind.docCount();
    vector<termProb> **tProbs = new vector<termProb> *[numDocs + 1];

    int i;
    for (i=1; i<=numTerms;i++)
        distQuery[i] = 0.0;
    for (i = 1; i <= numDocs; i++) {
        tProbs[i] = NULL;
    }

    // Put these in a faster structure.
    vector <TERMID_T> qTerms; // TERM_ID
    origRep.startIteration();
    while (origRep.hasMore()) {
        QueryTerm *qt = origRep.nextTerm();
        qTerms.push_back(qt->id());
        delete(qt);
    }
    COUNT_T numQTerms = qTerms.size();
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf fixme
        double P_w;
        double P_qw=0;
        double P_Q_w = 1.0;
        // P(q|w) = SUM_d P(q|d) P(w|d) p(d)
        dCounter->nextCount(wd, P_w);
        for (int j = 0; j < numQTerms; j++) {
            TERMID_T qtID = qTerms[j]; // TERM_ID
            relDocs.startIteration();
            while (relDocs.hasMore()) {
                int docID;
                double P_d, P_w_d, P_q_d;
                double dlength;
                relDocs.nextIDInfo(docID, P_d);
                dlength  = (double)ind.docLength(docID);
                if (tProbs[docID] == NULL) {
                    vector<termProb> * pList = new vector<termProb>;
                    TermInfoList *tList = ind.termInfoList(docID);
                    TermInfo *t;
                    tList->startIteration();
                    while (tList->hasMore()) {
                        t = tList->nextEntry();
                        termProb prob;
                        prob.id = t->termID();
                        prob.prob = expWeight*t->count()/dlength+
                                (1-expWeight)*ind.termCount(t->termID())/termCount;
                        pList->push_back(prob);
                    }
                    delete(tList);
                    tProbs[docID] = pList;
                }
                vector<termProb> * pList = tProbs[docID];
                P_w_d=0;
                P_q_d=0;
                for (int i = 0; i < pList->size(); i++) {
                    // p(q|d)= a*tf(q,d)/|d|+(1-a)*tf(q,C)/|C|
                    if((*pList)[i].id == qtID)
                        P_q_d = (*pList)[i].prob;

                    // p(w|d)= a*tf(w,d)/|d|+(1-a)*tf(w,C)/|C|
                    if((*pList)[i].id == wd)
                        P_w_d = (*pList)[i].prob;
                    if(P_q_d && P_w_d)
                        break;
                }
                P_qw += P_d*P_w_d*P_q_d;
            }
            // P(Q|w) = PROD_q P(q|w) / p(w)
            P_Q_w *= P_qw/P_w;
        }
        // P(w|Q) = k P(w) P(Q|w), k=1
        distQuery[wd] =P_w*P_Q_w;
    }

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, 0.0, qryParam.fbTermCount,
                            qryParam.fbPrSumTh, 0.0);
    delete fblm;
    delete dCounter;
    for (i = 1; i <= numDocs; i++) {
        delete(tProbs[i]);
    }
    delete[](tProbs);
    delete[] distQuery;
}

void lemur::retrieval::RetMethod::computeRM4FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs)
{
    COUNT_T numTerms = ind.termCountUnique();
    COUNT_T termCount = ind.termCount();
    //qryParam.fbCoeff =fbcoef;
    double expWeight = qryParam.fbCoeff;

    // RelDocUnigramCounter computes P(w)=SUM(D){P(w|D)*P(D|Q)} for each w
    // P(w) = SUM_d P(w|d) p(d)
    lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    COUNT_T numDocs = ind.docCount();
    vector<termProb> **tProbs = new vector<termProb> *[numDocs + 1];

    int i;
    for (i=1; i<=numTerms;i++)
        distQuery[i] = 0.0;
    for (i = 1; i <= numDocs; i++) {
        tProbs[i] = NULL;
    }
    double pSum=0.0;
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf FIXME
        double wdPr;
        dCounter->nextCount(wd, wdPr);
        distQuery[wd]=wdPr;
        pSum += wdPr;
    }
    //for(int i=0;i<numTerms+1;i++)
    //distQuery[i]=distQuery[i]/pSum;

    // Put these in a faster structure.
    vector <TERMID_T> qTerms; // TERM_ID
    origRep.startIteration();
    while (origRep.hasMore()) {
        QueryTerm *qt = origRep.nextTerm();
        qTerms.push_back(qt->id());
        delete(qt);
    }
    COUNT_T numQTerms = qTerms.size();
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf fixme
        double P_w;
        double P_qw=0;
        double P_Q_w = 1.0;
        // P(q|w) = SUM_d P(q|d) P(w|d) p(d)
        dCounter->nextCount(wd, P_w);
        P_w=distQuery[wd];
        //cerr << P_w << endl;
        for (int j = 0; j < numQTerms; j++) {
            TERMID_T qtID = qTerms[j]; // TERM_ID
            relDocs.startIteration();
            while (relDocs.hasMore()) {
                int docID;
                double P_d, P_w_d, P_q_d;
                double dlength;
                relDocs.nextIDInfo(docID, P_d);
                dlength  = (double)ind.docLength(docID);
                if (tProbs[docID] == NULL) {
                    vector<termProb> * pList = new vector<termProb>;
                    TermInfoList *tList = ind.termInfoList(docID);
                    TermInfo *t;
                    tList->startIteration();
                    while (tList->hasMore()) {
                        t = tList->nextEntry();
                        termProb prob;
                        prob.id = t->termID();
                        prob.prob = expWeight*t->count()/dlength+
                                (1-expWeight)*ind.termCount(t->termID())/termCount;
                        pList->push_back(prob);
                    }
                    delete(tList);
                    tProbs[docID] = pList;
                }
                vector<termProb> * pList = tProbs[docID];
                P_w_d=0.0;
                P_q_d=0.0;
                for (int i = 0; i < pList->size(); i++) {
                    // p(q|d)= a*tf(q,d)/|d|+(1-a)*tf(q,C)/|C|
                    if((*pList)[i].id == qtID)
                        P_q_d = (*pList)[i].prob;
                    // p(w|d)= a*tf(w,d)/|d|+(1-a)*tf(w,C)/|C|
                    if((*pList)[i].id == wd)
                        P_w_d = (*pList)[i].prob;
                    if(P_q_d && P_w_d)
                        break;
                }
                //cerr<<"###########"<<P_q_d<<" "<<P_w_d<<endl;
                P_qw += P_d*P_w_d*P_q_d;
            }
            // P(Q|w) = PROD_q P(q|w) / p(w)
            P_Q_w *= P_qw/P_w;
        }
        // P(w|Q) = k P(w) P(Q|w), k=1
        distQuery[wd] = P_w*P_Q_w;
    }

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        //if (distQuery[i] > 0) {
        lmCounter.incCount(i, distQuery[i]);
        //}
    }
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, 1.0-qryParam.fbCoeff, qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
    delete dCounter;
    for (i = 1; i <= numDocs; i++) {
        delete(tProbs[i]);
    }
    delete[](tProbs);
    delete[] distQuery;
}

#if 0
void lemur::retrieval::RetMethod::genSample(vector<pair<int,double> > &pos,vector<pair<int,double> > &neg,vector<int> &VS,
                                            map<int,int> &word_count,const DocIDSet &relDocs,QueryModel &origRep){

    double sum;
    int feedback_voc_count=0;
    COUNT_T numTerms = ind.termCountUnique();
    lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(relDocs, ind);

    double distQueryNorm=0;
    double *distQueryEst = new double[numTerms+1];
    double *distQuery = new double[numTerms+1];

    double noisePr = qryParam.fbMixtureNoise;
    //double meanLL=1e-40;
    set<int> neg_set;

    vector<pair<int,double> > scores;

    for (int i=1; i<=numTerms;i++)
    {
        //distQueryEst[i] = 0.0;
        distQueryEst[i] = 0.0;//rand()+0.001;
        distQueryNorm+= distQueryEst[i];
    }

    relDocs.startIteration();
    while (relDocs.hasMore())
    {
        int id;
        double pr;
        relDocs.nextIDInfo(id,pr);
        TermInfoList *tList = ind.termInfoList(id);
        DocModel *dm = dynamic_cast<DocModel *> (computeDocRep(id));
        TermInfo *info;
        tList->startIteration();
        while (tList->hasMore())
        {
            info = tList->nextEntry();
            word_count[info->termID()] += info->count();
            feedback_voc_count +=info->count();
            if(find(VS.begin(),VS.end(),info->termID())==VS.end())
            {
                VS.push_back(info->termID());
            }
            distQuery[info->termID()] += dm->seenProb(info->count(), info->termID());
        }
        delete tList;
        delete dm;
    }

    //int count =0;

    // NEGATIVE
    /**********************************************************************/
    /*scores.clear();
    for(map<int,int>::iterator it = word_count.begin();it!=word_count.end();it++){
        int i=it->first;
        double s =(collectLM->prob(i));
        scores.push_back(make_pair(i,s));
    }

    std::sort(scores.begin(),scores.end(),sort_pred());
    count=0;
    int count_=0;
    for(int i=0;i<scores.size();i++){
        int x = scores[i].first;
        double s = scores[i].second;
        if(w2v[x].size()){
            neg.push_back(make_pair(x,s));
            neg_set.insert(x);
            //cerr<<" "<<ind.term(x)<<" "<<word_count[x]<<" "<<collectLM->prob(x)<<endl;
            count++;

        }
        if(count>we_neg_n){
            break;
        }
    }*/

    sum=0.0;
    scores.clear();
    for(map<int,int>::iterator it = word_count.begin();it!=word_count.end();it++)
    {
        int i = it->first;
        distQuery[i] = word_count[i]*((1-noisePr)*distQuery[i])/((1-noisePr)*distQuery[i]+noisePr*collectLM->prob(i));
        sum+= distQuery[i];
    }

    for (int i=1; i<=numTerms;i++)
    {
        distQuery[i] /= sum;//(1-noisePr)*fbm/((1-noisePr)*fbm+noisePr*collectLM->prob(i));
        if(distQuery[i]>0)
        {
            scores.push_back(make_pair(i,distQuery[i]));
        }
    }
    std::sort(scores.begin(),scores.end(),sort_pred());
    origRep.startIteration();
    while (origRep.hasMore())
    {
        QueryTerm *qt = origRep.nextTerm();
        int id = qt->id();
        if(w2v[id].size()==0)
        {
            continue;
        }
        pos.push_back(make_pair(id,1.0));
        delete qt;
    }
    for(int i=0;i<scores.size();i++)
        if(w2v[scores[i].first].size() && pos.size()<we_pos_n && neg_set.find(scores[i].first)==neg_set.end())
        {
            int x = scores[i].first;
            double s = scores[i].second;
            pos.push_back(make_pair(x,s));
            //cerr<<" "<<ind.term(x)<<" "<<word_count[x]<<" "<<collectLM->prob(x)<<endl;
        }
    /**************************************************************/

    sum=0.0;
    // POSITIVE
    scores.clear();
    for(map<int,int>::iterator it = word_count.begin();it!=word_count.end();it++)
    {
        int i = it->first;
        distQuery[i] = word_count[i]*((1-noisePr)*distQuery[i])/((1-noisePr)*distQuery[i]+noisePr*collectLM->prob(i));
        sum+= distQuery[i];
    }

    for (int i=1; i<=numTerms;i++)
    {
        distQuery[i] /= sum;//(1-noisePr)*fbm/((1-noisePr)*fbm+noisePr*collectLM->prob(i));
        if(distQuery[i]>0)
        {
            scores.push_back(make_pair(i,distQuery[i]));
        }
    }
    std::sort(scores.begin(),scores.end(),sort_pred());
    origRep.startIteration();
    while (origRep.hasMore())
    {
        QueryTerm *qt = origRep.nextTerm();
        int id = qt->id();
        if(w2v[id].size()==0)
        {
            continue;
        }
        pos.push_back(make_pair(id,1.0));
        delete qt;
    }
    for(int i=0;i<scores.size();i++)
        if(w2v[scores[i].first].size() && pos.size()<we_pos_n && neg_set.find(scores[i].first)==neg_set.end())
        {
            int x = scores[i].first;
            double s = scores[i].second;
            pos.push_back(make_pair(x,s));
            //cerr<<" "<<ind.term(x)<<" "<<word_count[x]<<" "<<collectLM->prob(x)<<endl;
        }


    delete[] distQuery;
    delete[] distQueryEst;
    delete dCounter;
}

void lemur::retrieval::RetMethod::computeWEFBModel(QueryModel &origRep,const DocIDSet &relDocs ,const DocIDSet &relDocs ,bool isRelevant)
{

    vector<double> q2v;
    double eps = 0.0000000001;
    double sum=eps;
    //map<int,double> feedback_w2v_model;
    map<int,int> word_count;
    //int tot_count=0;
    //int feedback_voc_count=0;
    vector<int> VS;//relDocs term ids
    COUNT_T numTerms = ind.termCountUnique();
    //double distQueryNorm;
    //double noisePr = qryParam.fbMixtureNoise;
    double *distQuery = new double[numTerms+1];
    double *distQueryEst = new double[numTerms+1];
    qryParam.fbCoeff =fbcoef;

    lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(relDocs, ind);

    vector<pair<int,double> > scores;
    vector<pair<int,double> > pos;
    vector<pair<int,double> > neg;
    set<int> pos_terms;
    set<int> neg_terms;

    for (int i=1; i<=numTerms;i++)
    {
        distQueryEst[i] = 0.0;//rand()+0.001;
        distQuery[i] = 0.0;//rand()+0.001;
    }


    genSample(pos,neg,VS,word_count,relDocs,origRep,isRelevant);

    //for(map<int,int>::iterator it = word_count.begin();it!=word_count.end();it++)
    //  tot_count +=it->second;

    //cerr<<tot_count<<endl;

    int i;

    /*int qcnt=0;
    q2v.resize(dim);
    for(int i=0;i<dim;i++)
        q2v[i] = 0.0;


    origRep.startIteration();
    while (origRep.hasMore())
    {
        QueryTerm *qt = origRep.nextTerm();
        int id = qt->id();
        if(wordEmbedding[id].size()==0)
            continue;

        for(int i=0;i<dim;i++)
            q2v[i] += wordEmbedding[id][i];

        qcnt++;
        //cerr<<ind.term(id)<<endl;
        delete qt;
    }

    for(int i=0;i<dim;i++)
    {
        q2v[i] /= qcnt;
    }
    */
    for(int ii = 0 ; ii< Vq.size() ;ii++)
        q2v[ii] = Vq[ii];

    long double W[dim][dim]; // Feedback Matrix
    long double WT[dim][dim]; // Transformation Matrix
    long double delta_W[dim][dim];
    double etha = 0.0001;

    double alpha =we_alpha;
    double lambda =we_lambda;
    double beta =we_beta;

    int n_iter = 500;
    long double change_prev = 100.0;
    long double change_now = 1.0;
    for(int i=0;i<dim;i++)
    {
        for(int j=0;j<dim;j++)
        {
            //double f = (double) rand()/RAND_MAX*1-0.5;
            W[i][j] = 0.f;
        }
    }

    while(abs(change_now)>0.000001 && n_iter != 0)
    {
        change_prev = change_now;
        change_now = 0.0;
        etha = 0.0001; //* (1+n_iter);
        //etha = 0.000001 * (n_iter);
        n_iter--;

        for(int x=0;x<dim;x++)
        {
            for(int y=0;y<dim;y++)
            {
                WT[x][y] =W[y][x];
                delta_W[x][y] = 0.f;
            }
        }

        // positive samples
        for(int i=0;i<pos.size();i++)
        {
            int wn = pos[i].first;
            pos_terms.insert(wn);
            //double alpha_w = pos[i].second;
            long double Wvq[dim];
            double Wvq_vwn[dim];
            for(int x=0;x<dim;x++)
            {
                Wvq[x]=0.0;
                for(int y=0;y<dim;y++)
                {
                    Wvq[x] +=WT[x][y]*q2v[y];
                }
                Wvq_vwn[x] = Wvq[x]-w2v[wn][x];
            }
            for(int x=0;x<dim;x++)
            {
                for(int y=0;y<dim;y++)
                {
                    delta_W[x][y] += etha*(alpha*Wvq_vwn[x]*q2v[y]);
                    W[x][y] -= etha*(alpha*Wvq_vwn[x]*q2v[y]); //FIXME
                }
            }
        }

        // negative samples
        for(int i=0;i<neg.size();i++)
        {
            int wn_ = neg[i].first;
            neg_terms.insert(wn_);
            //double lambda_w = neg[i].second;
            long double Wvq[dim];
            double Wvq_vwn_[dim];
            for(int x=0;x<dim;x++)
            {
                Wvq[x]=0.0;
                for(int y=0;y<dim;y++)
                {
                    Wvq[x] +=WT[x][y]*q2v[y];
                }
                Wvq_vwn_[x] = Wvq[x]-w2v[wn_][x];
            }
            for(int x=0;x<dim;x++)
            {
                for(int y=0;y<dim;y++)
                {
                    delta_W[x][y] += etha*(-lambda*Wvq_vwn_[x]*q2v[y]);
                    W[x][y] -= etha*(-lambda*Wvq_vwn_[x]*q2v[y]);// FIXME
                }
            }
        }

        // regularization
        for (int x=0; x < dim; x++)
            for (int y = 0; y < dim; y++)
            {
                delta_W[x][y] -= etha*(beta*W[x][y]);
                W[x][y] -= -etha*(beta*W[x][y]); // FIXME
            }

        // update W
        for (int x = 0; x < dim; x++)
        {
            for (int y = 0; y < dim; y++)
            {
                //W[x][y] -= delta_W[x][y];//FIXME
                change_now += pow(delta_W[x][y],2);
            }
        }
        change_now = sqrt(change_now);//pow(change_now,1.0/4o);
        //cerr<<" # "<<change_now<<" ";
    }
    //cerr<<"(O o)"<<endl;

    sum=0.0;
    vector<double> WTvq;

    //long double mu[dim]; // Feedback Matrix
    WTvq.resize(dim);
    for(int x=0;x<dim;x++)
    {
        double uv = 0;
        for(int y=0;y<dim;y++)
        {
            uv +=WT[x][y]*q2v[y] ;
        }
        WTvq[x]= uv;
    }

    for(int id=0;id<VS.size();id++)
    {
        vector<double> vec = w2v[VS[id]];
        int wn = VS[id];
        if(vec.size()==0){
            continue;
        }
        double sim =0.0;
        double euc =0.0;
        double norm1=0.0;
        for(int i=0;i<dim;i++)
        {
            euc += pow(vec[i]-WTvq[i],2);
            //euc += pow(vec[i]-q2v[i],2);
        }
        for(int i=0;i<dim;i++)
        {
            norm1+= pow(vec[i],2);
        }

        double norm2=0.0;
        for(int i=0;i<dim;i++)
        {
            norm2+= pow(WTvq[i],2);
            //norm2+= pow(q2v[i],2);
        }

        for(int i=0;i<dim;i++)
        {
            sim += vec[i]*WTvq[i];
            //sim += vec[i]*q2v[i];
        }

        //double prTopic =log(6.0/sqrt(2.0*M_PI)*exp(-pow((double)sim/(sqrt(norm1)*sqrt(norm2)),2)/0.1)) ;//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //double prTopic =log(1.0+-pow((double)sim/(sqrt(norm1)*sqrt(norm2)),2));//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //double prTopic = sqrt(1+word_count[wn])*exp(1/(1+exp(-sim)));//sqrt(euc);//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //double prTopic = exp(1.0/(1.0+exp(-sim)));//sqrt(euc);//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        double prTopic = word_count[wn]*exp(sim/(sqrt(norm1)*sqrt(norm2)));//sqrt(euc);//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //double prTopic = exp(sim/(sqrt(norm1)*sqrt(norm2)));//sqrt(euc);//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //((1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))+noisePr*collectLM->prob(wn));

        //double p_w_f = (double)word_count[wn]/(double)tot_count;
        //long double s = (double)sim/(sqrt(norm1)*sqrt(norm2));//- 0.1 * (collectLM->prob(wn)));//*exp(prTopic);//(log(1.0+sim/(sqrt(norm1)*sqrt(norm2))))

        //long double s =((1.0-noisePr)*p_w_f/((1-noisePr)*(p_w_f)+noisePr*collectLM->prob(wn)))*prTopic;//- 0.1 * (collectLM->prob(wn)));//*exp(prTopic);//(log(1.0+sim/(sqrt(norm1)*sqrt(norm2))))
        //cerr<<s<<endl;
        if(pos_terms.find(wn)!=pos_terms.end())
        {
            //if(neg_terms.find(wn)==neg_terms.end()){
            scores.push_back(make_pair(wn,prTopic));
            distQuery[wn] += prTopic;
            sum+=distQuery[wn];
        }
    }

    std::sort(scores.begin(),scores.end(),sort_pred());

    for(int i=0;i<scores.size();i++){
        int wn;
        scores[i].second /=sum;
        distQuery[wn] /=sum;
    }
    //std::sort(scores.begin(),scores.end(),sort_pred());
    //ofstream write ("expanded-terms.txt", fstream::app);
    //for (int i=0; i<qryParam.fbTermCount;i++) {
    //for (int i=0; i<scores.size();i++) {
    //for (int i=0; i<100;i++) {
    //write<<ind.term(scores[i].first)<<":"<<scores[i].second<<" ";
    //}
    //write<<endl;
    //write.close();


    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    dCounter->startIteration();
    double counter =0;
    while (dCounter->hasMore()){
        int wd; //dmf FIXME
        double wdCt;
        dCounter->nextCount(wd, wdCt);
        counter+=distQuery[wd];
    }

    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
    delete dCounter;
    delete[] distQuery;
    delete[] distQueryEst;
}
#endif
