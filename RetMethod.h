/*==========================================================================
 *
 *  Original source copyright (c) 2001, Carnegie Mellon University.
 *  See copyright.cmu for details.
 *  Modifications copyright (c) 2002, University of Massachusetts.
 *  See copyright.umass for details.
 *
 *==========================================================================
 */


//#ifndef _SIMPLEKLRETMETHOD_HPP
//#define _SIMPLEKLRETMETHOD_HPP

#ifndef RetttMethod_H_
#define RetttMethod_H_

#include <cmath>
#include "UnigramLM.hpp"
#include "ScoreFunction.hpp"
#include "DocModel.h"
#include "TextQueryRep.hpp"
#include "TextQueryRetMethod.h"
#include "Counter.hpp"
#include "DocUnigramCounter.hpp"
#include "TermInfoList.hpp"
#include "Parameters.h"
#include <algorithm>

using namespace lemur::api;

extern double negGenMUHM;
extern int RSMethodHM;

namespace lemur
{
namespace retrieval
{

/// Query model representation for the simple KL divergence model

class QueryModel : public ArrayQueryRep {
public:
    /// construct a query model based on query text
    QueryModel(const lemur::api::TermQuery &qry,
               const lemur::api::Index &dbIndex) :
        ArrayQueryRep(dbIndex.termCountUnique()+1, qry, dbIndex), qm(NULL),
        ind(dbIndex), colKLComputed(false) {

        DNsize = 0;

        colQLikelihood = 0;
        colQueryLikelihood();
    }

    /// construct an empty query model
    QueryModel(const lemur::api::Index &dbIndex) :
        ArrayQueryRep(dbIndex.termCountUnique()+1), qm(NULL), ind(dbIndex),
        colKLComputed(false) {
        colQLikelihood = 0;

        DNsize = 0;
        startIteration();
        while (hasMore()) {
            lemur::api::QueryTerm *qt = nextTerm();
            countInNonRel[qt->id()] = 0;
            setCount(qt->id(), 0);
            delete qt;
        }
    }


    virtual ~QueryModel(){ if (qm) delete qm;}


    /// interpolate the model with any (truncated) unigram LM, default parameter  to control the truncation is the number of words
    /*!
        The interpolated model is defined as <tt> origModCoeff</tt>*p(w|original_model)+(1-<tt>origModCoeff</tt>*p(w|new_truncated_model).
        <p> The "new truncated model" gives a positive probability to all words that "survive" in the truncating process, but gives a zero probability to all others.
        So, the sum of all word probabilities according to the truncated model does not
        have to sum to 1. The assumption is that if a word has an extrememly small probability, adding it to the query model will not affect scoring that much. <p> The truncation procedure is as follows:  First, we sort the probabilities in <tt> qModel</tt> passed in, and then iterate over all the entries. For each entry, we check the stopping condition and add the entry to the existing query model if none of the following stopping conditions is satisfied. If, however, any of the conditions is satisfied, the process will terminate. The three stopping conditions are: (1) We already added <tt>howManyWord</tt> words. (2) The total sum of probabilities added exceeds the threshold <tt>prSumThresh</tt>. (3) The probability of the current word is below <tt>prThresh</tt>.
      */

    virtual void interpolateWith(const lemur::langmod::UnigramLM &qModel,
                                 double origModCoeff,
                                 int howManyWord, double prSumThresh=1,
                                 double prThresh=0);
    virtual double scoreConstant() const {
        return totalCount();
    }

    /// load a query model/rep from input stream is
    virtual void load(istream &is);

    /// save a query model/rep to output stream os
    virtual void save(ostream &os);

    /// save a query clarity to output stream os
    virtual void clarity(ostream &os);
    /// compute query clarity score
    virtual double clarity() const;

    /// get and compute if necessary query-collection KL-div (useful for recovering the true divergence value from a score)
    double colDivergence() const {
        if (colKLComputed) {
            return colKL;
        } else {
            colKLComputed = true;
            double d=0;
            startIteration();
            while (hasMore()) {
                lemur::api::QueryTerm *qt=nextTerm();
                double pr = qt->weight()/(double)totalCount();
                double colPr = ((double)ind.termCount(qt->id()) /
                                (double)(ind.termCount())); // ML smoothing
                d += pr*log(pr/colPr);
                delete qt;
            }
            colKL=d;
            return d;
        }
    }

    ///compute the KL-div of the query model and any unigram LM, i.e.,D(Mq|Mref)
    double KLDivergence(const lemur::langmod::UnigramLM &refMod) {
        double d=0;
        startIteration();
        while (hasMore()) {
            lemur::api::QueryTerm *qt=nextTerm();
            double pr = qt->weight()/(double)totalCount();
            d += pr*log(pr/refMod.prob(qt->id()));
            delete qt;
        }
        return d;
    }

    double colQueryLikelihood() const {
        if (colQLikelihood == 0) {
            //Sum w in Q qtf * log(qtcf/termcount);
            lemur::api::COUNT_T tc = ind.termCount();
            startIteration();
            while (hasMore()) {
                lemur::api::QueryTerm *qt = nextTerm();
                lemur::api::TERMID_T id = qt->id();
                double qtf = qt->weight();
                lemur::api::COUNT_T qtcf = ind.termCount(id);
                double s = qtf * log((double)qtcf/(double)tc);
                colQLikelihood += s;
                delete qt;
            }
        }
        return colQLikelihood;
    }

    /*
     * wichMethod: 0 --> baseline collection
     *             1 --> baseline nonRel
    */
    double negativeQueryGeneration( const lemur::api::DocumentRep *dRep, vector<int> JudgDocs,vector<int> relJudgDocs  , int whichMethod , bool newNonRel,bool newRel, double negMu , double delta, double lambda, double lambda_2) const
    {

        if(whichMethod == 0)//baseline(collection)
        {
            double mu= negMu;//ind.docLengthAvg();//negGenMUHM;//2500;
            negQueryGen =0;

            lemur::api::COUNT_T tc = ind.termCount();
            startIteration();
            lemur::utility::HashFreqVector hfv(ind,dRep->getID());
            while (hasMore())
            {
                lemur::api::QueryTerm *qt = nextTerm();
                double pwq = qt->weight()/totalCount();

                int freq=0;
                hfv.find(qt->id(),freq);
                if(freq>0)
                    delta =0.0;

                lemur::api::TERMID_T id = qt->id();
                lemur::api::COUNT_T qtcf = ind.termCount(id);

                double pwc = (double)qtcf/(double)tc;
                double pwdbar = (delta/(delta*ind.termCountUnique()+mu))+((mu*pwc)/(delta*ind.termCountUnique()+mu));
                negQueryGen+= pwq *log(pwq/pwdbar);

                delete qt;
            }
            return negQueryGen;
        }else if (whichMethod == 1)// using DN instead of collection
        {
            if (newNonRel)
                DNsize += ind.docLength(JudgDocs[JudgDocs.size()-1]);

            double mu= negMu;//ind.docLengthAvg();//negGenMUHM;//2500;
            negQueryGen =0;
            lemur::api::COUNT_T tc = ind.termCount();

            lemur::utility::HashFreqVector hfv(ind,dRep->getID()), *hfv2;
            if (newNonRel)
                hfv2 = new lemur::utility::HashFreqVector(ind,JudgDocs[JudgDocs.size()-1]);
            startIteration();
            while (hasMore())
            {
                lemur::api::QueryTerm *qt = nextTerm();
                double pwq = qt->weight()/totalCount();
                if (newNonRel)
                {
                    int freq;
                    hfv2->find(qt->id(),freq);
                    countInNonRel[qt->id()] += freq;
                }
                double cwdbar = 0;
                int freq=0 ;
                hfv.find(qt->id(),freq);
                if(freq>0){
                    cwdbar = 0;
                }
                else
                {
                    cwdbar = countInNonRel[qt->id()];
                }
                lemur::api::TERMID_T id = qt->id();
                lemur::api::COUNT_T qtcf = ind.termCount(id);

                double pwc = (double)qtcf/(double)tc;
                double pwdbar;
                pwdbar = (cwdbar/((DNsize+mu)))+((mu*pwc)/(DNsize+mu));
                negQueryGen+= pwq *log(pwq/pwdbar);
                delete qt;
            }
            if (newNonRel)
                delete hfv2;
            return negQueryGen;
        }
        else if (whichMethod == 2)//Uniform nonrel
        {

            double mu= negMu;
            negQueryGen =0;

            lemur::api::COUNT_T tc = ind.termCount();
            startIteration();
            lemur::utility::HashFreqVector hfv(ind,dRep->getID()), * hfv2;
            if (newNonRel){
                DNsize += ind.docLength(JudgDocs[JudgDocs.size()-1]);
                hfv2 = new lemur::utility::HashFreqVector(ind,JudgDocs[JudgDocs.size()-1]);

                TermInfoList *termList = ind.termInfoList(JudgDocs[JudgDocs.size()-1]);
                termList->startIteration();
                TermInfo *tEntry;
                while (termList->hasMore())
                {
                    tEntry = termList->nextEntry();
                    uniqueNonRel.insert(tEntry->termID());
                }
                delete termList;
            }
            while (hasMore())
            {
                lemur::api::QueryTerm *qt = nextTerm();
                double pwq = qt->weight()/totalCount();

                int freq=0;
                hfv.find(qt->id(),freq);
                if (newNonRel)
                {
                    int freq;
                    hfv2->find(qt->id(),freq);
                    countInNonRel[qt->id()] += freq;

                }
                if(freq>0  ||  countInNonRel[qt->id()]==0)
                    delta =0.0;

                lemur::api::TERMID_T id = qt->id();
                lemur::api::COUNT_T qtcf = ind.termCount(id);

                double pwc = (double)qtcf/(double)tc;
                double pwdbar = (delta/(delta*uniqueNonRel.size()+mu))+((mu*pwc)/(delta*uniqueNonRel.size()+mu));
                negQueryGen+= pwq *log(pwq/pwdbar);

                delete qt;
            }
            if (newNonRel)
                delete hfv2;

            return negQueryGen;

        }
        else if (whichMethod == 3)//smooth nonrel with collection
        {
            //double lambda = 0.5, lambda_2 = 0.5;//2000;
            if (newNonRel)
                DNsize += ind.docLength(JudgDocs[JudgDocs.size()-1]);

            double mu= negMu;//ind.docLengthAvg();//negGenMUHM;//2500;
            negQueryGen =0;
            lemur::api::COUNT_T tc = ind.termCount();

            lemur::utility::HashFreqVector hfv(ind,dRep->getID()), *hfv2;
            if (newNonRel)
                hfv2 = new lemur::utility::HashFreqVector(ind,JudgDocs[JudgDocs.size()-1]);
            startIteration();
            while (hasMore())
            {
                lemur::api::QueryTerm *qt = nextTerm();
                double pwq = qt->weight()/totalCount();
                if (newNonRel)
                {
                    int freq;
                    hfv2->find(qt->id(),freq);
                    countInNonRel[qt->id()] += freq;

                    TermInfoList *termList = ind.termInfoList(JudgDocs[JudgDocs.size()-1]);
                    termList->startIteration();
                    TermInfo *tEntry;
                    while (termList->hasMore())
                    {
                        tEntry = termList->nextEntry();
                        uniqueNonRel.insert(tEntry->termID());
                    }
                    delete termList;

                }
                double cwdbar = 0 , cwdbar_negColl = 1;
                int freq=0 ;
                hfv.find(qt->id(),freq);
                if(freq>0){
                    cwdbar = 0;
                    cwdbar_negColl = 0;
                }
                else
                {
                    //cwdbar = countInNonRel[qt->id()];
                    cwdbar = 1;//smooth uni
                }
                lemur::api::TERMID_T id = qt->id();
                lemur::api::COUNT_T qtcf = ind.termCount(id);

                //double pml_smoothed = (cwdbar/(DNsize+lambda)) + ((lambda * delta)/((DNsize+lambda)*(delta*ind.termCountUnique())));
                double pml_smoothed;
                if (DNsize == 0)
                    pml_smoothed = (1.0 - lambda) * (cwdbar_negColl/ind.termCountUnique());
                else
                    //pml_smoothed = lambda * (cwdbar/DNsize) + (1.0 - lambda) * (cwdbar_negColl/ind.termCountUnique());
                    pml_smoothed = lambda * (cwdbar/uniqueNonRel.size()) + (1.0 - lambda) * (cwdbar_negColl/ind.termCountUnique());

                double pwc = (double)qtcf/(double)tc;
                double pwdbar;
                //pwdbar = ((pml_smoothed*DNsize)/((DNsize+mu)))+((mu*pwc)/(DNsize+mu));
                pwdbar = lambda_2 * pml_smoothed + (1.0 - lambda_2) * pwc;
                negQueryGen+= pwq *log(pwq/pwdbar);
                delete qt;
            }
            if (newNonRel)
                delete hfv2;
            return negQueryGen;

        } else if (whichMethod == 4)
        {

            double mu= negMu;
            negQueryGen =0;

            lemur::api::COUNT_T tc = ind.termCount();
            startIteration();
            lemur::utility::HashFreqVector hfv(ind,dRep->getID()), * hfv2 ,*hfv3;
            if (newNonRel)
            {
                hfv2 = new lemur::utility::HashFreqVector(ind,JudgDocs[JudgDocs.size()-1]);

                TermInfoList *termList = ind.termInfoList(JudgDocs[JudgDocs.size()-1]);
                termList->startIteration();
                TermInfo *tEntry;
                while (termList->hasMore())
                {
                    tEntry = termList->nextEntry();
                    uniqueNonRel.insert(tEntry->termID());
                }
                delete termList;
                uniqueDiffRelNonRel.clear();
                set_difference(uniqueNonRel.begin() ,uniqueNonRel.end() ,uniqueRel.begin(),uniqueRel.end() ,inserter(uniqueDiffRelNonRel,uniqueDiffRelNonRel.begin()) );
            }
            if(newRel)
            {
                hfv3 = new lemur::utility::HashFreqVector(ind,relJudgDocs[relJudgDocs.size()-1]);

                TermInfoList *termList = ind.termInfoList(relJudgDocs[relJudgDocs.size()-1]);
                termList->startIteration();
                TermInfo *tEntry;
                while (termList->hasMore())
                {
                    tEntry = termList->nextEntry();
                    uniqueRel.insert(tEntry->termID());
                }
                delete termList;

                uniqueDiffRelNonRel.clear();
                set_difference(uniqueNonRel.begin() ,uniqueNonRel.end() ,uniqueRel.begin(),uniqueRel.end() ,inserter(uniqueDiffRelNonRel,uniqueDiffRelNonRel.begin()) );

            }

            while (hasMore())
            {
                lemur::api::QueryTerm *qt = nextTerm();
                double pwq = qt->weight()/totalCount();

                int freq=0;
                hfv.find(qt->id(),freq);
                if (newNonRel)
                {
                    int freq;
                    hfv2->find(qt->id(),freq);
                    countInNonRel[qt->id()] += freq;

                }

                if (newRel)
                {
                    int freq;
                    hfv3->find(qt->id(),freq);
                    countInRel[qt->id()] += freq;
                }

                if(freq>0  ||  countInNonRel[qt->id()]==0  || countInRel[qt->id()] != 0)
                    delta =0.0;

                lemur::api::TERMID_T id = qt->id();
                lemur::api::COUNT_T qtcf = ind.termCount(id);


                double pwc = (double)qtcf/(double)tc;
                double pwdbar = (delta/(delta*uniqueDiffRelNonRel.size()+mu))+((mu*pwc)/(delta*uniqueDiffRelNonRel.size()+mu));
                negQueryGen+= pwq *log(pwq/pwdbar);

                delete qt;
            }
            if (newNonRel)
                delete hfv2;
            if(newRel)
                delete hfv3;

            return negQueryGen;


        }
    }


    double negativeKL(const lemur::api::DocumentRep *dRep, vector<int> JudgDocs, bool newNonRel, double negMu, double beta = 1) const
    {
        if (newNonRel)
            DNsize += ind.docLength(JudgDocs[JudgDocs.size()-1]);

        double mu= negMu;//ind.docLengthAvg();//negGenMUHM;//2500;
        negQueryGen =0;
        //if(negQueryGen == 0)
        //{
        lemur::api::COUNT_T tc = ind.termCount();
        startIteration();
        lemur::utility::HashFreqVector hfv(ind,dRep->getID()), *hfv2;
        if (newNonRel)
            hfv2 = new lemur::utility::HashFreqVector(ind,JudgDocs[JudgDocs.size()-1]);
        while (hasMore())
        {

            lemur::api::QueryTerm *qt = nextTerm();
            double pwq = qt->weight()/totalCount();//????????????????????????????????????????
            if (newNonRel)
            {
                int freq;
                hfv2->find(qt->id(),freq);
                countInNonRel[qt->id()] += freq;
            }
            double cwdbar = 0;
            int freq=0 ;
            hfv.find(qt->id(),freq);
            cwdbar = countInNonRel[qt->id()];
            //Query Term Elimination
            if (RSMethodHM==2 && freq > 0){
                //cout<<"miad!"<<endl;
                cwdbar = 0;
            }
            lemur::api::TERMID_T id = qt->id();

            lemur::api::COUNT_T qtcf = ind.termCount(id);

            double pwc = (double)qtcf/(double)tc;
            double pwd =  ((double)freq/((double)ind.docLength(dRep->getID())+mu))+((mu*pwc)/((double)ind.docLength(dRep->getID())+mu)) ;
            double pwdbar = (cwdbar/(DNsize+mu))+((mu*pwc)/(DNsize+mu));
            negQueryGen+= pwdbar *log(pwdbar/pwd);


            //   cout<<"cwdbar: "<<cwdbar<<"\npwc: "<<pwc<<"\npwdbar: "<<pwdbar<<endl;

            delete qt;


        }
        if (newNonRel)
            delete hfv2;
        //}
        //    cout<<"Did: "<<dRep->getID()<<endl;
        //  cout<<"DNsize: "<<DNsize<<"\nnegQueryGen: "<<negQueryGen<<endl<<endl;
        return negQueryGen;


    }
    double interpolateSimsScore(lemur::api::TextQueryRep *textQR,int docID ,
                                vector<int> relJudgDoc ,vector<int> nonReljudgDoc , bool newNonRel)
    {
        return 0.0;
#if 0
        double relSim =0.0, nonRelSim = 0.0;
        double mu = 2500;

        //***********nonRelSim*******************//
        if (newNonRel)
            DNsize += ind.docLength(JudgDocs[JudgDocs.size()-1]);

        lemur::api::COUNT_T tc = ind.termCount();
        startIteration();
        lemur::utility::HashFreqVector hfv(ind,dRep->getID()), *hfv2;
        if (newNonRel)
            hfv2 = new lemur::utility::HashFreqVector(ind,JudgDocs[JudgDocs.size()-1]);
        while (hasMore())
        {
            lemur::api::QueryTerm *qt = nextTerm();
            double pwq = qt->weight()/totalCount();
            if (newNonRel)
            {
                int freq;
                hfv2->find(qt->id(),freq);
                countInNonRel[qt->id()] += freq;
            }
            double cwdbar = 0;
            int freq=0 ;
            hfv.find(qt->id(),freq);
            /*
            if(freq>0)
                cwdbar = 0;
            else
            {
                cwdbar = countInNonRel[qt->id()];
            }
            */

            lemur::api::TERMID_T id = qt->id();

            lemur::api::COUNT_T qtcf = ind.termCount(id);

            double pwc = (double)qtcf/(double)tc;

            cwdbar = countInNonRel[qt->id()];

            double pwdbar = (cwdbar/(DNsize+mu))+((mu*pwc)/(DNsize+mu));
            negQueryGen+= pwq *log(pwq/pwdbar);

            delete qt;
        }

        if (newNonRel)
            delete hfv2;
#endif
    }



protected:
    // For Query likelihood adjusted score
    mutable double negQueryGen;
    mutable double colQLikelihood;
    mutable double colKL;
    mutable bool colKLComputed;

    mutable map <int, int> countInNonRel;
    mutable set <int> uniqueNonRel;
    mutable set <int> uniqueDiffRelNonRel;
    mutable map <int, int> countInRel;
    mutable set <int> uniqueRel;

    mutable int DNsize ;

    lemur::api::IndexedRealVector *qm;
    const lemur::api::Index &ind;
};



/// Simple KL-divergence scoring function
/*!
      The KL-divergence formula D(model_q || model_d), when used for ranking
      documents, can be computed
      efficiently by re-writing the formula as a sum over all matched
      terms in a query and a document. The details of such rewriting are
      described in the following two papers:
      <ul>
      <li>C. Zhai and J. Lafferty. A study of smoothing methods for language models applied to ad hoc
      information retrieval, In 24th ACM SIGIR Conference on Research and Development in Information
      Retrieval (SIGIR'01), 2001.
      <li>P. Ogilvie and J. Callan. Experiments using the Lemur toolkit. In Proceedings of the Tenth Text
      Retrieval Conference (TREC-10).
      </ul>
    */

class ScoreFunc : public lemur::api::ScoreFunction {
public:
    enum RetParameter::adjustedScoreMethods adjScoreMethod;
    void setScoreMethod(enum RetParameter::adjustedScoreMethods adj) {
        adjScoreMethod = adj;
    }
    virtual double matchedTermWeight(const lemur::api::QueryTerm *qTerm,
                                     const lemur::api::TextQueryRep *qRep,
                                     const lemur::api::DocInfo *info,
                                     const lemur::api::DocumentRep *dRep) const {
        double w = qTerm->weight();
        double d = dRep->termWeight(qTerm->id(),info);//d = p_seen(w|d)/(a(d)*p(w|C)) [slide7-11]
        double l = log(d);
        double score = w*l;
        //cout<<info->docID()<<endl;
        /*
          cerr << "M:" << qTerm->id() <<" d:" << info->docID() << " w:" << w
          << " d:" << d << " l:" << l << " s:" << score << endl;
        */
        return score;
        //    return (qTerm->weight()*log(dRep->termWeight(qTerm->id(),info)));
    }
    /// score adjustment (e.g., appropriate length normalization)
    virtual double adjustedScore(double origScore,
                                 const lemur::api::TextQueryRep *qRep,
                                 const lemur::api::DocumentRep *dRep) const {
        const QueryModel *qm = dynamic_cast<const QueryModel *>(qRep);
        // this cast is unnecessary
        //SimpleKLDocModel *dm = (SimpleKLDocModel *)dRep;
        // dynamic_cast<SimpleKLDocModel *>dRep;

        double qsc = qm->scoreConstant();//|q|

        double dsc = log(dRep->scoreConstant());//log(a(d))

        double cql = qm->colQueryLikelihood();// sigma(c(w,q)*P(w|C))//sigma(p(Wq|C)
        // real query likelihood

        double s = dsc * qsc + origScore + cql; // s = log(P(q|d))

        double qsNorm = origScore/qsc;

        double qmD = qm->colDivergence();

        /*
          cerr << "A:"<< origScore << " dsc:" << dsc  << " qsc:" << qsc
          << " cql:" << cql << " s:"  << s << endl;
        */
        /// The following are three different options for scoring
        switch (adjScoreMethod) {
        case RetParameter::QUERYLIKELIHOOD:
            /// ==== Option 1: query likelihood ==============
            // this is the original query likelihood scoring formula
            return s;
            //      return (origScore+log(dm->scoreConstant())*qm->scoreConstant());
        case RetParameter::CROSSENTROPY:
            /// ==== Option 2: cross-entropy (normalized query likelihood) ====
            // This is the normalized query-likelihood, i.e., cross-entropy
            assert(qm->scoreConstant()!=0);
            // return (origScore/qm->scoreConstant() + log(dm->scoreConstant()));
            // add the term colQueryLikelihood/qm->scoreConstant
            s = qsNorm + dsc + cql/qsc;
            return (s);
        case RetParameter::NEGATIVEKLD:
            /// ==== Option 3: negative KL-divergence ====
            // This is the exact (negative) KL-divergence value, i.e., -D(Mq||Md)

            assert(qm->scoreConstant()!=0);
            s = qsNorm + dsc - qmD;

            /*
            cerr << origScore << ":" << qsNorm << ":" << dsc  << ":" << qmD  << ":" << s << endl;
          */
            return s;
            //      return (origScore/qm->scoreConstant() + log(dm->scoreConstant())
            //          - qm->colDivergence());
        default:
            cerr << "unknown adjusted score method" << endl;
            return origScore;
        }
    }

};

/// KL Divergence retrieval model with simple document model smoothing
class RetMethod : public lemur::api::TextQueryRetMethod {
public:

    /// Construction of SimpleKLRetMethod requires a smoothing support file, which can be generated by the application GenerateSmoothSupport. The use of this smoothing support file is to store some pre-computed quantities so that the scoring procedure can be speeded up.
    RetMethod(const lemur::api::Index &dbIndex,
              const string &supportFileName,
              lemur::api::ScoreAccumulator &accumulator);
    virtual ~RetMethod();

    virtual lemur::api::TextQueryRep *computeTextQueryRep(const lemur::api::TermQuery &qry) {
        return (new QueryModel(qry, ind));
    }

    virtual lemur::api::DocumentRep *computeDocRep(lemur::api::DOCID_T docID);


    virtual lemur::api::ScoreFunction *scoreFunc() {
        return (scFunc);
    }

    virtual void updateTextQuery(lemur::api::TextQueryRep &origRep,
                                 const lemur::api::DocIDSet &relDocs, const lemur::api::DocIDSet &nonRelDocs);

    virtual void updateProfile(lemur::api::TextQueryRep &origRep,
                               vector<int> relJudglDoc , vector<int> nonReljudgDoc, bool isRel);
    virtual void updateThreshold(lemur::api::TextQueryRep &origRep,
                                 vector<int> relJudglDoc ,vector<int> nonReljudgDoc ,int mode,double relSumScores , double nonRelSumScores);
    virtual float computeProfDocSim(lemur::api::TextQueryRep *origRep,int docID ,vector<int>relDocs ,vector<int>nonRelDocs , bool newNonRel,bool newRel);

    virtual float cosineSim(vector<double> aa, vector<double> bb);
    virtual float softMaxFunc(vector<double> aa, vector<double> bb);
    virtual float softMaxFunc2(vector<double> aa, vector<double> bb);

    virtual vector<double> extractKeyWord(int newDocId);

    void computeWEFBModel(QueryModel &origRep,const DocIDSet &relDocs);
    void genSample(vector<pair<int,double> > &pos,vector<pair<int,double> > &neg,vector<int> &VS,map<int,int> &word_count,const DocIDSet
            &relDocs,QueryModel &origRep);

    double fangScore( DocIDSet &fbDocs, int docID, bool newNonRel)
    {

        COUNT_T numTerms = ind.termCountUnique();

        double *distQuery = new double[numTerms+1];
        double *distQueryEst = new double[numTerms+1];
        if (newNonRel)
        {
            lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(fbDocs, ind);
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
                prev_distQuery.clear();
                for (i=1; i<=numTerms;i++) {

                    distQuery[i] = distQueryEst[i]/distQueryNorm;
                    //prev_distQuery[i] = distQuery[i];
                    // cerr << "dist: "<< distQuery[i] << endl;
                    if(distQuery[i]>0)
                        prev_distQuery[i] = distQuery[i];
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
                    //    << " with likelihood= "<< ll << endl;
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
            delete dCounter;
        }

        //lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
        double fang_score = 0;
        lemur::utility::HashFreqVector hfv(ind,docID);
        //DocModel *dm;
        //dm = dynamic_cast<DocModel *> (dRep);
        DocModel * dm = new DPriorDocModel(docID,
                                           ind.docLength(docID),
                                           *collectLM,
                                           docProbMass,
                                           docParam.DirPrior,
                                           docParam.smthStrategy);

        // for (int i=1; i<=numTerms; i++) {
        //   if (distQuery[i] > 0) {
        for(map<int,double>::iterator it = prev_distQuery.begin(); it!= prev_distQuery.end() ; it++){
            int tf=0 ;
            hfv.find(it->first,tf);
            fang_score+= it->second * log (it->second/dm->seenProb(tf, it->first));
        }
        //lmCounter.incCount(i, distQuery[i]);
        // }
        //}

        delete dm;
        //lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
        //origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
        //                      qryParam.fbPrSumTh, qryParam.fbPrTh);

        //delete fblm;

        delete[] distQuery;
        delete[] distQueryEst;
        return fang_score;
    }

    void setCoeffParam(double coe);

    void setDocSmoothParam(RetParameter::DocSmoothParam &docSmthParam);
    void setQueryModelParam(RetParameter::QueryModelParam &queryModParam);


    double getThreshold(){return mozhdehHosseinThreshold;}
    void setThreshold(double thr)
    {
        mozhdehHosseinThreshold = thr;
    }
    double getNegWeight(){return mozhdehHosseinNegWeight;}
    void setNegWeight(double negw)
    {
        mozhdehHosseinNegWeight =negw;
    }

    //for linear thr updating method
    double getC1(){return C1;}
    void setC1(double val){C1=val;}
    double getC2(){return C2;}
    void setC2(double val){C2=val;}

    double getInitC1(){return initC1;}
    double getInitC2(){return initC2;}
    void setInitC1(double val){initC1 = val;}
    void setInitC2(double val){initC2 = val;}

    void clearPrevDistQuery()
    {
        prev_distQuery.clear();
    }

    //for diff thr updating method
    void setDiffThrUpdatingParam(double alpha){diffThrUpdatingParam=alpha;}
    double getDiffThrUpdatingParam(){return diffThrUpdatingParam;}


    double getNegMu(){return NegMu;}
    void setNegMu(double val){NegMu=val;}

    double getLambda1(){return lambda_1;}
    void setLambda1(double val){lambda_1=val;}

    double getLambda2(){return lambda_2;}
    void setLambda2(double val){lambda_2=val;}


    double getDelta(){return delta;}
    void setDelta(double val){delta = val;}

    void clearRelNonRelCountFlag()
    {
        newNonRelRecieved = false;
        newRelRecieved = false;
        newNonRelRecievedCnt = 0;
        newRelRecievedCnt =0;
    }


protected:

    bool newNonRelRecieved,newRelRecieved;
    int newNonRelRecievedCnt,newRelRecievedCnt;

    double mozhdehHosseinThreshold;
    double mozhdehHosseinNegWeight;
    mutable int thresholdUpdatingMethod;/* 0->no updating
                                           1->linear
                                           2->diff
                                        */
    double C1,C2;//for linear
    double initC1,initC2;
    double diffThrUpdatingParam;//for diff

    //double *prev_distQuery;
    map <int,double>prev_distQuery;

    double NegMu;
    double delta;

    double lambda_1;
    double lambda_2;
    //Matrix Factorization method for query expansion
    bool MF;

    /// needed for fast one-step Markov chain
    double *mcNorm;

    /// needed for fast alpha computing
    double *docProbMass;
    /// needed for supporting fast absolute discounting
    lemur::api::COUNT_T *uniqueTermCount;
    /// a little faster if pre-computed
    lemur::langmod::UnigramLM *collectLM;
    /// support the construction of collectLM
    lemur::langmod::DocUnigramCounter *collectLMCounter;
    /// keep a copy to be used at any time
    ScoreFunc *scFunc;



    /// @name query model updating methods (i.e., feedback methods)
    //@{
    /// Mixture model feedback method
    void computeMixtureFBModel(QueryModel &origRep,
                               const lemur::api::DocIDSet & relDocs, const lemur::api::DocIDSet &nonRelDocs);
    /// Divergence minimization feedback method
    void computeDivMinFBModel(QueryModel &origRep,
                              const lemur::api::DocIDSet &relDocs);
    void computeMEDMMFBModel(QueryModel &origRep,
                             const lemur::api::DocIDSet &relDocs);

    /// Markov chain feedback method
    void computeMarkovChainFBModel(QueryModel &origRep,
                                   const lemur::api::DocIDSet &relDocs) ;
    /// Relevance model1 feedback method
    void computeRM1FBModel(QueryModel &origRep,
                           const lemur::api::DocIDSet & relDocs,const lemur::api::DocIDSet & nonRelDocs);
    /// Relevance model1 feedback method
    void computeRM2FBModel(QueryModel &origRep,
                           const lemur::api::DocIDSet & relDocs);
    void computeRM3FBModel(QueryModel &origRep,
                           const lemur::api::DocIDSet & relDocs);
    void computeRM4FBModel(QueryModel &origRep,
                           const lemur::api::DocIDSet & relDocs);



    void multiplyMatrix2Vec(vector<double> &res );
    void multiplyVec2Vec(vector<double> m1,  vector<vector<double> >&res );
    void computeCoefMatrix();
    void computeNearestTerm2Vec(vector<double>);
    //@}

    RetParameter::DocSmoothParam docParam;
    RetParameter::QueryModelParam qryParam;

    /// Load support file support
    void loadSupportFile();
    const string supportFile;

public:
    void computeRelNonRelDist(TextQueryRep &origRep,
                              const vector<int> relDocs, const vector<int> nonRelDocs, bool isRelevant, bool computeCoeff); //!!!

    void setNumberOfPositiveSelectedTopWordAndFBcount(double n)
    {
        numberOfPositiveSelectedTopWord = n;
        this->qryParam.fbTermCount = n;
    }


    bool *relComputed;
    bool *nonRelComputed;

    int W2VecDimSize;
    vector<double> Vq;
    vector<pair<int ,double> >pos, neg;

    vector<int> initRel;
    vector<int> initNonRel;


    double alphaCoef,lambdaCoef,etaCoef,betaCoef;
    double **coefMatrix;
    //vector<vector<double> >coefMatrix(100,vector<double>(100));
    //vector<vector<double> >coefMatrix;
    //vector<double> queryAvgVec;

    //vector<vector<double> > Vwn ,Vbwn;
    double numberOfPositiveSelectedTopWord,numberOfNegativeSelectedTopWord;

    //ofstream inputfileee;

};


inline  void RetMethod::setDocSmoothParam(RetParameter::DocSmoothParam &docSmthParam)
{
    docParam = docSmthParam;
    loadSupportFile();
}

inline  void RetMethod::setQueryModelParam(RetParameter::QueryModelParam &queryModParam)
{
    qryParam = queryModParam;
    // add a parameter to the score function.
    // isn't available in the constructor.
    scFunc->setScoreMethod(qryParam.adjScoreMethod);
    loadSupportFile();
}
}
}

#endif /* _SIMPLEKLRETMETHOD_HPP */

