#ifndef SMTH
#define SMTH

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <sstream>
#include <DocStream.hpp>
#include <BasicDocStream.hpp>
#include "IndexManager.hpp"
#include "ResultFile.hpp"
//#include "DocUnigramCounter.hpp"
#include "RetMethod.h"
#include "QueryDocument.hpp"
#include <sstream>

#include "Parameters.h"

#include <iomanip>//for setprecisoin

using namespace lemur::api;
using namespace lemur::langmod;
using namespace lemur::parse;
using namespace lemur::retrieval;
using namespace std;

#define DATASET 0 //0-->infile, 1-->ohsu
#define RETMODE RSMethodHM//LM(0) ,RS(1), NegKLQTE(2),NegKL(3)
#define NEGMODE negGenModeHM//coll(0) ,NonRel(1)
#define FBMODE feedbackMode//NoFB(0),NonRel(1),Normal(2),Mixture(3)
#define UPDTHRMODE 1//updatingThresholdMode//No(0),Linear(1) ,Diff(2)

template <typename T>
string numToStr(T number)
{
    ostringstream s;
    s << number;
    return s.str();
}

void loadJudgment();
void computeRSMethods(Index *);
void MonoKLModel(Index* ind);
vector<int> queryDocList(Index* ind,TextQueryRep *textQR);
void readWordEmbeddingFile(Index *);
void writeDocs2File(Index*);
void showNearerTermInW2V(DocStream *qs,RetMethod *myMethod ,Index *ind);
bool pairCompare(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem);
void showNearerTerms2QueryVecInW2V(DocStream *qs,RetMethod *myMethod ,Index *ind, int avgOrMax);
void computeQueryAvgVec(Document *d,RetMethod *myMethod );
void computeMixtureForDocsAndWriteToFile(Index *ind,RetMethod *myMethod);
void readDocIdKeyWords();
void initJudgDocsVector(Index* ind,vector<int>&rel , vector<int>&nonRel,string queryID);
void readStopWord(Index *ind);

extern double startThresholdHM , endThresholdHM , intervalThresholdHM ;
extern int WHO;// 0--> server , 1-->Mozhdeh, 2-->AP, other-->Hossein
extern string outputFileNameHM;
extern string resultFileNameHM;
extern int feedbackMode;
extern double startNegWeight,endNegWeight , negWeightInterval;
extern double startNegMu, endNegMu, NegMuInterval;
extern double startDelta, endDelta, deltaInterval;
extern int RSMethodHM;
extern int negGenModeHM;
extern double smoothJMInterval1,smoothJMInterval2;
extern int updatingThresholdMode;

//map<string , vector<string> >queryRelDocsMap;
map<string , set<string> >queryRelDocsMap;
map<string ,set<string> > queryNonRelDocsMap;
string judgmentPath,indexPath,queryPath;
string resultPath = "";
map<int,vector<double> >wordEmbedding;
map<int ,vector<double> >docIdKeyWords;
set<int> stopWords;

//int numberOfInitRelDocs = 5;
//int numberOfInitNonRelDocs = 15;

int main(int argc, char * argv[])
{
    readParams(string(argv[1]));
    cout<< "reading param file: "<<argv[1]<<endl;
    switch (WHO)
    {
    case 0:
        if(DATASET == 0)//infile
        {
            judgmentPath = "/home/iis/Desktop/Edu/thesis/Data/INFILE/qrels_en";
            indexPath= "/home/iis/Desktop/Edu/thesis/index/infile/en/index.key";
            queryPath = "/home/iis/Desktop/Edu/thesis/Data/INFILE/q_en_titleKeyword_en.stemmed.xml";
        }else if(DATASET == 1)//ohsu
        {
            //judgmentPath = "/home/iis/Desktop/Edu/thesis/Data/ohsumed/trec9-train/qrels.ohsu.adapt.87";
            judgmentPath = "/home/iis/Desktop/Edu/thesis/Data/ohsumed/trec9-test/qrels.ohsu.88-91";
            //indexPath= "/home/iis/Desktop/Edu/thesis/index/ohsumed/ohsu/index/index.key";
            indexPath= "/home/iis/Desktop/Edu/thesis/index/ohsumed/ohsu/testIndex/index/index.key";
            queryPath = "/home/iis/Desktop/Edu/thesis/Data/ohsumed/trec9-train/trec9-train_output/stemmed_ohsu_query.txt";
        }
        break;
    case 1:
        judgmentPath = "/home/mozhdeh/Desktop/INFILE/hosein-data/qrels_en";
        indexPath = "/home/mozhdeh/Desktop/INFILE/javid-index/index.key";
        queryPath = "/home/mozhdeh/Desktop/INFILE/hosein-data/q_en_titleKeyword_en.stemmed.xml";
        break;
        //case 2:
        //    judgmentPath ="/home/mozhdeh/Desktop/AP/Data/jud-ap.txt";
        //    indexPath = "/home/mozhdeh/Desktop/AP/index/index.key";
        //   queryPath = "/home/mozhdeh/Desktop/AP/Data/topics.stemmed.xml";
        //   break;
    case 4:
        judgmentPath = "/home/iis/Desktop/RS-Framework/DataSets/Infile/Data/qrels_en";
        indexPath= "/home/iis/Desktop/RS-Framework/DataSets/Infile/Index/En/index.key";
        queryPath = "/home/iis/Desktop/RS-Framework/DataSets/Infile/Data/q_en_titleKeyword_en.stemmed.xml";

        break;
    default:
        if(DATASET == 0)//infile
        {
            judgmentPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/qrels_en";
            //indexPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Index/en_notStemmed_withoutSW/index.key";
            //queryPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/q_en_titleKeyword_notStemmed_en.xml";

            indexPath ="/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Index/en_Stemmed_withoutSW/index.key";
            queryPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/q_en_titleKeyword_en.stemmed.xml";

        }else if(DATASET == 1)//ohsu
        {
            judgmentPath = "/home/hossein/Desktop/IIS/lemur/DataSets/Ohsumed/Data/trec9-train/qrels.ohsu.adapt.87";
            indexPath = "/home/hossein/Desktop/IIS/lemur/DataSets/Ohsumed/Index/trec9-train/index.key";
            queryPath = "/home/hossein/Desktop/IIS/lemur/DataSets/Ohsumed/Data/trec9-train/stemmed_ohsu_query.txt";

        }

        break;
    }

    Index *ind = IndexManager::openIndex(indexPath);// with StopWord && stemmed
    cerr<<ind->term("because")<<endl;
    cerr<<ind->term("doping")<<endl;

    //writeDocs2File(ind);

#if 1
    //readStopWord(ind);

    readWordEmbeddingFile(ind);
    loadJudgment();
    computeRSMethods(ind);
#endif

}

void computeRSMethods(Index* ind)
{
    DocStream *qs = new BasicDocStream(queryPath); // Your own path to topics
    ArrayAccumulator accumulator(ind->docCount());
    RetMethod *myMethod = new RetMethod(*ind,"",accumulator);

    //computeMixtureForDocsAndWriteToFile(ind,myMethod);
    //readDocIdKeyWords();

    //showNearerTermInW2V(qs,myMethod,ind);
    //showNearerTerms2QueryVecInW2V(qs,myMethod,ind,1);
    //return;

    string outFilename;
    if(DATASET == 0)
        outFilename =outputFileNameHM+"_infile";
    else if (DATASET == 1)
        outFilename =outputFileNameHM+"_ohsu";

    //myMethod->relComputed = new bool[200];
    //myMethod->nonRelComputed = new bool[200];

#define UpProf  1
#define COMPAVG 1
    string methodName = "Stemmed_NoSW_MFWE_NonRel_Coef0.95";

    outFilename += methodName;
    //outFilename += "#topPosW:20-70(20)_CsNoT_NumbersT_CoefT";

    ofstream out(outFilename.c_str());


    cout<< "RSMethod: "<<RSMethodHM<<" NegGenMode: "<<negGenModeHM<<" feedbackMode: "<<feedbackMode<<" updatingThrMode: "<<updatingThresholdMode<<"\n";
    cout<< "RSMethod: "<<RETMODE<<" NegGenMode: "<<NEGMODE<<" feedbackMode: "<<FBMODE<<" updatingThrMode: "<<UPDTHRMODE<<"\n";
    cout<<"outfile: "<<outFilename<<endl;

    double start_thresh =startThresholdHM, end_thresh= endThresholdHM;

    for (double thresh = start_thresh ; thresh<=end_thresh ; thresh += intervalThresholdHM)
    {
        for(myMethod->betaCoef = 0.05; myMethod->betaCoef < 1 ; myMethod->betaCoef += 0.3)//4
        {
            for(myMethod->lambdaCoef = 0.05; myMethod->lambdaCoef < 1; myMethod->lambdaCoef += 0.3)//4
            {
                //for(double fbCoef = 0.05 ; fbCoef <=0.99 ; fbCoef+=0.15)//7
                {
                    //myMethod->alphaCoef = 0.81;
                    //myMethod->betaCoef = 0.2;
                    //myMethod->lambdaCoef = 0.0;
                    double fbCoef = 0.95;
                    //double topPos = 30.0;

                    for(myMethod->alphaCoef = 0.05; myMethod->alphaCoef <= 1; myMethod->alphaCoef+=0.19)//5
                    {
                        for(double topPos = 20; topPos <= 65 ; topPos+=20)//3
                        {
                            //for(double c1 = 0.10 ; c1<=0.36 ;c1+=0.05)//inc//6
                            double c1 = 0.30;
                            {
                                myMethod->setC1(c1);
                                //for(double c2 = 0.01 ; c2 <= 0.2 ; c2+=0.03)//dec //7
                                double c2 = 0.05;
                                {
                                    //if(c2 > c1)
                                    //    break;
                                    //myMethod->setThreshold(init_thr);
                                    myMethod->setC2(c2);

                                    //for(int numOfShownNonRel = 4;numOfShownNonRel< 11;numOfShownNonRel+=3 )//3
                                    int numOfShownNonRel = 5;
                                    {

                                        //for(int numOfnotShownDoc = 100 ;numOfnotShownDoc <= 501 ; numOfnotShownDoc+=100)//4
                                        int numOfnotShownDoc = 350;
                                        {
                                            myMethod->setThreshold(thresh);
                                            myMethod->setNumberOfPositiveSelectedTopWordAndFBcount(topPos);


                                            cout<<"c1: "<<c1<<" c2: "<<c2<<" numOfShownNonRel: "<<numOfShownNonRel<<" numOfnotShownDoc: "<<numOfnotShownDoc<<" "<<endl;
                                            resultPath = resultFileNameHM.c_str() +numToStr( myMethod->getThreshold() )+"_c1:"+numToStr(c1)+"_c2:"+numToStr(c2)+"_#showNonRel:"+numToStr(numOfShownNonRel)+"_#notShownDoc:"+numToStr(numOfnotShownDoc)+"#topPosW:"+numToStr(myMethod->numberOfPositiveSelectedTopWord)+"#topNegW:"+numToStr(myMethod->numberOfNegativeSelectedTopWord);
                                            resultPath += "fbCoef:"+numToStr(fbCoef)+methodName+"CsNoT_NumberNoT_"+"a:"+numToStr(myMethod->alphaCoef)+"b:"+numToStr(myMethod->betaCoef)+"l:"+numToStr(myMethod->lambdaCoef)+".res";


                                            //myMethod->setThreshold(thresh);
                                            out<<"threshold: "<<thresh<<" fbcoef: "<<fbCoef<<" topPos: "<<topPos<<" alpha: "<<myMethod->alphaCoef<<" beta: "<<myMethod->betaCoef<<" lambda: "<<myMethod->lambdaCoef<<endl ;

                                            IndexedRealVector results;

                                            qs->startDocIteration();
                                            TextQuery *q;


                                            ofstream result(resultPath.c_str());
                                            ResultFile resultFile(1);
                                            resultFile.openForWrite(result,*ind);

                                            double relRetCounter = 0 , retCounter = 0 , relCounter = 0;
                                            vector<double> queriesPrecision,queriesRecall;
                                            while(qs->hasMore())
                                            {
                                                //myMethod->clearRelNonRelCountFlag();
                                                //myMethod->clearPrevDistQuery();

                                                myMethod->setThreshold(thresh);
                                                myMethod->setCoeffParam(fbCoef);

                                                double relSumScores =0.0,nonRelSumScores = 0.0;

                                                int numberOfNotShownDocs = 0,numberOfShownNonRelDocs = 0;

                                                vector<int> relJudgDocs,nonRelJudgDocs;


                                                results.clear();



                                                Document *d = qs->nextDoc();
                                                q = new TextQuery(*d);
                                                QueryRep *qr = myMethod->computeQueryRep(*q);
                                                cout<<"qid: "<<q->id()<<endl;


                                                ///*******************************************************///
#if COMPAVG
                                                /*vector<int> rell,nonrell;
                                                //cerr<<"before: "<<myMethod->initRel.size() <<" "<<myMethod->initNonRel.size()<<endl;
                                                initJudgDocsVector(ind ,rell ,nonrell, q->id());

                                                myMethod->initNonRel.clear();
                                                myMethod->initRel.clear();

                                                myMethod->initNonRel.assign(nonrell.begin(), nonrell.end() );
                                                myMethod->initRel.assign(rell.begin() , rell.end() );
                                                //cerr<<"after: "<<myMethod->initRel.size() <<" "<<myMethod->initNonRel.size()<<endl;*/


                                                computeQueryAvgVec(d,myMethod);

                                                //myMethod->computeRelNonRelDist(*((TextQueryRep *)(qr)),rell,nonrell,false,false);
                                                //myMethod->computeRelNonRelDist(*((TextQueryRep *)(qr)),rell,nonrell,true,false);
#endif
                                                ///*******************************************************///

                                                bool newNonRel = false , newRel = false;

                                                //vector<string> relDocs;
                                                set<string> relDocs;
                                                if( queryRelDocsMap.find(q->id()) != queryRelDocsMap.end() )//find it!
                                                    relDocs = queryRelDocsMap[q->id()];
                                                else
                                                {
                                                    cerr<<"*******this query has no rel judg(ignore)**********\n";
                                                    continue;
                                                }



                                                /*for(int ii = 0; ii < 200; ii++)
                                                {
                                                    myMethod->relComputed[ii] = false;
                                                    myMethod->nonRelComputed[ii] = false;
                                                }*/

                                                //for(int docID = 1 ; docID < ind->docCount() ; docID++){ //compute for all doc
                                                vector<int> docids = queryDocList(ind,((TextQueryRep *)(qr)));

                                                cout<<"reldocsize: "<<relDocs.size()<<endl;

                                                for(int i = 0 ; i<docids.size(); i++) //compute for docs which have queryTerm
                                                {
                                                    int docID = docids[i];

                                                    float sim = myMethod->computeProfDocSim(((TextQueryRep *)(qr)) ,docID, relJudgDocs , nonRelJudgDocs , newNonRel,newRel);


                                                    if(sim >=  myMethod->getThreshold() )
                                                    {
                                                        //cerr<<sim<<"\n";
                                                        numberOfNotShownDocs=0;
                                                        bool isRel = false;


                                                        if(relDocs.find(ind->document(docID) ) != relDocs.end())
                                                        {
                                                            isRel = true;
                                                            newNonRel = false;
                                                            newRel = true;
                                                            relJudgDocs.push_back(docID);
                                                        }
                                                        else
                                                        {
                                                            nonRelJudgDocs.push_back(docID);
                                                            newNonRel = true;
                                                            newRel = false;
                                                            numberOfShownNonRelDocs++;
                                                        }
                                                        results.PushValue(docID , sim);

                                                        if(results.size() > 200)
                                                        {
                                                            cout<<"BREAKKKKKKKKKK because of results size > 200\n";
                                                            break;
                                                        }

#if UpProf

                                                        //if(isRel)
                                                        if (results.size() % 15 == 0 /*&& feedbackMode > 0*/)
                                                        {
                                                            //cout<<"Updating profile. Result size: "<<results.size()<<endl;
                                                            myMethod->updateProfile(*((TextQueryRep *)(qr)),relJudgDocs , nonRelJudgDocs , isRel);
                                                        }
#endif
                                                        if(!isRel)
                                                            if( numberOfShownNonRelDocs == numOfShownNonRel )
                                                            {
                                                                myMethod->updateThreshold(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs ,0,relSumScores,nonRelSumScores);//inc thr
                                                                numberOfShownNonRelDocs =0;
                                                            }

                                                    }
                                                    else
                                                    {
                                                        numberOfNotShownDocs++;
                                                    }
#if UPDTHRMODE == 1
                                                    if(numberOfNotShownDocs == numOfnotShownDoc)//not show anything after |numOfnotShownDoc| docs! -->dec(thr)
                                                    {
                                                        myMethod->updateThreshold(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs ,1,relSumScores,nonRelSumScores);//dec thr
                                                        numberOfNotShownDocs = 0;
                                                    }
#endif
                                                }//endfor docs

                                                cerr<<"\nresults size : "<<results.size()<<endl;

                                                results.Sort();
                                                resultFile.writeResults(q->id() ,&results,results.size());
                                                relRetCounter += relJudgDocs.size();
                                                retCounter += results.size();
                                                relCounter += relDocs.size();

                                                if(results.size() != 0)
                                                {
                                                    queriesPrecision.push_back((double)relJudgDocs.size() / results.size());
                                                    queriesRecall.push_back((double)relJudgDocs.size() / relDocs.size() );
                                                }else // have no suggestion for this query
                                                {
                                                    queriesPrecision.push_back(0.0);
                                                    queriesRecall.push_back(0.0);
                                                }



                                                delete q;
                                                delete qr;

                                                //delete d; error

                                            }//end queries


                                            double avgPrec = 0.0 , avgRecall = 0.0;
                                            for(int i = 0 ; i < queriesPrecision.size() ; i++)
                                            {
                                                avgPrec+=queriesPrecision[i];
                                                avgRecall+= queriesRecall[i];
                                                out<<"Prec["<<i<<"] = "<<queriesPrecision[i]<<"\tRecall["<<i<<"] = "<<queriesRecall[i]<<endl;
                                            }
                                            avgPrec/=queriesPrecision.size();
                                            avgRecall/=queriesRecall.size();

#if UPDTHRMODE == 1
                                            out<<"C1: "<< c1<<"\nC2: "<<c2<<endl;
                                            out<<"numOfShownNonRel: "<<numOfShownNonRel<<"\nnumOfnotShownDoc: "<<numOfnotShownDoc<<endl;
#endif
                                            out<<"Avg Precision: "<<avgPrec<<endl;
                                            out<<"Avg Recall: "<<avgRecall<<endl;
                                            out<<"F-measure: "<<(2*avgPrec*avgRecall)/(avgPrec+avgRecall)<<endl<<endl;

                                            double pp = relRetCounter/retCounter;
                                            double dd = relRetCounter/relCounter;
                                            out<<"rel_ret: "<<relRetCounter<<" ret: "<<retCounter<<" rels: "<<relCounter<<endl;
                                            out<<"old_Avg Precision: "<<pp<<endl;
                                            out<<"old_Avg Recall: "<<dd<<endl;
                                            out<<"old_F-measure: "<<(2*pp*dd)/(pp+dd)<<endl<<endl;




#if UPDTHRMODE == 1
                                        }//end numOfnotShownDoc for
                                    }//end numOfShownNonRel for
                                }//end c1 for
                            }//end c2 for
                            //}alpha
                            //}beta
                            //}lambda
#endif


                        }
                    }
                }
            }
        }

    }
    //#endif
    delete qs;
    delete myMethod;
}

void initJudgDocsVector(Index *ind,vector<int>&rel , vector<int>&nonRel,string queryID)
{
    set<string> docs;
    set<string>::iterator it;
    int counter = 5;
    if( queryRelDocsMap.find(queryID) != queryRelDocsMap.end() )//find it!
    {
        docs = queryRelDocsMap[queryID];
        //rel.assign(docs.begin(),docs.begin() + numberOfInitRelDocs - 1 );
        for(it = docs.begin() ; it !=docs.end() && counter-- > 0 ;++it )
            rel.push_back(ind->document( *it));
        if( queryNonRelDocsMap.find(queryID) != queryNonRelDocsMap.end() )//find it!
        {
            docs.clear();
            docs = queryNonRelDocsMap[queryID];
            //nonRel.assign(docs.begin(),docs.begin() + numberOfInitNonRelDocs -1);
            counter = 10;
            for(it = docs.begin() ; it !=docs.end() && counter-- > 0 ;++it )
                nonRel.push_back(ind->document(*it));
        }
    }
}
void loadJudgment()
{
    int judg,temp;
    string docName,id;

    ifstream infile;
    infile.open (judgmentPath.c_str());

    string line;
    while (getline(infile,line))
    {
        stringstream ss(line);
        if(DATASET == 0)//infile
        {
            ss >> id >> temp >> docName >> judg;
            if(judg == 1)
            {

                queryRelDocsMap[id].insert(docName);
                //map<string,bool>m;m.insert("ss",false)
                //cerr<<id<<" "<<docName<<endl;
            }else
            {
                queryNonRelDocsMap[id].insert(docName);
            }


        }else if(DATASET == 1)//ohsu
        {
            ss >> id >> docName;
            queryRelDocsMap[id].insert(docName);
            //queryNonRelDocsMap[id].insert(docName); FIX ME for nonRel!!!!
        }
    }
    infile.close();


    //110,134,147 rel nadaran
    /*map<string , vector<string> >::iterator it;
    for(it = queryRelDocsMap.begin();it!= queryRelDocsMap.end() ; ++it)
        cerr<<it->first<<endl;*/

}

void computeMixtureForDocsAndWriteToFile(Index *ind ,RetMethod *myMethod)
{

    vector<int>documentIDs;
    DocStream *qs = new BasicDocStream(queryPath); // Your own path to topics
    qs->startDocIteration();
    TextQuery *q;
    while(qs->hasMore())
    {
        Document *d = qs->nextDoc();
        q = new TextQuery(*d);
        QueryRep *qr = myMethod->computeQueryRep(*q);

        vector<int>temp = queryDocList(ind , ((TextQueryRep *)(qr)));
        documentIDs.insert(documentIDs.begin() ,temp.begin(), temp.end());

        delete q;
        delete qr;
    }
    delete qs;

    cout<<"before: "<<documentIDs.size()<<endl;
    sort( documentIDs.begin(), documentIDs.end() );
    documentIDs.erase( unique( documentIDs.begin(), documentIDs.end() ), documentIDs.end() );

    cout<<"after: "<<documentIDs.size()<<endl;


    ofstream out;
    out.open("docKeyWords_top20word.txt");
    out<<std::setprecision(14);
    for(int i = 0 ; i < documentIDs.size() ;i++)
    {
        out<<documentIDs[i]<< " ";
        vector<double> dd = myMethod->extractKeyWord(documentIDs[i]);
        for(int j = 0 ; j < dd.size() ; j++)
            out<<setprecision(14)<<dd[j]<<" ";
        out<<endl;
    }
    out.close();
}

void readDocIdKeyWords()
{
    ifstream input("docKeyWords_top20word.txt");
    if(input.is_open())
    {
        string line;
        while(getline(input ,line))
        {

            istringstream iss(line);
            int docid=0;
            iss >> docid;
            vector<double> temp;
            do
            {
                double sub;
                iss >> sub;
                temp.push_back(sub);
                //cout << "Substring: " << sub << endl;
            } while (iss);
            docIdKeyWords.insert(pair<int , vector<double> >(docid,temp));

        }

    }else
        cerr<<"docKeyWords.txt doesn't exist!!!!!!!!!";

    input.close();

}
vector<int> queryDocList(Index* ind,TextQueryRep *textQR)
{
    vector<int> docids;
    set<int> docset;
    textQR->startIteration();
    while (textQR->hasMore()) {
        QueryTerm *qTerm = textQR->nextTerm();
        if(qTerm->id()==0){
            cerr<<"**********"<<endl;
            continue;
        }
        DocInfoList *dList = ind->docInfoList(qTerm->id());

        dList->startIteration();
        while (dList->hasMore()) {
            DocInfo *info = dList->nextEntry();
            DOCID_T id = info->docID();
            docset.insert(id);
        }
        delete dList;
        delete qTerm;
    }
    docids.assign(docset.begin(),docset.end());
    return docids;
}

void MonoKLModel(Index* ind){
    DocStream *qs = new BasicDocStream(queryPath.c_str()); // Your own path to topics
    ArrayAccumulator accumulator(ind->docCount());
    RetMethod *myMethod = new RetMethod(*ind,"",accumulator);
    IndexedRealVector results;
    qs->startDocIteration();
    TextQuery *q;

    ofstream result("res.my_ret_method");
    ResultFile resultFile(1);
    resultFile.openForWrite(result,*ind);
    PseudoFBDocs *fbDocs;
    while(qs->hasMore()){
        Document* d = qs->nextDoc();
        //d->startTermIteration(); // It is how to iterate over query terms
        //ofstream out ("QID.txt");
        //while(d->hasMore()){
        //	const Term* t = d->nextTerm();
        //	const char* q = t->spelling();
        //	int q_id = ind->term(q);
        //	out<<q_id<<endl;
        //}
        //out.close();
        q = new TextQuery(*d);
        QueryRep *qr = myMethod->computeQueryRep(*q);
        myMethod->scoreCollection(*qr,results);
        results.Sort();
        //fbDocs= new PseudoFBDocs(results,30,false);
        //myMethod->updateQuery(*qr,*fbDocs);
        //myMethod->scoreCollection(*qr,results);
        //results.Sort();
        resultFile.writeResults(q->id(),&results,results.size());
        cerr<<"qid "<<q->id()<<endl;
        break;
    }
}
void writeDocs2File(Index *ind)
{
    ofstream outfile;
    outfile.open("infile_docs_Stemmed_withoutSW.txt");
    {

        for(int docID = 1 ; docID < ind->docCount(); docID++)
        {
            outfile << ind->document(docID )<<endl;

            TermInfoList *docTermInfoList =  ind->termInfoList(docID);
            docTermInfoList->startIteration();
            vector<string> doc(3*ind->docLength(docID)," ");

            while(docTermInfoList->hasMore())
            {
                TermInfo *ti = docTermInfoList->nextEntry();
                const LOC_T *poses = ti->positions();

                for(int i = 0 ; i < ti->count() ;i++)
                {
                    doc[poses[i] ]=ind->term(ti->termID());
                }
                //delete poses;
                //delete ti;

            }
            for(int i = 0 ;i < doc.size();i++)
            {
                if(doc[i] != " ")
                    outfile<<doc[i]<<" ";
            }
            outfile<<endl;

            //delete docTermInfoList;
        }

    }
    outfile.close();
}
void readWordEmbeddingFile(Index *ind)
{
    int cc=0;
    cout << "ReadWordEmbeddingFile\n";
    string line;

#if 1
    //glove
    ifstream in("dataSets/infile_docs_Stemmed_withoutSW_W2V.vectors");
    //ifstream in("dataSets/infile_vectors_100D_W2V.txt");
    //ifstream in("dataSets/infile_docs_Stemmed_withoutSW_W2V.vectors");
    getline(in,line);//first line is statistical in W2V

#endif
#if 0
    ifstream in("dataSets/infile_docs_Stemmed_withoutSW_LDA.vectors");
#endif

    while(getline(in,line))
    {
        cc++;
        istringstream iss(line);

        string sub;
        double dd;
        iss >> sub;

        //if(sub.size() <= 1)
        //    continue;

        int termID = ind->term(sub);

        //if(termID == 0)
        //{
        //cout<<sub<<" ";
        //continue;
        //}
        while (iss>>dd)
            wordEmbedding[termID].push_back(dd);
    }
    cout<<"ReadWordEmbeddingFile END\n";
}


bool pairCompare(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem)
{
    return firstElem.first > secondElem.first;
}

void showNearerTerms2QueryVecInW2V(DocStream *qs,RetMethod *myMethod ,Index *ind, int avgOrMax)
{
    ofstream inputfile;
    inputfile.open("outputfiles/termsNearer2QueryWordsMaximum.txt");

    qs->startDocIteration();
    TextQuery *q;
    while(qs->hasMore())//queries
    {
        Document* d = qs->nextDoc();
        q = new TextQuery(*d);
        QueryRep *qr = myMethod->computeQueryRep(*q);
        TextQueryRep *textQR = (TextQueryRep *)(qr);

        //cout<<wordEmbedding.size()<<" "<<ind->termCountUnique()<<endl;



        vector<vector<double> > queryTerms;
        double counter =0 ;
        textQR->startIteration();
        while(textQR->hasMore())
        {

            counter += 1;
            QueryTerm *qt = textQR->nextTerm();
            if(wordEmbedding.find(qt->id()) != wordEmbedding.end())
            {
                queryTerms.push_back(wordEmbedding[qt->id()]);
            }
            else
            {
                delete qt;
                continue;
            }

            inputfile<<ind->term(qt->id())<<" ";
            delete qt;
        }

        inputfile<<" : ";
        vector<double> queryAvg( myMethod->W2VecDimSize);

        if(avgOrMax == 0)
        {
            for(int i =0 ; i< queryTerms.size() ; i++)
            {
                for(int j = 0 ;j<queryTerms[i].size() ; j++)
                    queryAvg[j] += queryTerms[i][j];
            }
            for(int i = 0 ; i < queryAvg.size() ;i++)
                queryAvg[i] /= counter;
        }else if (avgOrMax == 1)
        {
            for(int i =0 ; i< queryTerms.size() ; i++)
            {
                for(int j = 0 ;j<queryTerms[i].size() ; j++)
                {
                    if(queryAvg[j] < queryTerms[i][j])
                        queryAvg[j] = queryTerms[i][j];
                }
            }

        }


        vector<double>dtemp;
        vector<pair<double,int> >simTermid;
        for(int i = 1 ; i < ind->termCountUnique() ; i++)
        {
            if(wordEmbedding.find(i) != wordEmbedding.end())
                dtemp = wordEmbedding[i];
            else
                continue;


            double sim = myMethod->cosineSim(queryAvg,dtemp);
            simTermid.push_back(pair<double,int>(sim,i));
        }
        std::sort(simTermid.begin() , simTermid.end(),pairCompare);

        for(int i = 0 ; i < 10 ; i++)
            inputfile <<"( "<< ind->term(simTermid[i].second)<<" , "<<simTermid[i].first<<" ) ";

        inputfile<<endl;
        simTermid.clear();dtemp.clear();queryAvg.clear();


        delete textQR;
        delete q;
    }

    //delete qr;
    //delete d;

    inputfile<<endl;
    inputfile.close();

}

void computeQueryAvgVec(Document *d,RetMethod *myMethod )
{
    TextQuery *q = new TextQuery(*d);
    QueryRep *qr = myMethod->computeQueryRep(*q);
    TextQueryRep *textQR = (TextQueryRep *)(qr);
    vector<vector<double> > queryTerms;
    //double counter = 0;
    const std::map<int,vector<double> >::iterator endIt = wordEmbedding.end();
    textQR->startIteration();
    while(textQR->hasMore())
    {
        //counter += 1;
        QueryTerm *qt = textQR->nextTerm();
        const std::map<int,vector<double> >::iterator it = wordEmbedding.find(qt->id());

        if(it != endIt)//found
        {
            queryTerms.push_back(it->second);
            //cerr<<"F:"<<it->first<<","<<it->second.size()<<" ";
        }
        else
        {
            //cerr<<"NotFound ";
            delete qt;
            continue;
        }
        delete qt;
    }
    vector<double> queryAvg( myMethod->W2VecDimSize ,0.0);
    for(int i = 0 ; i< queryTerms.size() ; i++)
    {
        for(int j = 0 ; j < myMethod->W2VecDimSize /*queryTerms[i].size()*/ ; j++)
            queryAvg[j] += queryTerms[i][j];
    }
    for(int i = 0 ; i < queryAvg.size() ;i++)
        queryAvg[i] /= (double)(queryTerms.size());

    //cerr<<endl<<queryTerms.size()<<"\n\n\n";

    myMethod->Vq.clear();
    //myMethod->Vq.assign(myMethod->W2VecDimSize ,0.0);
    //myMethod->Vq = queryAvg;
    myMethod->Vq.assign( queryAvg.begin(),queryAvg.end());

    delete qr;
    delete q;
    //delete textQR;

}

void showNearerTermInW2V(DocStream *qs,RetMethod *myMethod ,Index *ind)
{
    ofstream inputfile;
    inputfile.open("outputfiles/similar2QueryWord.txt");



    qs->startDocIteration();
    TextQuery *q;
    while(qs->hasMore())//queries
    {
        Document* d = qs->nextDoc();
        q = new TextQuery(*d);
        QueryRep *qr = myMethod->computeQueryRep(*q);

        TextQueryRep *textQR = (TextQueryRep *)(qr);


        textQR->startIteration();
        while(textQR->hasMore())//query terms
        {
            vector<pair<double,int> >simTermid;
            vector<double> qtemp,dtemp;
            QueryTerm *qt = textQR->nextTerm();

            if(wordEmbedding.find(qt->id()) != wordEmbedding.end())
                qtemp = wordEmbedding[qt->id()];
            else
                continue;

            cout<<wordEmbedding.size()<<" "<<ind->termCountUnique()<<endl;

            for(int i =1 ; i< ind->termCountUnique() ; i++)
            {

                if(wordEmbedding.find(i) != wordEmbedding.end())

                {
                    dtemp = wordEmbedding[i];
                    //cout<<"here!\n";
                }
                else
                {
                    //cout<<"here22222!\n";
                    continue;
                }
                //if(dtemp.size() == 0 )
                //    continue;


                double sim = myMethod->cosineSim(qtemp,dtemp);
                simTermid.push_back(pair<double,int>(sim,i));
            }
            std::sort(simTermid.begin() , simTermid.end(),pairCompare);


            inputfile<<ind->term(qt->id())<<": ";
            //for(int i=simTermid.size()-1 ; i> simTermid.size()- 5;i--)
            for(int i = 0 ; i < 5 ; i++)
                inputfile <<"( "<< ind->term(simTermid[i].second)<<" , "<<simTermid[i].first<<" ) ";

            inputfile<<endl;
            delete qt;
            simTermid.clear();
            qtemp.clear();
            dtemp.clear();
        }

        delete textQR;
        delete q;
        //delete qr;
        //delete d;
    }
    inputfile<<endl;
    inputfile.close();
}

void readStopWord(Index *ind)
{
    string mterm;
    ifstream input("dataSets/stops_en.txt");
    if(input.is_open())
    {
        int cc=0;
        while(getline(input,mterm))
        {
            cc++;
            //std::cout<<mterm<<" aaa ";
            if(mterm.size()>1)
                mterm.erase(mterm.size()-1,mterm.size());
            //std::cout<<" ttt "<<mterm<<endl;
            stopWords.insert( ind->term(mterm) );
        }
        cout<<cc<<" SW size: "<<stopWords.size()<<endl;

        input.close();
    }else
    {
        cerr<<"FILE NOT OPENED";
    }
    stopWords.erase(stopWords.find(0));

}

#if 0
#include "pugixml.hpp"
using namespace spugi;
void ParseQuery(){
    ofstream out("topics.txt");
    xml_document doc;
    xml_parse_result result = doc.load_file("/home/hossein/Desktop/lemur/DataSets/Infile/Data/q_en.xml");// Your own path to original format of queries
    xml_node topics = doc.child("topics");
    for (xml_node_iterator topic = topics.begin(); topic != topics.end(); topic++){
        xml_node id = topic->child("identifier");
        xml_node title = topic->child("title");
        xml_node desc = topic->child("description");
        xml_node nar = topic->child("narrative");
        out<<"<DOC>"<<endl;
        out<<"<DOCNO>"<<id.first_child().value()<<"</DOCNO>"<<endl;
        out<<"<TEXT>"<<endl;
        out<<title.first_child().value()<<endl;
        out<<"</TEXT>"<<endl;
        out<<"</DOC>"<<endl;

    }
    printf("Query Parsed.\n");
}
#endif
#endif

