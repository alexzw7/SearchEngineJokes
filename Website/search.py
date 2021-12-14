import pyterrier as pt
import os
import json
import testing
import fastrank
import csv
import evaluate


def buildIndex():
    if not pt.started():
        pt.init(boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT'])

    # list of filenames to index
    files = pt.io.find_files("./trec_files/docs")

    # build the index
    indexer = pt.TRECCollectionIndexer("./pyterrier_indexes/docs", meta={'docno': 26, 'body': 4096}, meta_tags={'body' : 'TEXT'})
    indexref = indexer.index(files)

    # load the index, print the statistics
    index = pt.IndexFactory.of(indexref)
    return index

def readDocs():
    jokes = []
    dict = {}
    with open("docs.json") as file:
        jokes = json.load(file)
    for i in range(len(jokes)):
        dict[jokes[i]['docno']] = jokes[i]['text']
    return dict

def loadIndex():
    if not pt.started():
        pt.init(boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT'])
    if os.path.exists("./pyterrier_indexes/docs/data.properties"):
        index_ref = pt.IndexRef.of("./pyterrier_indexes/docs/data.properties")
        index = pt.IndexFactory.of(index_ref)
        return index
    else:
        return buildIndex()

def search(query, pipeline):
    if not pt.started():
        pt.init(boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT'])
    results = pipeline.search(query).iloc[0:25]['docno']
    return results

def bm_pipe(index):
    if not pt.started():
        pt.init(boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT'])
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"c": 0.9}, metadata=["docno", "body"])
    return bm25

def rm_pipe(index):
    if not pt.started():
        pt.init(boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT'])
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"c": 0.9}, metadata=["docno", "body"])
    rm3 = bm25 >> pt.rewrite.RM3(index) >> bm25
    return rm3

def fit_l2r(index):
    if not pt.started():
        pt.init(boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT'])
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"c": 0.9})
    rm3 = pt.rewrite.RM3(index)
    topics = testing.generateQueryDFrame()
    train_topics = topics.iloc[0:8]
    qrels = testing.getQRels()
    features = (bm25 % 50) >> pt.text.get_text(index, "body") >> (
        pt.transformer.IdentityTransformer()
        **
        (rm3 >> bm25)
        **
        pt.BatchRetrieve(index, wmodel="CoordinateMatch")
        **
        pt.BatchRetrieve(index, wmodel="PL2")
        **
        pt.BatchRetrieve(index, wmodel="DFR_BM25")
        **
        pt.BatchRetrieve(index, wmodel="DFIZ")
    )
    train_request = fastrank.TrainRequest.coordinate_ascent()

    params = train_request.params
    params.init_random = True
    params.normalize = True
    params.seed = 1234567

    ca_pipe = features >> pt.ltr.apply_learned_model(train_request, form='fastrank')
    ca_pipe.fit(train_topics, qrels)

    return ca_pipe

def eval(ranker, docs):
    topics = testing.generateQueryDFrame()
    test_topics = topics.iloc[8:10]
    docnos = ranker.search(test_topics.iloc[1]['query']).iloc[0:25]['docno']
    qrels = {}
    with open('eval.csv', 'r') as csvfile:
        next(csvfile)
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if int(row[0]) == 9:
                qrels[row[1]] = int(row[2])
    ranking = []
    optimal = sorted(qrels.values(), reverse=True)
    optimal = evaluate.getOptimal(optimal, 3)
    for docno in docnos:
        if docno in qrels:
            ranking.append(qrels[docno])
        else:
            print(docno)
            print(docs[docno])
    evaluate.ndcg(ranking, optimal, min(len(optimal), 25))
    evaluate.getPrecision(ranking)
    return docnos

def annotate(ranker):
    topics = testing.generateQueryDFrame()
    test_topics = topics.iloc[8:10]
    docnos = ranker.search(test_topics.iloc[1]['query']).iloc[0:25]['docno']
    qrels = {}
    with open('eval.csv', 'r') as csvfile:
        next(csvfile)
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if int(row[0]) == 9:
                qrels[row[1]] = int(row[2])
    missing = []
    with open('eval.csv', 'a') as csvfile:
        csv_writer = csv.writer(csvfile)
        for doc in docnos:
            if doc not in qrels:
                missing.append(doc)
                csv_writer.writerow([9, doc, -1])
    return missing