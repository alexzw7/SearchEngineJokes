import pandas as pd 
import csv

def generateQueryDFrame():
    queries = []
    queries.append(['0', "funny life stories"])
    queries.append(['1', "jokes about sports"])
    queries.append(['2', "college experience"])
    queries.append(['3', "friend jokes"])
    queries.append(['4', "shows and movies"])
    queries.append(['5', "story time"])
    queries.append(['6', "job and work jokes"])
    queries.append(['7', "family"])
    queries.append(['8', "computer science"])
    queries.append(['9', "car jokes"])

    df = pd.DataFrame(queries, columns=['qid', 'query'])
    return df

def getQRels():
    rels = []
    dist = [0, 0, 0, 0, 0]
    with open('qrels.csv', 'r') as csvfile:
        next(csvfile)
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            dist[int(row[2]) - 1] += 1
            rels.append(row)
            rels[-1][2] = int(rels[-1][2])
    df = pd.DataFrame(rels, columns=['qid', 'docno', 'label'])
    df['label'] = df['label'].astype(float)
    return df