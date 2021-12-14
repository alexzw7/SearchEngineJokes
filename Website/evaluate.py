import json
import re
import math

def ndcg(r, o, n):
    # Calculates NDCG at rank n
    print('NDCG at ' + str(n) + ':')
    dcg_r = 0
    dcg_o = 0
    for i in range(n):
        dcg_r += r[i] / max(math.log2(i + 1), 1.0)
        dcg_o += o[i] / max(math.log2(i + 1), 1.0)
    print(dcg_r / dcg_o)
    return dcg_r / dcg_o

def getOptimal(orig, k):
    # Returns optimal results given relevant floor k
    optimal = []
    for x in range(len(orig)):
        if orig[x] >= k:
            optimal.append(orig[x])
    return sorted(optimal, reverse=True)

def getPrecision(r):
    # Calculates precision using document scores in r
    print('Precision:')
    count = 0
    for num in r:
        if num >= 3:
            count += 1
    print(count / len(r))
    return count / len(r)
