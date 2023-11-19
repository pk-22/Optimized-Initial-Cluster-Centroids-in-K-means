import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cosine
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
import os
from sklearn.metrics import silhouette_score
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import Vectors
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import pyspark
from pyspark import SparkContext

def makeNonZero(singleRow):
    count = np.count_nonzero(singleRow == 0)
    if count == len(singleRow):
        for i in range(len(singleRow)):
            singleRow[i]=0.01
    return singleRow

def findMinimum(singleColumn):
    return np.min(singleColumn)

# For feature vector
def scalable_feature_extraction(singleRow):
    distance = euclidean(singleRow, reference_vector)
    angles =   cosine(singleRow, reference_vector)
    return distance + angles

def findMedianList(singleColumn):
    medianList =[]
    for i in partition:
        medianList.append(np.median(singleColumn[i[0]:i[1]]))
    return medianList

def findClusterNumber(singleRow):
    distances = np.linalg.norm(singleRow-centroid,axis=1)
    idx = np.argmin(distances)
    return idx


if _name_ == '_main_':
    sc = SparkContext(appName="Scalable Feature Vector Extraction", master = "local[*]")

    # for file_name in os.listdir("featurevector"):
    file_name = "Haberman.csv"
    df = pd.read_csv("featurevector"+"/"+file_name, header=None)
    ground_truth_labels = (pd.read_csv("labels/"+file_name, header=None)).values.tolist()
    ground_truth_labels = [item for sublist in ground_truth_labels for item in sublist]
    scaler = StandardScaler()

    # Scale the features (excluding the 'cluster' column)
    scaled_data = scaler.fit_transform(df.iloc[:,:])
    numRow = len(scaled_data)
    numCol = len(scaled_data[0])
    rowWiseDataRDD = sc.parallelize(scaled_data)

    # Row wise cleaned data
    cleanedData = np.array((rowWiseDataRDD.map(lambda x: makeNonZero(x))).collect())
    
    columnWiseData = np.transpose(cleanedData)

    columnWiseDataRDD = sc.parallelize(columnWiseData)
    reference_vector = columnWiseDataRDD.map(lambda x:findMinimum(x))
    reference_vector = reference_vector.collect()

    rdd1 = sc.parallelize(cleanedData)
    combined_score = rdd1.map(lambda x:scalable_feature_extraction(x))

    indexedRDD = combined_score.zipWithIndex()
    sortedRDD = indexedRDD.sortBy(lambda x: x[0])

    sortedVAlues = sortedRDD.map(lambda x: x[1])
    modified_data = cleanedData[sortedVAlues.collect()]

    modifiedColumnWiseData = np.transpose(modified_data)
    columnWiseDataRDD = sc.parallelize(modifiedColumnWiseData)

    K = 2
    split_indices = np.linspace(0, numRow, num=K + 1, dtype=int)
    partition = []
    for i in range(1,len(split_indices)):
        partition.append([split_indices[i-1], split_indices[i]])

    centroid = np.transpose(np.array((columnWiseDataRDD.map(lambda x: findMedianList(x))).collect()))

    for i in range(2):
        clusterIndex = np.array((rdd1.map(lambda x: findClusterNumber(x))).collect())
        newCentre = [[0]*numCol]*K
        numberOfPoints = [0]*K
        for j in range(numRow):
            newCentre[clusterIndex[j]]+=cleanedData[j]
            numberOfPoints[clusterIndex[j]]+=1
        for j in range(K):
            if numberOfPoints[j]!=0:
                newCentre[j]/=numberOfPoints[j]
        centroid = newCentre

    # Calculate the silhouette score
    silhouette_avg = silhouette_score(cleanedData, clusterIndex)
    rand_index = adjusted_rand_score(ground_truth_labels, clusterIndex)
    # rand_index = (1+rand_index)/2
    print("Rand Index:", rand_index*100)
    nmi = normalized_mutual_info_score(ground_truth_labels, clusterIndex)
    print("Normalized Mutual Info Score:", nmi*100)
    for i in range(10):
        print(" ")
    print(silhouette_avg)