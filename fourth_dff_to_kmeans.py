#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:35:54 2021
load data from 'entireDF_DFF.csv' and use Column H which is abs dft output. 
This script will generate cluster
@author: jasonlee
"""



import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import os
import math
import ast
from scipy.fft import fft, ifft
from sklearn.cluster import KMeans
from ast import literal_eval

def DFtoCsv(filename, df):
	cwd = os.getcwd()
	out_dir = os.path.join(cwd,filename+".csv")
	print(out_dir)
	df.to_csv(out_dir, index = None)

def NPtoCsv(nparray,fn):
    cwd = os.getcwd()
    fn = os.path.join(cwd,fn)
    pd.DataFrame(nparray).to_csv(fn+".csv")
	
# elbow method
def distortionFinder(X):
    # X = X[:,[0,1]]
    distortions = []
    for i in range(1,30):
        km = KMeans(n_clusters=i, random_state=1).fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1, 30), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.savefig('./dff_elbow.png', dpi=300)
    plt.show()
	
	
# get all the useful output with fixed seed.
# input : #ofCluster=20
# output : centroids , labels , inertia, distances, iVal, varianceVal , retVal , retCount
def kmeanFinal(arr, K, rand_state):
	kmeans = KMeans(n_clusters=K, random_state=rand_state).fit(arr)
	kmeans_transform = kmeans.transform(arr)
# 	km = KMeans(n_clusters=20, random_state=1)
# 	distances = kmeans.fit_transform(arr)
# 	print(distances)
	
	
	# iVal = total counts of vector points
    # label = centroid index
	centroids = kmeans.cluster_centers_
	labels = kmeans.labels_
	inertia = kmeans.inertia_
	
	iVal=0
	varianceVal = 0
	retVal = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	retCount = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for label in kmeans.labels_:
		print(label)
		varianceVal = varianceVal + kmeans_transform[iVal][label]*kmeans_transform[iVal][label]
		retVal[label] += kmeans_transform[iVal][label]*kmeans_transform[iVal][label] 
		retCount[label] += 1
		iVal = iVal + 1
		
	return centroids , labels , inertia, kmeans_transform, iVal, varianceVal , retVal , retCount


# Not gon use
# def strToArr(df):
#     # df = df['abs_y_list']
#     arr = []
#     df['abs_y_list'] = df['abs_y_list'].apply(literal_eval)
#     for index, row in df.iterrows():
#         arr.append(row['abs_y_list'])
#         # print(type(row['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']))
#     return arr


def main():
# 	dir_name = "haydnAnalysis"
# 	cwd = os.getcwd()
# 	directory = os.path.join(cwd,dir_name)
	entireDF = pd.read_csv("entireDF_DFF.csv")
	entireDF.head()
# 	replace [nan,] to 
	entireDF.at[0,'abs_y_list'] = "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
	dffCol = entireDF['abs_y_list']



# 	random state param to fix the output
	rand_state = 50
	arr = []
	for row in dffCol:
		i = ast.literal_eval(row)
		arr.append(i)

	entireVP = arr
	type(entireVP)
	vectorPointE = np.asarray(entireVP)
	#check optimized number of centroid
	type(vectorPointE)
	
# 	This is abs dff result vec point. Start clustring from here
	vectorPointE
# 	find out the proper cluster# using elbow method
	distortionFinder(vectorPointE)
# 	centroids , labels , inertia, kmeans_fit_transform_distance, iVal, varianceVal , retVal , retCount
	centroidsVectorE, labelsArrayE, inertiaValueE, kmeans_fit_transform_distance, iValE, varianceValE, retVal, retCount = kmeanFinal(vectorPointE,20,rand_state )
	
	for x in range(0,20):
		retVal[x] = retVal[x] / retCount[x]
		print(x)
		
	
	randPrefix = "_rand_state_" + str(rand_state)
	NPtoCsv(centroidsVectorE,"centroidsVectorE"+randPrefix)
# 	NPtoCsv(labelsArrayE,"labelsArrayE"+randPrefix)
# 	NPtoCsv(inertiaValueE,"inertiaValueE"+randPrefix)
	NPtoCsv(kmeans_fit_transform_distance,"kmeans_fit_transform_distance"+randPrefix)
# 	NPtoCsv(iValE,"iValE"+randPrefix)
# 	NPtoCsv(varianceValE,"varianceValE"+randPrefix)
	NPtoCsv(retVal,"retVal"+randPrefix)
	NPtoCsv(retCount,"retCount"+randPrefix)
	
	# selected column index: 1, 6, 7
# 	new = old.iloc[: , [1, 6, 7]].copy() 
	selected_columns = entireDF.iloc[:,[0,1,2,3,5,7]].copy()
	seriesE = pd.Series(labelsArrayE)
	selected_columns.insert(2, "CentroidLabelIndex_entireUnits", seriesE)
	NPtoCsv(selected_columns, "ClusteringData"+randPrefix)
    





if __name__ == '__main__':
    main()


