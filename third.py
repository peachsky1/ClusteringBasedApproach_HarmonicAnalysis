#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:32:17 2021
dft
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



def toCsv(filename, df):
	cwd = os.getcwd()
	out_dir = os.path.join(cwd,filename+".csv")
	print(out_dir)
	df.to_csv(out_dir, index = None)




def main():

	dir_name = "haydnAnalysis"
	cwd = os.getcwd()
	directory = os.path.join(cwd,dir_name)
	entireDF = pd.read_csv("haydnDF.csv")
# 	sampleDF = entireDF.head() 
	sampleDF = entireDF
	
	for col in sampleDF:
		columnSeriesObj = sampleDF[col]
		print('Colunm Name : ', col)
		print('Column Contents : ', columnSeriesObj.values)
# 		
# Column Contents :  
# ['[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]'
#  '[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]'
#  '[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]'
#  '[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]'
#  '[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]']
# 	empty dataset with col name
	column_names = ["C","C#","D","E-","E","F","F#","G","G#","A","B-","B"]
	triadDF = pd.DataFrame(columns = column_names)
	
	
	for row in sampleDF[['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']]:
		columnSeriesObj = sampleDF[row]
		for i in columnSeriesObj.values:
# 			Convert a string representation of list into list
			i = ast.literal_eval(i)
# 			print(type(i))
			print(i)
# 			arr = np.asarray(i)
			df_length = len(triadDF)
			triadDF.loc[df_length] = i
# 			
# triadDF
# Out[7]: 
#    C C#  D E-  E  F F#  G G#  A B-  B
# 0  0  0  0  2  0  0  0  0  0  0  0  0
# 1  0  0  1  0  0  1  0  0  0  0  0  0
# 2  0  0  0  1  0  0  0  1  0  0  0  0
# 3  0  0  0  0  0  0  0  1  0  0  1  0
# 4  0  0  0  1  0  0  0  1  0  0  0  0
# 			
	triadDF
	print("Dataset:")
	print(triadDF);
	# Calculate the difference between rows - By default, periods = 1
	difference = triadDF.diff(axis=0);
	print("Difference between rows(Period=1):");
	print(difference);
	toCsv("difference", difference)


# 	
# 	list1 = [2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 	list2 = [0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# 	difference = []
# 	zip_object = zip(list1, list2)
# 	for l1, l2 in zip_object:
# 	    difference.append(l2-l1)
# 	print(difference)
	
	diff_list = difference.values.tolist()
# 	https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
	y_list = []
	yinv_list = []
	for i in diff_list:
		x = np.array(i)
		y = fft(x)
# 		y
		yinv = ifft(y)
# 		yinv
		y_list.append(y.tolist())
		yinv_list.append(yinv.tolist())
		
	y_list
	yinv_list
	
	
	abs_y_list = []
	abs_yinv_list = []
	for i in diff_list:
		x = np.array(i)
		y = fft(x)
# 		y
		yinv = ifft(y)
# 		yinv
		y = np.absolute(y)
		yinv = np.absolute(yinv)
	
		abs_y_list.append(y.tolist())
		abs_yinv_list.append(yinv.tolist())
		
	abs_y_list
	abs_yinv_list
	
	columns = ['y_list','abs_y_list', 'yinv_list', 'abs_yinv_list']
	dffDF = pd.DataFrame(data=list(zip(y_list, abs_y_list, yinv_list, abs_yinv_list)),columns=columns)

	sampleDF
	dffDF
	entireDF = pd.concat([sampleDF, dffDF], axis=1)
	toCsv("entireDF_DFF", entireDF)
	







if __name__ == '__main__':
    main()
