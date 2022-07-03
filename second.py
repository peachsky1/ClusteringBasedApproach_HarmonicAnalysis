#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:32:17 2021
combining all the splitted files into one csv fil 
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



def toCsv(filename, df):
	cwd = os.getcwd()
	out_dir = os.path.join(cwd,filename+".csv")
	print(out_dir)
	df.to_csv(out_dir, index = None)



def main():

	dir_name = "haydnAnalysis/outputFiles"
	cwd = os.getcwd()
	directory = os.path.join(cwd,dir_name)

	entireDF = pd.DataFrame()
	currDF = pd.DataFrame()


	# iterate dir
	for file in os.listdir(directory):
		if file.endswith(".csv"):
			filename = file
			print(filename)
			path = os.path.join(directory,filename)
			# defining engine to avoid memory overflow issue
			df= pd.read_csv(path, engine="python")
			df['filename'] = filename
			print(df)
			entireDF = entireDF.append(df)
		else:
			print("\"{}\" is not csv file".format(file))
	entireDF
	toCsv("haydnDF",entireDF)

	
	list1 = [2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
	list2 = [0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0]
	difference = []
	zip_object = zip(list1, list2)
	for l1, l2 in zip_object:
	    difference.append(l2-l1)
	print(difference)
	









if __name__ == '__main__':
    main()
