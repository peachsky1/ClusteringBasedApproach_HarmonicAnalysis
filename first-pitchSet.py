#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:32:17 2021
Preprocess and getting triad label.
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



def octScalePitch(c):
	s = c['Chord']
	l = s[21:-1].split()
	# print(l)
	return l





# This method take chuck of dataframe which has been splitted by user input interval of offset.
# Will return the set of pitchs
def pitchSet(df_x):

	head_node = df_x.head(1)
	tail_node = df_x.tail(1)
	offset_begin = head_node['offset'].iloc[0]
	offset_end = tail_node['offset'].iloc[0]
	offsetRange = str(offset_begin) + " to " + str(offset_end)
	
	pitch_list=[]
	oct_list = []
	for index, row in df_x.iterrows():
		k = row['PCsInNormalForm']
		m = row['octScalePitch']
		k = ast.literal_eval(k)
		pitch_list +=k
		oct_list +=m
	# k.sort(key=lambda x:(not x.islower(), x))
	# sorted(k)
	k = k.sort()
	# sorted(pitch_list)
	# sorted(oct_list)

	# print(pitch_list)
	# print(oct_list)
	# sorted(oct_list_set)
	# sorted(list_set)

	for x in range(0,len(oct_list)):
		oct_list[x] = ''.join([c for c in oct_list[x] if c in '1234567890ABCDEFGabcdefg-#'])
		# print(oct_list[x])

	list_set = set(pitch_list)
	oct_list_set = set(oct_list)
	list_set = sorted(list_set)
	oct_list_set = sorted(oct_list_set)

	pitch_count_list = [0,0,0,0,0,0,0,0,0,0,0,0]

# 	duplicates has been removed and sorted.
	# print(type(oct_list_set))
	for x in oct_list_set:
		# x is one pitch string.
		i = indexFinder(x)
		if i != None:
			pitch_count_list[i] += 1
	# print(pitch_count_list)
# '[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]'
	data_row = [(offsetRange,list_set,oct_list_set,pitch_count_list)]
	# print(pitch_count_list)

	df_new = pd.DataFrame(data_row, columns=['offsetRange','pitchSet','oct_list_set','[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]'])

	# returning df which has 2 col('offsetRange', 'pitchSet') of CHUNK splitted by user custom offset interval 
	return df_new , pitch_count_list



def indexFinder(x):
	# x is string like 'E3' or 'E-5'
	x = str(x)

	x = ''.join([c for c in x if c in 'ABCDEFGabcdefg-#'])
	# C#, E-, F#, G#, B-
	if len(x) == 2:
		if x == 'C#':
			return 1
		elif x == 'E-':
			return 3
		elif x == 'F#':
			return 6
		elif x == 'G#':
			return 8
		elif x == 'B-':
			return 10
		else:
			return None

	# C, D, E, F, G, A, B
	elif len(x) ==1:
		if x == 'C':
			return 0
		elif x == 'D':
			return 2
		elif x == 'E':
			return 4
		elif x == 'F':
			return 5
		elif x == 'G':
			return 7
		elif x == 'A':
			return 9
		elif x == 'B':
			return 11
		else:
			return None

	print(x)



def csv_gen(df_X, directoryName, pieceName, composerName, optionalString):
	directory = directoryName
	cwd = os.getcwd()
	title = str(pieceName)
	composer = str(composerName)
	opt = str(optionalString)
	filename = opt +'pitchSet_' + title + '_by_' + composer + '.csv'
	dir_path = os.path.join(cwd,directory)
	new_path = os.path.join(dir_path,composer)
	# print(df_X)
	if not os.path.exists(new_path):
		os.mkdir(new_path)
		print("=======================================")
		print("directory \"{}\" has been created".format(new_path))
		print("=======================================")
	final_dir = os.path.join(new_path,filename)
	df_X.to_csv(final_dir, index = None)
	print("filename: \"{}\" has been successfully created :^D".format(filename))



def vectorGen():
	# input list is vector which is length of 12
	majorC = [1,0,0,0,1,0,0,1,0,0,0,0]
	minorA = [1,0,0,0,1,0,0,0,0,1,0,0]
	length = len(majorC)
	majorRot = []
	minorRot = []
	for x in range(0,length):
		majorRot.append(rotate(majorC,x))
		minorRot.append(rotate(minorA,x))

	# print(majorRot)
	# print(minorRot)
	return majorRot, minorRot


# n = 0 = 12
def rotate(l,n):
	return l[-n:] + l[:-n]

def triadLabeling(df1, df2):
	x = df1['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']
	coef_list = []
	for e in range(0,len(df2.index)):
		coef_list.append(np.corrcoef(x,df2['triad'][e])[0][1])
		# print(e)
		# print(df2[])
	# print(coef_list)
	max_coef = max(coef_list)
	label_index = coef_list.index(max_coef)
	max_coef_label = df2['label'][label_index]

	# print(max_coef)
	# print(max_coef_label)
	return max_coef_label


# df1(offsetRange, [C,C#,D,E-,E,F,F#,G,G#,A,B-,B]), df2(triad, label)
def labelCoef(df1, df2):
	# print(df1)
	# print(df2)
	x = df1['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']
	coef_list = []
	# for i in range(len(df1)):
	# 	next_x = df1[i,:]
	# 	print(next_x)
	# 	coef_list = []
	# 	for e in range(len(df2)):
	# 		next_cor = np.corrcoef(next_x, e)[0][1]
	# 		coef_list.append(next_cor)
	# 	label_index = np.argmax(np.array(coef_list))
	# 	max_corr = coef_list[label_index]
	# 	label = df2['label'][label_index]
	# 	# print(coef_list)
	# 	# print(label_index)
	# 	# print(max_corr)
	# return max_corr 


	for e in range(0,len(df2.index)):
		coef_list.append(np.corrcoef(x,df2['triad'][e])[0][1])
	max_coef = max(coef_list)
	label_index = coef_list.index(max_coef)
	max_coef_label = df2['label'][label_index]
	# print("herehere")
	# print(max_coef)
	# print(max_coef_label)
	return max_coef

def coefList(df1, df2):
	x = df1['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']
	coef_list = []
	for e in range(0,len(df2.index)):
		coef_list.append(np.corrcoef(x,df2['triad'][e])[0][1])
		# print(e)
		# print(df2[])
	# print(coef_list)
	# max_coef = max(coef_list)
	# label_index = coef_list.index(max_coef)
	# max_coef_label = df2['label'][label_index]

	# print(max_coef)
	# print(max_coef_label)
	return coef_list




def main():

	dir_name = "haydnAnalysis/haydnStrQminuetsEarly"
	cwd = os.getcwd()
	directory = os.path.join(cwd,dir_name)

	entireDF = pd.DataFrame()
	entireVP =[]
	offset_term = 1

	# iterate dir
	for file in os.listdir(directory):
		if file.endswith(".csv"):
			filename = file
			print(filename)
			path = os.path.join(directory,filename)
			# defining engine to avoid memory overflow issue
			df= pd.read_csv(path, engine="python")
			# print(df)
			df['octScalePitch'] = df.apply(octScalePitch, axis = 1)

			piece_name = df['file'].iloc[0]
			composer_name = df['Composer'].iloc[0]

			last_offset = df['offset'].max()
			if (last_offset%offset_term)==0:
				chunk_count = math.ceil(last_offset/offset_term+0.00001)
			elif (last_offset%offset_term) !=0:
				chunk_count = math.ceil(last_offset/offset_term)
			last_offset = chunk_count * offset_term

			chunk_list = list(range(0,last_offset,offset_term))
			print(chunk_list)
			df_pitchSetOnly = pd.DataFrame()
			# k = None
			vectorArrRet = []
			df_att2 = pd.DataFrame(columns=['offsetRange','pitchSet','oct_list_set','[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]'])
			for x in chunk_list:
				k = df[(df['offset']>=x)&(df['offset']<x+offset_term)]
				# print(k)
				if not k.empty:
					df_row, vectorArr = pitchSet(k)
					# asArray = np.asarray(vectorArr)
					# print(asArray)
					vectorArrRet.append(vectorArr)
					entireVP.append(vectorArr)
					# print(vectorArr)
					df_att = pd.DataFrame(columns=['offsetRange','pitchSet','oct_list_set','[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]'])
					for z in range(len(k)):
						# print(z)
						df_att = df_att.append(df_row, ignore_index=True)
					c1 = df_row['offsetRange']
					c2 = df_row['pitchSet']
					c3 = df_row['oct_list_set']
					c4 = df_row['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']
					df_pitchSetOnly = df_pitchSetOnly.append(df_row, ignore_index=True)
					df_att2 = df_att2.append(df_att,ignore_index=True)
			# print(vectorArrRet)
			majorRot, minorRot = vectorGen()
			df['offsetRange'] = df_att2['offsetRange']
			df['pitchSet'] = df_att2['pitchSet']
			df['oct_list_set'] = df_att2['oct_list_set']
			df['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]'] = df_att2['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']

			# print(df)
			optionalString = ""

			csv_gen(df_pitchSetOnly, directory, piece_name, composer_name,"pitchSetOnly_")
			csv_gen(df, directory, piece_name, composer_name,optionalString)

			# use df, majorRot, minorRot from now here.
			df_p = df_pitchSetOnly
			df_major = pd.DataFrame()
			df_minor = pd.DataFrame()
			df_triad = pd.DataFrame()
			df_major['triad'] = majorRot
			df_minor['triad'] = minorRot
			major_label = ['cMajor','c#Major','dMajor','e-Major','eMajor','fMajor','f#Major','gMajor','g#Major','aMajor','b-Major','bMajor']
			minor_label = ['cMinor','c#Minor','dMinor','e-Minor','eMinor','fMinor','f#Minor','gMinor','g#Minor','aMinor','b-Minor','bMinor']
			df_major['label'] = major_label
			df_minor['label'] = minor_label
			df_triad = df_major.append(df_minor, ignore_index=True)
			df_p = df_p.drop(['pitchSet','oct_list_set'], axis=1)

			# create label on df_p
			df_p['triadLabel']= df_p.apply(triadLabeling,args=(df_triad,), axis = 1)
			df_p['triadCoef']= df_p.apply(labelCoef,args=(df_triad,), axis = 1)
			df_p['coefList']= df_p.apply(coefList,args=(df_triad,), axis = 1)

			# print(df_p) 
			outfileDir = "haydnAnalysis/outputFiles"
			outfileDirFull = os.path.join(cwd,outfileDir)
			out_dir = os.path.join(outfileDirFull, filename)
			df_p.to_csv(out_dir, index = None)

			# print(type(vectorArrRet))
			# this vectorpoints type is np array. it was list.
			vectorPoints = np.asarray(vectorArrRet)
			# print(type(vectorPoints))

			# print(vectorPoints)
			# input, number of array, clusters
			

		else:
			print("\"{}\" is not csv file".format(file))





        
    
    


if __name__ == '__main__':
    main()
