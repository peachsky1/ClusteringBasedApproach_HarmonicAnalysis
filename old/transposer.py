import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import os
import math
import ast

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# 1st param = dataframe, 2nd param String(title)
# splitter method find chuck of data set of each title
def splitter(dataframe, title):
	df_x = dataframe[dataframe["file"]==title]
	return df_x

# taking input file as return value(chuck of data frame) to creat csv file
def csv_generator(df_X, d):
	directory = d
	cwd = os.getcwd()
	title = str(df_X["file"].iloc[0])
	composer = str(df_X["Composer"].iloc[0])
	filename = title + ' by ' + composer + '.csv'
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

# n = 0 = 12
def rotate(l,n):
	return l[-n:] + l[:-n]

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

def octScalePitch(c):
	s = c['Chord']
	l = s[21:-1].split()
	# print(l)
	return l

# df1(offsetRange [,C,C#,D,E-,E,F,F#,G,G#,A,B-,B]), df2(triad, label)
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

def filenameAttach(df,fName):
	return fName



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

def histogram(df):
	major_label = ['cMajor','c#Major','dMajor','e-Major','eMajor','fMajor','f#Major','gMajor','g#Major','aMajor','b-Major','bMajor']
	minor_label = ['cMinor','c#Minor','dMinor','e-Minor','eMinor','fMinor','f#Minor','gMinor','g#Minor','aMinor','b-Minor','bMinor']
			
	# Make a separate list for each triad
	# Major
	x1 = list(df[df['triadLabel'] == 'cMajor']['triadCoef'])
	x2 = list(df[df['triadLabel'] == 'c#Major']['triadCoef'])
	x3 = list(df[df['triadLabel'] == 'dMajor']['triadCoef'])
	x4 = list(df[df['triadLabel'] == 'e-Major']['triadCoef'])
	x5 = list(df[df['triadLabel'] == 'eMajor']['triadCoef'])
	x6 = list(df[df['triadLabel'] == 'fMajor']['triadCoef'])
	x7 = list(df[df['triadLabel'] == 'f#Major']['triadCoef'])
	x8 = list(df[df['triadLabel'] == 'gMajor']['triadCoef'])
	x9 = list(df[df['triadLabel'] == 'g#Major']['triadCoef'])
	x10 = list(df[df['triadLabel'] == 'aMajor']['triadCoef'])
	x11 = list(df[df['triadLabel'] == 'b-Major']['triadCoef'])
	x12 = list(df[df['triadLabel'] == 'bMajor']['triadCoef'])
	# Minor
	x13 = list(df[df['triadLabel'] == 'cMinor']['triadCoef'])
	x14 = list(df[df['triadLabel'] == 'c#Minor']['triadCoef'])
	x15 = list(df[df['triadLabel'] == 'dMinor']['triadCoef'])
	x16 = list(df[df['triadLabel'] == 'e-Minor']['triadCoef'])
	x17 = list(df[df['triadLabel'] == 'eMinor']['triadCoef'])
	x18 = list(df[df['triadLabel'] == 'fMinor']['triadCoef'])
	x19 = list(df[df['triadLabel'] == 'f#Minor']['triadCoef'])
	x20 = list(df[df['triadLabel'] == 'gMinor']['triadCoef'])
	x21 = list(df[df['triadLabel'] == 'g#Minor']['triadCoef'])
	x22 = list(df[df['triadLabel'] == 'aMinor']['triadCoef'])
	x23 = list(df[df['triadLabel'] == 'b-Minor']['triadCoef'])
	x24 = list(df[df['triadLabel'] == 'bMinor']['triadCoef'])
	# Assign colors for each triad and the names
	colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00', '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075']
	names = major_label + minor_label
	         
	# Make the histogram using a list of lists
	# Normalize the df and assign colors and names

	print(names)
	# print("hi")
	plt.hist([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24], bins = int(180/15), normed=True,
	         color = colors, label=names)

	# Plot formatting
	plt.legend()
	plt.xlabel('Corr coef')
	plt.ylabel('Normalized traid frequency')
	plt.title('Side-by-Side Histogram with Triads')

# input df
def centroids_finder(arr, K):
	# print(arr)
	kmeans = KMeans(n_clusters=K, random_state=1).fit(arr)
	labels = kmeans.labels_
	inertia = kmeans.inertia_
	print(inertia)
	centroids = kmeans.cluster_centers_
	return centroids , labels , inertia

def listToArray(df):
	df = df['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']
	# coef_list = []
	for e in range(0,len(df.index)):

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

def main():
	
# =============================================================================
	# comand for input file or directory
	parser = argparse.ArgumentParser()
	# if input is directory which contains multiple csv files
	parser.add_argument("-i", "--inputdirname", required=True,
	    help="Filepath to the input file csv file.")
	# if input is individual file
	# parser.add_argument("-i", "--inputfilename", required=True,
	#     help="Filepath to the input file csv file.")
	args = parser.parse_args()
	# filename = args.inputfilename
	dir_name = args.inputdirname
# =============================================================================
    
	# dir_name = "sample"
	cwd = os.getcwd()
	directory = os.path.join(cwd,dir_name)

	offset_term = int(input("Enter offset value to split: "))
	transposeBy = int(input("Enter a value which transposed by: "))

	entireDF = pd.DataFrame()
	entireVP =[]
	# iterate dir
	for file in os.listdir(directory):
		if file.endswith(".csv"):
			filename = file
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
			# print(chunk_list)
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
			cwd = os.getcwd()
			out_dir = os.path.join(cwd,"outfile.csv")
			df_p.to_csv(out_dir, index = None)

			# print(type(vectorArrRet))
			# this vectorpoints type is np array. it was list.
			vectorPoints = np.asarray(vectorArrRet)
			# print(type(vectorPoints))

			# print(vectorPoints)
			# input, number of array, clusters
			
			centroidsVector, labelsArray, inertiaValue = centroids_finder(vectorPoints,10)
			series = pd.Series(labelsArray)
			df_p.insert(2, "CentroidLabelIndex_individualUnit", series)
			df_p = df_p.drop(['triadCoef', 'coefList'], axis=1)
			df_p['filename'] = df_p.apply(filenameAttach, args=(filename,), axis=1)
			df_p = df_p.reset_index()
			# index || offsetRange  || [C,C#,D,E-,E,F,F#,G,G#,A,B-,B] || CentroidLabelIndex || triadLabel || filename
			# print(df_p)
			entireDF = entireDF.append(df_p, ignore_index=True)
			# output = pd.DataFrame(centroidsVector)
			# output.to_csv(os.path.join(cwd,"vectorPoints.csv"), index = None)

		else:
			print("\"{}\" is not csv file".format(file))

	vectorPointE = np.asarray(entireVP)
	centroidsVectorE, labelsArrayE, inertiaValueE = centroids_finder(vectorPointE,10)
	seriesE = pd.Series(labelsArrayE)
	entireDF.insert(2, "CentroidLabelIndex_entireUnits", seriesE)
	print(entireDF)

if __name__ == '__main__':
	main()














































