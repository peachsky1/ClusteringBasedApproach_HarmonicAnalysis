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
from ast import literal_eval
import matplotlib
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
	# directory = directoryName
	cwd = os.getcwd()
	title = str(pieceName)
	composer = str(composerName)
	opt = str(optionalString)
	filename = opt +'_' + title + '_' + composer + '.csv'
	# dir_path = os.path.join(cwd,directory)
	# new_path = os.path.join(dir_path,composer)
	new_path = os.path.join(cwd,composer)
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
def pitchSet(df_x,transpose):

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
	# print(pitch_count_list)
	pitch_count_list = (pitch_count_list[-transpose:] + pitch_count_list[:-transpose])
	# print(pitch_count_list)
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
	distortionFinder(arr)
	# print(K)
	kmeans = KMeans(n_clusters=K, random_state=1).fit(arr)
	labels = kmeans.labels_
	inertia = kmeans.inertia_
	# print(inertia)
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
	return max_coef_label
def strToArr(df):
	# df = df['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']
	arr = []
	df['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]'] = df['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]'].apply(literal_eval)
	for index, row in df.iterrows():
		arr.append(row['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]'])
		# print(type(row['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']))
	return arr
def distortionFinder(X):
	# X = X[:,[0,1]]
	distortions = []
	for i in range(1,30):
	    km = KMeans(n_clusters=i, 
	                init='k-means++', 
	                n_init=10, 
	                max_iter=300, 
	                random_state=0)
	    km.fit(X)
	    distortions.append(km.inertia_)
	plt.plot(range(1, 30), distortions, marker='o')
	plt.xlabel('Number of clusters')
	plt.ylabel('Distortion')
	plt.tight_layout()
	# plt.savefig('./figures/elbow.png', dpi=300)
	plt.show()



def table(df,centroidIndex):
	
	c1 = df[df['CentroidLabelIndex_entireUnits'] == centroidIndex]
	c1G = c1.groupby('triadLabel').size()


	label = c1G.index.values.tolist()
	count = c1G.tolist()
	leng = len(label)
	centr = [centroidIndex]*leng

	data = [centr,label,count]
	rows = zip(data[0], data[1], data[2])
	# headers = ['centroid', 'label', 'count']
	headers = ['Year', 'Month', 'Value']
	df = pd.DataFrame(rows, columns=headers)	
	return df
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
    
	dir_name = "MozartQuarterNote"
	cwd = os.getcwd()
	directory = os.path.join(cwd,dir_name)

	entireDF = pd.DataFrame()
	# print(entireDF)
	for file in os.listdir(directory):
		if file.endswith(".csv"):
			filename = file
			path = os.path.join(directory,filename)
			# defining engine to avoid memory overflow issue
			df= pd.read_csv(path, engine="python")
			entireDF = entireDF.append(df, ignore_index=True)
			# print(df)

		else:
			print("\"{}\" is not csv file".format(file))
	optionalString= ""
	composer_name = "Kmean"
	piece_name = ""
	directory = "all"
	
	entireVP = strToArr(entireDF)

	vectorPointE = np.asarray(entireVP)
    #check optimized number of centroid
	centroidsVectorE, labelsArrayE, inertiaValueE = centroids_finder(vectorPointE,20)
    
    
	print(labelsArrayE)
	a = np.asarray(labelsArrayE)
	#np.savetxt("countList.csv", a, delimiter=",")

	seriesE = pd.Series(labelsArrayE)
	entireDF.insert(2, "CentroidLabelIndex_entireUnits", seriesE)
	# print(entireDF)
    
    #entireDF, directory, piece_name, composer_name, optionalString
	csv_gen(entireDF, directory, piece_name, composer_name, optionalString)
	print(centroidsVectorE)

	# np csv out
	a = np.asarray(centroidsVectorE)
	np.savetxt("CentroidVectorCounts.csv", a, delimiter=",")

	print(entireDF.head(10))

   
    from pandas.plotting import table
    
    
    # entire bar chart
    countDF = entireDF.CentroidLabelIndex_entireUnits.value_counts(normalize=True)
    countDF
    countDF = countDF.sort_index(axis=0, level=None, ascending=True, inplace=False, sort_remaining=True)
    #value as y axis. normalized
    countDF.plot.bar(y=[0], alpha=0.5)
    countDF


    #Stage split
    stageOne
    stageTwo
    stageThree
    stageFour
    stageFive

    entireDF
      #Splitting entire DF to small and sequence
      #v = file name
      #smallDF = each context
    for v,smallDF in entireDF.groupby('filename'):
        #print(smallDF['filename'].iloc[0])
        #print(v)
        print(smallDF)
        cwd = os.getcwd()
        filename = v+'.csv'
        new_path = os.path.join(cwd,"smallDF")
        if not os.path.exists(new_path):
            os.mkdir(new_path)
            print("=======================================")
            print("directory \"{}\" has been created".format(new_path))
            print("=======================================")
        final_dir= os.path.join(new_path,filename)
        #smallDF.to_csv(final_dir, index = None)
        print("filename: \"{}\" has been successfully created :^D".format(filename))

        figDir = os.path.join(cwd,"smallDF")
        firDir = os.path.join(figDir,v)
        #smallDF.plot(linewidth=0.5)
        smallNorm = smallDF.CentroidLabelIndex_entireUnits.value_counts(normalize=True)
        smallNorm = smallNorm.sort_index(axis=0, level=None, ascending=True, inplace=False, sort_remaining=True)
        #smallNorm.plot.bar(y=[0], alpha=0.5, title=v)
        smallNorm
        countDF
        plt.cla()
        fig = plt.figure()
        
        df = pd.DataFrame({'Total': countDF, v: smallNorm} )
        fig = df.plot.bar(rot=0)
       
        plt.savefig(firDir+'.norm-bar.png',dpi=500)     
        print("Current working piece is: "+v)
        print(smallDF.head(5))
        
        
        
        
        
        
        
        
        
        
        
        
  ##  ========resource below=================
    
    
    

    #Splitting entire DF to small and sequence
    for v,smallDF in entireDF.groupby('filename'):
        #print(smallDF['filename'].iloc[0])
        #print(v)
        
        cwd = os.getcwd()
        filename = v+'.csv'
        new_path = os.path.join(cwd,"smallDF")
        if not os.path.exists(new_path):
            os.mkdir(new_path)
            print("=======================================")
            print("directory \"{}\" has been created".format(new_path))
            print("=======================================")
        final_dir= os.path.join(new_path,filename)
        smallDF.to_csv(final_dir, index = None)
        print("filename: \"{}\" has been successfully created :^D".format(filename))

        figDir = os.path.join(cwd,"smallDF")
        firDir = os.path.join(figDir,v)
        smallDF.plot(linewidth=0.5)
        plt.savefig(firDir+'.sequence.png',dpi=500)



    
    
    #creating frequency chart ()
    for v,smallDF in entireDF.groupby('filename'):
        #print(smallDF['filename'].iloc[0])
        #print(v)
        
        cwd = os.getcwd()
        filename = v+'.csv'
        new_path = os.path.join(cwd,"smallDF")
        if not os.path.exists(new_path):
            os.mkdir(new_path)
            print("=======================================")
            print("directory \"{}\" has been created".format(new_path))
            print("=======================================")
        final_dir= os.path.join(new_path,filename)
        #smallDF.to_csv(final_dir, index = None)
        #print("filename: \"{}\" has been successfully created :^D".format(filename))


        countDF = smallDF.reset_index(drop=True)
        dfdf = countDF.CentroidLabelIndex_entireUnits.value_counts()
        
        dfdf = dfdf.sort_index(axis=0, level=None, ascending=True, inplace=False, sort_remaining=True)
        print(dfdf)
        dfdf.plot('bar', linewidth=0.5)
        print("bar plot Generated")

        figDir = os.path.join(cwd,"smallDF")
        firDir = os.path.join(figDir,v)
        #countDF.plot()
        plt.savefig(firDir+'.frequency.png',dpi=500)



    
    #sequence without index. centroid 
    for v,smallDF in entireDF.groupby('filename'):
        #print(smallDF['filename'].iloc[0])
        #print(v)
        
        cwd = os.getcwd()
        filename = v+'.csv'
        new_path = os.path.join(cwd,"smallDF")
        if not os.path.exists(new_path):
            os.mkdir(new_path)
            print("=======================================")
            print("directory \"{}\" has been created".format(new_path))
            print("=======================================")
        final_dir= os.path.join(new_path,filename)
        #smallDF.to_csv(final_dir, index = None)
        #print("filename: \"{}\" has been successfully created :^D".format(filename))
        smallDF = smallDF.reset_index(drop=True)
        
        columns = ['index','CentroidLabelIndex_individualUnit'] 
        smallDF.drop(columns, inplace=True, axis=1)
        #smallDF = smallDF.iloc[:,2:].values
        #smallDF = smallDF.drop(columns=[0])
        #print(smallDF)
        
        
        figDir = os.path.join(cwd,"smallDF")
        firDir = os.path.join(figDir,v)
        smallDF.plot(linewidth=0.5)
        plt.savefig(firDir+'.sequenceWithoutIndex.png',dpi=500)





    #normalized histogram
    for v,smallDF in entireDF.groupby('filename'):
        #print(smallDF['filename'].iloc[0])
        #print(v)
        
        cwd = os.getcwd()
        filename = v+'.csv'
        new_path = os.path.join(cwd,"smallDF")
        if not os.path.exists(new_path):
            os.mkdir(new_path)
            print("=======================================")
            print("directory \"{}\" has been created".format(new_path))
            print("=======================================")
        final_dir= os.path.join(new_path,filename)
        #smallDF.to_csv(final_dir, index = None)
        #print("filename: \"{}\" has been successfully created :^D".format(filename))

        figDir = os.path.join(cwd,"smallDF")
        firDir = os.path.join(figDir,v)
        smallDF.hist(column='CentroidLabelIndex_entireUnits',normed=true)
        plt.xlabel('Centroid Label')
        plt.ylabel('Frequency')
        plt.show()
            
    
            # Remove title
        smallDF.set_title("")

        # Set x-axis label
        smallDF.set_xlabel("CentroidLabel", labelpad=20, weight='bold', size=12)

        # Set y-axis label
        smallDF.set_ylabel("Frequency", labelpad=20, weight='bold', size=12)
        
        #smallDF.plot(linewidth=0.5)
        #plt.savefig(firDir+'.sequence.png',dpi=500)

    
       
        

if __name__ == '__main__':
	main()














































