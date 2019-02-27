import pandas as pd
import numpy as np
import argparse
import csv
import os
import math
import ast
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
	filename = opt +'_pitchSet_' + title + '_by_' + composer + '.csv'
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




	list_set = set(pitch_list)
	oct_list_set = set(oct_list)
	list_set = sorted(list_set)
	oct_list_set = sorted(oct_list_set)
	data_row = [(offsetRange,list_set,oct_list_set)]

	df_new = pd.DataFrame(data_row, columns=['offsetRange','pitchSet','oct_list_set'])

	# returning df which has 2 col('offsetRange', 'pitchSet') of CHUNK splitted by user custom offset interval 
	return df_new
def octScalePitch(c):
	s = c['Chord']
	l = s[21:-1].split()
	# print(l)
	return l

def main():
	
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
	cwd = os.getcwd()
	directory = os.path.join(cwd,dir_name)

	offset_term = int(input("Enter offset value to split: "))

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
			chunk_count = math.ceil(last_offset/offset_term)
			last_offset = chunk_count * offset_term

			chunk_list = list(range(0,last_offset,offset_term))
			df_pitchSetOnly = pd.DataFrame()
			# k = None
			df_att2 = pd.DataFrame(columns=['offsetRange','pitchSet','oct_list_set'])
			for x in chunk_list:
				k = df[(df['offset']>=x)&(df['offset']<x+offset_term)]
				df_row = pitchSet(k)
				df_att = pd.DataFrame(columns=['offsetRange','pitchSet','oct_list_set'])
				for z in range(len(k)):
					# print(z)
					df_att = df_att.append(df_row, ignore_index=True)
				c1 = df_row['offsetRange']
				c2 = df_row['pitchSet']
				c3 = df_row['oct_list_set']
				df_pitchSetOnly = df_pitchSetOnly.append(df_row, ignore_index=True)
				df_att2 = df_att2.append(df_att,ignore_index=True)

			df['offsetRange'] = df_att2['offsetRange']
			df['pitchSet'] = df_att2['pitchSet']
			df['oct_list_set'] = df_att2['oct_list_set']
			print(df)
			optionalString = ""

			csv_gen(df_pitchSetOnly, directory, piece_name, composer_name,"pitchSetOnly")
			csv_gen(df, directory, piece_name, composer_name,optionalString)





		# ignoring some other log files
		else:
			print("\"{}\" is not csv file".format(file))

if __name__ == '__main__':
	main()








