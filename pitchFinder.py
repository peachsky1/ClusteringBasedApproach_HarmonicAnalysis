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


def csv_gen(df_X, directoryName, pieceName, composerName):
	directory = directoryName
	cwd = os.getcwd()
	title = str(pieceName)
	composer = str(composerName)
	filename = 'pitchSet_' + title + '_by_' + composer + '.csv'
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
	# print(offsetRange)
	# df_x["offsetRange"] = offsetRange
	# print(df_x)
	
	str_list=[]
	for index, row in df_x.iterrows():
		k = row['PCsInNormalForm']
		# converting str to list
		k = ast.literal_eval(k)
		str_list +=k
	list_set = set(str_list)
	# print(list_set)
	# print(offsetRange)
	# print(list_set)
	data_row = [(offsetRange,list_set)]
	df_new = pd.DataFrame(data_row, columns=['offsetRange','pitchSet'])

	# returning df which has 2 col('offsetRange', 'pitchSet') of CHUNK splitted by user custom offset interval 
	return df_new

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
			piece_name = df['file'].iloc[0]
			composer_name = df['Composer'].iloc[0]

			last_offset = df['offset'].max()
			chunk_count = math.ceil(last_offset/offset_term)
			last_offset = chunk_count * offset_term

			chunk_list = list(range(0,last_offset,offset_term))
			df_final = pd.DataFrame()
			# k = None
			for x in chunk_list:
				# print(x)
				k = df[(df['offset']>=x)&(df['offset']<x+offset_term)]
				df_row = pitchSet(k)
				df_final = df_final.append(df_row, ignore_index=True)
			print(df_final)


			csv_gen(df_final, directory, piece_name, composer_name)


			# print(df_final)

			# print(df['offset']==df['offset'].between(4,4+offset_term))
			# df = df[(df['offset'] >= 4) & (df['offset'] < 8)]
			# print(df)
			# print(k)
			# print(df['offset'].between(0,4))

			# This will return the entire row with max value
			# df[df['Value']==df['Value'].max()]

			# # taking col name as "file" which is title
			# df_file = df["file"]
			# # eliminating duplicates and store as list
			# df_unique_list = df_file.unique().tolist()
			# for x in df_unique_list:
			# 	csv_generator(splitter(df,x),directory)

		# ignoring some other log files
		else:
			print("\"{}\" is not csv file".format(file))

if __name__ == '__main__':
	main()








