import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import os
import math
import ast
from pandas.plotting import table
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

from ast import literal_eval
import matplotlib
from matplotlib import cm


import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
    

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

#     duplicates has been removed and sorted.
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
    #     next_x = df1[i,:]
    #     print(next_x)
    #     coef_list = []
    #     for e in range(len(df2)):
    #         next_cor = np.corrcoef(next_x, e)[0][1]
    #         coef_list.append(next_cor)
    #     label_index = np.argmax(np.array(coef_list))
    #     max_corr = coef_list[label_index]
    #     label = df2['label'][label_index]
    #     # print(coef_list)
    #     # print(label_index)
    #     # print(max_corr)
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
    distances, iVal, varianceVal, retVal , retCount = inertiaFinder(arr)
    # print(K)
    kmeans = KMeans(n_clusters=K, random_state=1).fit(arr)
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    # print(inertia)
    centroids = kmeans.cluster_centers_
    return centroids , labels , inertia, distances, iVal, varianceVal , retVal , retCount

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
        km = KMeans(n_clusters=i, random_state=1).fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1, 30), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    # plt.savefig('./figures/elbow.png', dpi=300)
    plt.show()

def inertiaFinder(X):
    # X = X[:,[0,1]]
    km = KMeans(n_clusters=20, random_state=1)
    distances= km.fit_transform(X)
    print(distances)
    variance = 0
    i=0
    retVal = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    retCount = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    
    
    
    # i = total counts of vector points
    # label = centroid index
    
    for label in km.labels_:
        print(label)
  
        variance = variance + distances[i][label]*distances[i][label]
        retVal[label] += distances[i][label]*distances[i][label] 
        retCount[label] += 1
        i = i + 1
    return distances, i, variance , retVal , retCount
    
    



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





def compute_pos(xticks, width, i, models):
    index = np.arange(len(xticks))
    n = len(models)
    correction = i-0.5*(n-1)
    return index + width*correction

def present_height(ax, bar):
    for rect in bar:
        height = rect.get_height()
        posx = rect.get_x()+rect.get_width()*0.5
        posy = height*1.01
        ax.text(posx, posy, '%.3f' % height, rotation=90, ha='center', va='bottom')
		
                

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
    
   # Run above in cli. Run below at spyder
    
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
    centroidsVectorE, labelsArrayE, inertiaValueE, distancesE, iValE, varianceValE, retVal, retCount = centroids_finder(vectorPointE,20)
    for x in range(0,20):
        retVal[x] = retVal[x] / retCount[x]
        print(x)
        
    
    myorder = [11, 19, 8, 15, 2, 17, 10, 9, 4, 3, 16, 18, 13, 6, 12, 0, 1, 5, 7, 14]
    #retval = summation of squared distance of each centroid / count of element in the cluster 
    retVal = [retVal[i] for i in myorder]
    retCount =  [retCount[i] for i in myorder]
    type(retVal)
    plt.plot(retVal)
    plt.ylabel('Sum of squared Euclidean distance / count of points\n(of Each cluster)')
    plt.xlabel('Cluster index')
    plt.xticks(np.arange(0,20,1))
    plt.yticks(np.arange(1,2.5,.1))
    figDir = os.path.join(cwd,"smallDF")
    firDir = os.path.join(figDir,'distortion_plot')
    plt.savefig(firDir+'.png',dpi=500)  
    retValSeries = pd.Series(retVal, name = 'Tightness')
    retCountSeries = pd.Series(retCount, name ='VectorCount')
    outFile = pd.concat([retValSeries, retCountSeries],axis = 1)
    cwd = os.getcwd()
    fn = 'DistortionData.csv'
    fn = os.path.join(cwd,fn)
    outFile.to_csv(fn)
    
    

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



    entireDF

    
#    
#    #Stage split
#    stageOne
#    stageTwo
#    stageThree
#    stageFour
#    stageFive
#
#    
    

    ######## ######## ######## ######## ######## ######## ######## ######## ######## ########
     ######## ######## ######## ######## ######## ######## ######## ######## ######## ########
      ######## ######## ######## ######## From there to bottom, adding year, nad oldCentroid    chordName    pc-labels    color    newCentroidIndex ######## ######## ########
       ######## ######## ######## ######## ######## ######## ######## ######## ######## ########
    #ADD YEAR of composition HERE!!
    #adding column
    entireDF['year'] = 0 
    entireDF.head(10)
    fileList = entireDF['filename'].unique().tolist()
    sorted(fileList)
    
    

    
    
    
    
    
    entireDF.loc[entireDF['filename']=='Piano Sonata K279 i C major.mid by Mozart.csv','year']=1774
    entireDF.loc[entireDF['filename']=='Piano Sonata K279 ii F major.mid by Mozart.csv','year']=1774
    entireDF.loc[entireDF['filename']=='Piano Sonata K279 iii C major.mid by Mozart.csv','year']=1774
    entireDF.loc[entireDF['filename']=='Piano Sonata K280 i F major.mid by Mozart.csv','year']=1774
    entireDF.loc[entireDF['filename']=='Piano Sonata K282 i Eb major.mid by Mozart.csv','year']=1775
    entireDF.loc[entireDF['filename']=='Piano Sonata K283 i G major.mid by Mozart.csv','year']=1775
    entireDF.loc[entireDF['filename']=='Piano Sonata K284 i D major.mid by Mozart.csv','year']=1775
    entireDF.loc[entireDF['filename']=='Piano Sonata K309 i C major.mid by Mozart.csv','year']=1777
    entireDF.loc[entireDF['filename']=='Piano Sonata K311 i D major.mid by Mozart.csv','year']=1777
    entireDF.loc[entireDF['filename']=='Piano Sonata K330 i C major.mid by Mozart.csv','year']=1783
    entireDF.loc[entireDF['filename']=='Piano Sonata K330 iii C major.mid by Mozart.csv','year']=1783
    entireDF.loc[entireDF['filename']=='Piano Sonata K332 i F major.mid by Mozart.csv','year']=1783
    entireDF.loc[entireDF['filename']=='Piano Sonata K332 ii Bb major.mid by Mozart.csv','year']=1783
    entireDF.loc[entireDF['filename']=='Piano Sonata K333 i Bb major.mid by Mozart.csv','year']=1783
    entireDF.loc[entireDF['filename']=='Piano Sonata K545 i C major.mid by Mozart.csv','year']=1788
    entireDF.loc[entireDF['filename']=='Piano Sonata K570 i Bb major.mid by Mozart.csv','year']=1789
    entireDF.loc[entireDF['filename']=='Piano Sonata K576 ii A major.mid by Mozart.csv','year']=1789
    
    yearList = entireDF['year'].unique().tolist()
    sorted(yearList)
    entireDF.head(10)
    yearCount = entireDF.sort_values(['year'],ascending=False).groupby('year').count()[['index']]
    yearCount['index'] = yearCount
    yearCount = yearCount.rename({'index': 'rowCount'}, axis='columns')
    yearCount['percentage'] = yearCount['rowCount']/yearCount['rowCount'].sum()*100
      
    

    oldCentroid = [11, 19, 8, 15, 2, 17, 10, 9, 4, 3, 16, 18, 13, 6, 12, 0, 1, 5, 7, 14]
    
    chordName=['iv_ofDmin',
                 '4ofC',
                 'Cmaj-root',
                 'Cmaj1-4tet',
                 '2/4ofCmaj',
                 'CmajTriad',
                 'Cmaj2-5tet',
                 '6-1ofC',
                 'Cmaj3rd5th',
                 'Gmaj-root',
                 'chr',
                 'V-IofC',
                 'Cmaj5-1tet',
                 'GmajTriad',
                 'AminTriad',
                 'VofG',
                 'Gmaj-IV/vii',
                 'Gmaj3rd5th',
                 'V7ofG',
                 'VofAmin']
     
    pc_labels = ['GBb-C#D',
                 'F-AC',
                 'C-EG',
                 'C-DEF',
                 'DF',
                 'CEG',
                 'DEFG',
                 'A-CF',
                 'EG-C',
                 'G-BD',
                 'FGAB',
                 'D-GBC',
                 'BC-AG',
                 'GBD',
                 'ACE',
                 'D-F#A',
                 'F#G-ACE',
                 'BD-G',
                 'DF#AC',
                 'E-BCD']

    
    color = ['yellow',
                 'cyan',
                 'slateblue',
                 'blue',
                 'green',
                 'blueviolet',
                 'violet',
                 'deepskyblue',
                 'mediumvioletred',
                 'red',
                 'mediumpurple',
                 'crimson',
                 'darkviolet',
                 'orange',
                 'steelblue',
                 'greenyellow',
                 'salmon',
                 'gold',
                 'lime',
                 'indigo']
    entireDF['chordName'] = 'null'
    #chordName
    entireDF['pc-labels'] = 'null'
    #pc_labels
    entireDF['color'] = 'null'
    #color
    entireDF['newCentroidIndex'] = 'null'
    entireDF.head(10)
    #oldCentroid
    
    
    
    #i value is newCentroidIndex
    for i in range(0,20):
        entireDF.loc[entireDF['CentroidLabelIndex_entireUnits']==oldCentroid[i],'chordName'] = chordName[i]
        entireDF.loc[entireDF['CentroidLabelIndex_entireUnits']==oldCentroid[i],'pc-labels'] = pc_labels[i]
        entireDF.loc[entireDF['CentroidLabelIndex_entireUnits']==oldCentroid[i],'color'] = color[i]
        entireDF.loc[entireDF['CentroidLabelIndex_entireUnits']==oldCentroid[i],'newCentroidIndex'] = i
    

    entireDF
    
     
#        
#    
#  '''
#  #HARDCODE
#    newNumbering = pd.DataFrame()
#    
#    newPath = os.path.join(cwd,'NewNumbering.xlsx')
#    xls_file = pd.ExcelFile(newPath)
#    xls_file.sheet_names
#    newNumbering = xls_file.parse('NewNumbering')
#    newNumbering
#    oldCentroid = newNumbering['oldCentroid'].tolist()
#    oldCentroid
#    #[11, 19, 8, 15, 2, 17, 10, 9, 4, 3, 16, 18, 13, 6, 12, 0, 1, 5, 7, 14]
#    chordName = newNumbering['chordName'].tolist()
#    chordName
#    '''
#    ['iv_ofDmin',
# '4ofC',
# 'Cmaj-root',
# 'Cmaj1-4tet',
# '2/4ofCmaj',
# 'CmajTriad',
# 'Cmaj2-5tet',
# '6-1ofC',
# 'Cmaj3rd5th',
# 'Gmaj-root',
# 'chr',
# 'V-IofC',
# 'Cmaj5-1tet',
# 'GmajTriad',
# 'AminTriad',
# 'VofG',
# 'Gmaj-IV/vii',
# 'Gmaj3rd5th',
# 'V7ofG',
# 'VofAmin']
#    '''
#    pc_labels = newNumbering['pc-labels'].tolist()
#    pc_labels
#    '''
#    ['GBb-C#D',
# 'F-AC',
# 'C-EG',
# 'C-DEF',
# 'DF',
# 'CEG',
# 'DEFG',
# 'A-CF',
# 'EG-C',
# 'G-BD',
# 'FGAB',
# 'D-GBC',
# 'BC-AG',
# 'GBD',
# 'ACE',
# 'D-F#A',
# 'F#G-ACE',
# 'BD-G',
# 'DF#AC',
# 'E-BCD']
#    '''
#    color = newNumbering['color'].tolist()
#    color
#    '''
#    ['yello',
# 'cyan',
# 'slateblue',
# 'blue',
# 'green',
# 'blueviolet',
# 'violet',
# 'deepskyblue',
# 'mediumvioletred',
# 'red',
# 'mediumpurple',
# 'crimson',
# 'darkviolet',
# 'orange',
# 'steelblue',
# 'greenyellow',
# 'salmon',
# 'gold',
# 'lime',
# 'indigo']
#    '''
#    
#    
#    
#    newCentroidIndex = newNumbering['newCentroidIndex'].tolist()
#    newCentroidIndex
#    #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#    
#    
#    
#    
#    #hard code just in case.
#    
#    '''
#    
        
    ######## ######## ######## ######## ######## ######## ######## ######## ######## ########
     ######## ######## ######## ######## ######## ######## ######## ######## ######## ########
      ######## ######## ######## ######## From there to bottom, adding year, nad oldCentroid    chordName    pc-labels    color    newCentroidIndex ######## ######## ########
       ######## ######## ######## ######## ######## ######## ######## ######## ######## ########
   
    
        

    
    
    entireDF = entireDF.drop(['CentroidLabelIndex_individualUnit'],axis=1)
    entireDF.rename(columns={'CentroidLabelIndex_entireUnits':'oldCentroidIndex'}, inplace=True)
    
    # entire bar chart
    
    countDF = entireDF.newCentroidIndex.value_counts(normalize=True)
    countDF
    countDF = countDF.sort_index(axis=0, level=None, ascending=True, inplace=False, sort_remaining=True)
    #value as y axis. normalized
    countDF.plot.bar(y=[0], alpha=0.5)
    countDF

    entireDF
  
        
    entireDF['PS'] = 0
    entireDF.head()
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 i C major.mid by Mozart.csv') & (entireDF['index'] < 93),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 i C major.mid by Mozart.csv') & (entireDF['index'] >= 93) & (entireDF['index'] < 212),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 i C major.mid by Mozart.csv') & (entireDF['index'] >= 212) & (entireDF['index'] < 316),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 i C major.mid by Mozart.csv') & (entireDF['index'] >= 316) & (entireDF['index'] < 381),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 i C major.mid by Mozart.csv') & (entireDF['index'] >= 381) ,'PS']=5
    ##
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 ii F major.mid by Mozart.csv') & (entireDF['index'] < 45),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 ii F major.mid by Mozart.csv') & (entireDF['index'] >= 45) & (entireDF['index'] < 120),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 ii F major.mid by Mozart.csv') & (entireDF['index'] >= 120) & (entireDF['index'] < 173),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 ii F major.mid by Mozart.csv') & (entireDF['index'] >= 173) & (entireDF['index'] < 210),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 ii F major.mid by Mozart.csv') & (entireDF['index'] >= 210) ,'PS']=5
    ##
    
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 iii C major.mid by Mozart.csv') & (entireDF['index'] < 62),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 iii C major.mid by Mozart.csv') & (entireDF['index'] >= 62) & (entireDF['index'] < 151),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 iii C major.mid by Mozart.csv') & (entireDF['index'] >= 151) & (entireDF['index'] < 235),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 iii C major.mid by Mozart.csv') & (entireDF['index'] >= 235) & (entireDF['index'] < 294),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K279 iii C major.mid by Mozart.csv') & (entireDF['index'] >= 294) ,'PS']=5
    ##
    
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K280 i F major.mid by Mozart.csv') & (entireDF['index'] < 111),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K280 i F major.mid by Mozart.csv') & (entireDF['index'] >= 111) & (entireDF['index'] < 235),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K280 i F major.mid by Mozart.csv') & (entireDF['index'] >= 235) & (entireDF['index'] < 342),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K280 i F major.mid by Mozart.csv') & (entireDF['index'] >= 342) & (entireDF['index'] < 449),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K280 i F major.mid by Mozart.csv') & (entireDF['index'] >= 449) ,'PS']=5
    ##
    
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K282 i Eb major.mid by Mozart.csv') & (entireDF['index'] < 69),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K282 i Eb major.mid by Mozart.csv') & (entireDF['index'] >= 69) & (entireDF['index'] < 123),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K282 i Eb major.mid by Mozart.csv') & (entireDF['index'] >= 123) & (entireDF['index'] < 169),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K282 i Eb major.mid by Mozart.csv') & (entireDF['index'] >= 169) & (entireDF['index'] < 199),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K282 i Eb major.mid by Mozart.csv') & (entireDF['index'] >= 199) ,'PS']=5
    ##
    

    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K283 i G major.mid by Mozart.csv') & (entireDF['index'] < 69),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K283 i G major.mid by Mozart.csv') & (entireDF['index'] >= 69) & (entireDF['index'] < 160),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K283 i G major.mid by Mozart.csv') & (entireDF['index'] >= 160) & (entireDF['index'] < 214),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K283 i G major.mid by Mozart.csv') & (entireDF['index'] >= 214) & (entireDF['index'] < 267),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K283 i G major.mid by Mozart.csv') & (entireDF['index'] >= 267) ,'PS']=5
    ##
    
    
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K284 i D major.mid by Mozart.csv') & (entireDF['index'] < 175),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K284 i D major.mid by Mozart.csv') & (entireDF['index'] >= 175) & (entireDF['index'] < 414),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K284 i D major.mid by Mozart.csv') & (entireDF['index'] >= 414) & (entireDF['index'] < 573),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K284 i D major.mid by Mozart.csv') & (entireDF['index'] >= 573) & (entireDF['index'] < 739),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K284 i D major.mid by Mozart.csv') & (entireDF['index'] >= 739) ,'PS']=5
    ##
    
    
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K309 i C major.mid by Mozart.csv') & (entireDF['index'] < 250),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K309 i C major.mid by Mozart.csv') & (entireDF['index'] >= 250) & (entireDF['index'] < 447),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K309 i C major.mid by Mozart.csv') & (entireDF['index'] >= 447) & (entireDF['index'] < 712),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K309 i C major.mid by Mozart.csv') & (entireDF['index'] >= 712) & (entireDF['index'] < 962),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K309 i C major.mid by Mozart.csv') & (entireDF['index'] >= 962) ,'PS']=5
    ##
    
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K311 i D major.mid by Mozart.csv') & (entireDF['index'] < 89),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K311 i D major.mid by Mozart.csv') & (entireDF['index'] >= 89) & (entireDF['index'] < 209),'PS']=2
    #entireDF.loc[(entireDF['filename']=='Piano Sonata K311 i D major.mid by Mozart.csv') & (entireDF['index'] >= 209) & (entireDF['index'] < 316),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K311 i D major.mid by Mozart.csv') & (entireDF['index'] >= 209) & (entireDF['index'] < 411),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K311 i D major.mid by Mozart.csv') & (entireDF['index'] >= 411) ,'PS']=5
    ##
    
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K330 i C major.mid by Mozart.csv') & (entireDF['index'] < 125),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K330 i C major.mid by Mozart.csv') & (entireDF['index'] >= 125) & (entireDF['index'] < 283),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K330 i C major.mid by Mozart.csv') & (entireDF['index'] >= 283) & (entireDF['index'] < 422),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K330 i C major.mid by Mozart.csv') & (entireDF['index'] >= 422) & (entireDF['index'] < 541),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K330 i C major.mid by Mozart.csv') & (entireDF['index'] >= 541) ,'PS']=5
    ##
    
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K330 iii C major.mid by Mozart.csv') & (entireDF['index'] < 136),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K330 iii C major.mid by Mozart.csv') & (entireDF['index'] >= 136) & (entireDF['index'] < 284),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K330 iii C major.mid by Mozart.csv') & (entireDF['index'] >= 284) & (entireDF['index'] < 395),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K330 iii C major.mid by Mozart.csv') & (entireDF['index'] >= 395) & (entireDF['index'] < 559),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K330 iii C major.mid by Mozart.csv') & (entireDF['index'] >= 559) ,'PS']=5
    ##
    
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K332 i F major.mid by Mozart.csv') & (entireDF['index'] < 119),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K332 i F major.mid by Mozart.csv') & (entireDF['index'] >= 119) & (entireDF['index'] < 273),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K332 i F major.mid by Mozart.csv') & (entireDF['index'] >= 273) & (entireDF['index'] < 386),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K332 i F major.mid by Mozart.csv') & (entireDF['index'] >= 386) & (entireDF['index'] < 514),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K332 i F major.mid by Mozart.csv') & (entireDF['index'] >= 514) ,'PS']=5
    ##
    
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K333 i Bb major.mid by Mozart.csv') & (entireDF['index'] < 92),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K333 i Bb major.mid by Mozart.csv') & (entireDF['index'] >= 92) & (entireDF['index'] < 255),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K333 i Bb major.mid by Mozart.csv') & (entireDF['index'] >= 255) & (entireDF['index'] < 415),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K333 i Bb major.mid by Mozart.csv') & (entireDF['index'] >= 415) & (entireDF['index'] < 506),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K333 i Bb major.mid by Mozart.csv') & (entireDF['index'] >= 506) ,'PS']=5
    ##
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K545 i C major.mid by Mozart.csv') & (entireDF['index'] < 48),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K545 i C major.mid by Mozart.csv') & (entireDF['index'] >= 48) & (entireDF['index'] < 106),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K545 i C major.mid by Mozart.csv') & (entireDF['index'] >= 106) & (entireDF['index'] < 154),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K545 i C major.mid by Mozart.csv') & (entireDF['index'] >= 154) & (entireDF['index'] < 212),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K545 i C major.mid by Mozart.csv') & (entireDF['index'] >= 212) ,'PS']=5
    ##
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K570 i Bb major.mid by Mozart.csv') & (entireDF['index'] < 118),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K570 i Bb major.mid by Mozart.csv') & (entireDF['index'] >= 118) & (entireDF['index'] < 230),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K570 i Bb major.mid by Mozart.csv') & (entireDF['index'] >= 230) & (entireDF['index'] < 382),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K570 i Bb major.mid by Mozart.csv') & (entireDF['index'] >= 382) & (entireDF['index'] < 491),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K570 i Bb major.mid by Mozart.csv') & (entireDF['index'] >= 491) ,'PS']=5
    
    entireDF.loc[(entireDF['filename']=='Piano Sonata K576 ii A major.mid by Mozart.csv') & (entireDF['index'] < 35),'PS']=1
    entireDF.loc[(entireDF['filename']=='Piano Sonata K576 ii A major.mid by Mozart.csv') & (entireDF['index'] >= 35) & (entireDF['index'] < 74),'PS']=2
    entireDF.loc[(entireDF['filename']=='Piano Sonata K576 ii A major.mid by Mozart.csv') & (entireDF['index'] >= 74) & (entireDF['index'] < 124),'PS']=3
    entireDF.loc[(entireDF['filename']=='Piano Sonata K576 ii A major.mid by Mozart.csv') & (entireDF['index'] >= 124) & (entireDF['index'] < 152),'PS']=4
    entireDF.loc[(entireDF['filename']=='Piano Sonata K576 ii A major.mid by Mozart.csv') & (entireDF['index'] >= 152) ,'PS']=5
    
    
    
    
    
    
    
    cwd = os.getcwd()
    fn = 'testresult.csv'
    fn = os.path.join(cwd,fn)
    fn
    entireDF.to_csv(fn)
    
    entireDF.head()
    anovaDF = entireDF[['newCentroidIndex','PS','filename']].copy()
    anovaDF.head()
    
    
    
    
    
    #stateCount => stCount.
    stateGroup = anovaDF.groupby('PS').count()
    stateCount = stateGroup.iloc[:,0:1].copy().values
    stCount = []
    type(stateCount)
    for l in stateCount:
        print(l[0])
        stCount.append(l[0])
    print(stCount)
    st1 = []
    st2 = []
    st3 = []
    st4 = []
    st5 = []
    
    
    clusterGroup = anovaDF.groupby(['newCentroidIndex','PS']).count()
    fn = 'clusterGroup.csv'
    fn = os.path.join(cwd,fn)
    clusterGroup.to_csv(fn)
    clusterCounts = clusterGroup.iloc[:,0:1].copy().values
    clusterCount = []
    for l in clusterCounts:
        clusterCount.append(l[0])
        #filling up the missing data. 
        if l[0]==156:
            clusterCount.append(0)            
    print(len(clusterCount))
    
    st0 = clusterCount[0::6]
    st1 = clusterCount[1::6]
    st2 = clusterCount[2::6]
    st3 = clusterCount[3::6]
    st4 = clusterCount[4::6]
    st5 = clusterCount[5::6]
    

    st1 = [round(x / stCount[1] * 100, 2) for x in st1] 
    st2 = [round(x / stCount[2] * 100, 2) for x in st2]
    st3 = [round(x / stCount[3] * 100, 2) for x in st3]
    st4 = [round(x / stCount[4] * 100, 2) for x in st4]
    st5 = [round(x / stCount[5] * 100, 2) for x in st5]
    

#======================== stage by cluster prob analysis begins================================
    newDF=pd.DataFrame({ 'x':range(1,6)})

    
    dfdf['prob(%)'] = dfdf['countInCluster']/dfdf['TotalCount'] * 100
    dfdf = dfdf.reset_index()
    xx = np.arange(len(dfdf))
    yy= dfdf['year'].tolist()
    width = 0.35
    tc = dfdf['TotalCount'].tolist()
    cic = dfdf['countInCluster'].tolist()
    p = dfdf['prob(%)'].tolist()
    p = list(np.around(np.array(p),3))
    fig, ax = plt.subplots()
    rects1 = ax.bar(xx, p, width,label = 'countInCluster/TotalCount(%)')
    
    ax.set_ylabel('Probabilities%')
    ax.set_title('Percentage of data points in the given cluster over all pieces')
    ax.set_xticks(xx)
    ax.set_xticklabels(yy)
    ax.legend()
    autolabel(rects1)
    
    fig.tight_layout()
    #plt.show()
    
    plt.savefig(firDir+str(v)+'_norm_by_year.png',dpi=500)     
    print("Current working cluster index is: "+str(v))
    
    cwd = os.getcwd()
    fn = 'Index_'+str(v) + '_Cluster by year.csv'
    fn = os.path.join(cwd,fn)
    dfdf.to_csv(fn)
    
    for x in range(0,20):
        probList = []
        probList.append(st1[x])
        probList.append(st2[x])
        probList.append(st3[x])
        probList.append(st4[x])
        probList.append(st5[x])
        print(probList)
        
        ddddd=pd.DataFrame({ x: probList })
        
        newDF = pd.concat([newDF, ddddd],1)
        newDF.set_index('x')
        
        plt.cla()
        plt.plot( 'x', 'y', data=ddddd, linestyle='-', marker='o', color = color[x], label=str(x))
        plt.ylabel('Prob Of Cluster$(\%)$')
        plt.xlabel('Stage')
        xint = range(1, 6)
        plt.xticks(xint)
        #plt.show()
        
        figDir = os.path.join(cwd,"smallDF")
        firDir = os.path.join(figDir,'Cluster_'+str(x))
        #plt.savefig(firDir+'.png',dpi=500)   
        #plt.cla()
    #plt.show()
    figDir = os.path.join(cwd,"smallDF")
    firDir = os.path.join(figDir,'All_cluster_'+str(x))
    plt.savefig(firDir+'.png',dpi=500)  
    
#Run until right here!!! different results based on the entireDF below
    
    
        
    
    
    
    
    
    def anova_table(aov):
        aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
        
        aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
        
        aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
        
        cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
        aov = aov[cols]
        return aov


    
    
#one-way ANOVA
    import io
    import csv    
    output = io.StringIO()
    #v= cluster index
    for v,smallDF in anovaDF.groupby('newCentroidIndex'):
        cwd = os.getcwd()
        fn = str(v)+'_anovaTable_result.csv'
        fn = os.path.join(cwd,fn)
        k = rp.summary_cont(smallDF['PS'].groupby(smallDF['filename']))
        results = ols('PS ~ C(filename)', data=smallDF).fit()
        ret = results.summary().as_csv()
        print(ret)
        type(ret)
        aov_table = sm.stats.anova_lm(results, typ=2)
        type(aov_table)
        #aov_table.to_csv(fn)

        smallDF.boxplot('PS', by= 'filename')
        plt.subplots_adjust(bottom=0.25)
        plt.xticks(rotation=90)
        #plt.show()
        figDir = os.path.join(cwd,"smallDF")
        firDir = os.path.join(figDir,'stat_cluster_'+str(v))
        
        plt.savefig(firDir+'.png',dpi=500,bbox_inches="tight")
        anova_table(aov_table).to_csv(fn)
                
        
       
    
 

   
    
# year / cluster         = This is not good
    for v,smallDF in entireDF.groupby('year'):
        v = str(v)
        #print(smallDF['filename'].iloc[0])
        #print(v)
        print(smallDF)
        cwd = os.getcwd()
        filename = 'year_'+v+'.csv'
        new_path = os.path.join(cwd,"smallDF")
        if not os.path.exists(new_path):
            os.mkdir(new_path)
            print("=======================================")
            print("directory \"{}\" has been created".format(new_path))
            print("=======================================")
        final_dir= os.path.join(new_path,filename)
        ####!!!!Save smallDF as needeed!
        smallDF.to_csv(final_dir, index = None)
        print("filename: \"{}\" has been successfully created :^D".format(filename))

        figDir = os.path.join(cwd,"smallDF")
        firDir = os.path.join(figDir,'Year '+v)
        #smallDF.plot(linewidth=0.5)
        smallNorm = smallDF.newCentroidIndex.value_counts(normalize=True)
        smallNorm = smallNorm.sort_index(axis=0, level=None, ascending=True, inplace=False, sort_remaining=True)
        #smallNorm.plot.bar(y=[0], alpha=0.5, title=v)
        smallNorm
        countDF
        plt.cla()
        fig = plt.figure()
        df = pd.DataFrame({'Total': countDF, 'Year '+v: smallNorm} )
        
        

        
        fig = df.plot.bar(rot=45)
        
        plt.savefig(firDir+' by cluster.png',dpi=500)     
        print("Current working piece is: "+v)
        print(smallDF.head(5))
        
    
# stage / cluster        
    for v,smallDF in entireDF.groupby('PS'):
        v = str(v)
        #print(smallDF['filename'].iloc[0])
        #print(v)
        print(smallDF)
        cwd = os.getcwd()
        filename = 'stage_'+v+'.csv'
        new_path = os.path.join(cwd,"smallDF")
        if not os.path.exists(new_path):
            os.mkdir(new_path)
            print("=======================================")
            print("directory \"{}\" has been created".format(new_path))
            print("=======================================")
        final_dir= os.path.join(new_path,filename)
        ####!!!!Save smallDF as needeed!
        smallDF.to_csv(final_dir, index = None)
        print("filename: \"{}\" has been successfully created :^D".format(filename))

        figDir = os.path.join(cwd,"smallDF")
        firDir = os.path.join(figDir,'Stage '+v)
        #smallDF.plot(linewidth=0.5)
        smallNorm = smallDF.newCentroidIndex.value_counts(normalize=True)
        smallNorm = smallNorm.sort_index(axis=0, level=None, ascending=True, inplace=False, sort_remaining=True)
        #smallNorm.plot.bar(y=[0], alpha=0.5, title=v)
        smallNorm
        countDF
        plt.cla()
        fig = plt.figure()
        df = pd.DataFrame({'Total': countDF, 'Stage '+v: smallNorm} )
        
        

        
        fig = df.plot.bar(rot=45)
        
        plt.savefig(firDir+' by cluster.png',dpi=500)     
        print("Current working piece is: "+v)
        print(smallDF.head(5))
        
        
#    dfOut=os.path.join(cwd,"entireDF.csv")
#    entireDF.to_csv(dfOut,index=None)
    
    
    
        
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
        ####!!!!Save smallDF as needeed!
        smallDF.to_csv(final_dir, index = None)
        print("filename: \"{}\" has been successfully created :^D".format(filename))

        figDir = os.path.join(cwd,"smallDF")
        firDir = os.path.join(figDir,v)
        #smallDF.plot(linewidth=0.5)
        smallNorm = smallDF.newCentroidIndex.value_counts(normalize=True)
        smallNorm = smallNorm.sort_index(axis=0, level=None, ascending=True, inplace=False, sort_remaining=True)
        #smallNorm.plot.bar(y=[0], alpha=0.5, title=v)
        smallNorm
        countDF
        plt.cla()
        fig = plt.figure()
        df = pd.DataFrame({'Total': countDF, v: smallNorm} )
        
        

        
        fig = df.plot.bar(rot=45)
        
        plt.savefig(firDir+'.norm-bar.png',dpi=500)     
        print("Current working piece is: "+v)
        print(smallDF.head(5))
        
        
    dfOut=os.path.join(cwd,"entireDF.csv")
    entireDF.to_csv(dfOut,index=None)
    
#    ====================================== Year analysis begins=======================
    
    yearCount = entireDF.groupby('year').count()[['index']]
    yearCount.reset_index().set_index('year')
    yearCount.rename(columns = {'index':'TotalCount'},inplace = True)
    
    yearCount.values[2][0]

    # x-axis: Year(in order), y-axis: norm dist, 20 charts
    # normal distributed 20 charts vs year per Chart
    # v = cluster index, smallDF = corresponding dataframe
        
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
    for v, smallDF in entireDF.groupby('newCentroidIndex'):
        print(smallDF)
        print(v)
        
        #print(smallDF['filename'].iloc[0])
        #print(v)
#        print(smallDF)
        cwd = os.getcwd()
#        filename = 'custerIndex_'+ str(v)+'.csv'
        new_path = os.path.join(cwd,"smallDF")
        if not os.path.exists(new_path):
            os.mkdir(new_path)
            print("=======================================")
            print("directory \"{}\" has been created".format(new_path))
            print("=======================================")
#        final_dir= os.path.join(new_path,filename)
        ####!!!!Save smallDF as needeed!
#        Do not save this dataframe. only for plot
#        smallDF.to_csv(final_dir, index = None)
#        print("filename: \"{}\" has been successfully created :^D".format(filename))
        
    
        figDir = os.path.join(cwd,"smallDF")
        firDir = os.path.join(figDir,'ClusterIndex_')
        yearCountCluster = smallDF.groupby('year').count()[['index']]
        yearCountCluster.rename(columns = {'index':'countInCluster'}, inplace = True)
        dfdf = pd.concat([yearCount, yearCountCluster],1)
        dfdf['prob(%)'] = dfdf['countInCluster']/dfdf['TotalCount'] * 100
        dfdf = dfdf.reset_index()
        xx = np.arange(len(dfdf))
        yy= dfdf['year'].tolist()
        width = 0.35
        tc = dfdf['TotalCount'].tolist()
        cic = dfdf['countInCluster'].tolist()
        p = dfdf['prob(%)'].tolist()
        p = list(np.around(np.array(p),3))
        fig, ax = plt.subplots()
        rects1 = ax.bar(xx, p, width,label = 'countInCluster/TotalCount(%)')
        
        ax.set_ylabel('Probabilities%')
        ax.set_title('Percentage of data points in the given cluster over all pieces')
        ax.set_xticks(xx)
        ax.set_xticklabels(yy)
        ax.legend()
        autolabel(rects1)
        
        fig.tight_layout()
        #plt.show()
        
        plt.savefig(firDir+str(v)+'_norm_by_year.png',dpi=500)     
        print("Current working cluster index is: "+str(v))

        cwd = os.getcwd()
        fn = 'Index_'+str(v) + '_Cluster by year.csv'
        fn = os.path.join(cwd,fn)
        dfdf.to_csv(fn)
    
        
        
        
        #=======Year analysis ends =======================
        
        
        
        
        
        
          
      #Splitting entire DF to small and sequence
      #v = file name
      #smallDF = each context
      #Refactoring - coloring task1
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
        ####!!!!Save smallDF as needeed!
        smallDF.to_csv(final_dir, index = None)
        print("filename: \"{}\" has been successfully created :^D".format(filename))

        figDir = os.path.join(cwd,"smallDF")
        firDir = os.path.join(figDir,v)
        #smallDF.plot(linewidth=0.5)
        smallNorm = smallDF.newCentroidIndex.value_counts(normalize=True)
        smallNorm = smallNorm.sort_index(axis=0, level=None, ascending=True, inplace=False, sort_remaining=True)
        #smallNorm.plot.bar(y=[0], alpha=0.5, title=v)
        #reindex for missing index
        #smallNorm.reindex(list(range(smallNorm.index.min(),smallNorm.index.max()+1)),fill_value=0)
        smallNorm = smallNorm.reindex(list(range(0,20)),fill_value=0)
        
        plt.cla()
        fig = plt.figure()
        x = list(range(0,20,1))
        y = list(smallNorm)

        height = y
        bars = x
        y_pos = x
        plt.bar(y_pos, height, color=color)
        plt.xticks(y_pos, bars)
        

        plt.savefig(firDir+str(v)+'_norm_colormap.png',dpi=500)
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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














































