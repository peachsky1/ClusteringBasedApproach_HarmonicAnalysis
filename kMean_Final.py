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

from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances


import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
    
def transition_matrix(transitions):
    n = 1+ max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M




def transition_prob(transitions):
    n = 1+ max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    tot = 0
    for row in M:
        tot += sum(row)
    
    for row in M:
        if tot > 0:
            row[:] = [f/tot for f in row]
    return M
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

def makeCSV(nparray,fn):
    cwd = os.getcwd()
    fn = os.path.join(cwd,fn)
    pd.DataFrame(nparray).to_csv(fn+".csv")
    
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
    coef_list=[]
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
# =============================================================================
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     #init ax
#     fig, ax = plt.subplots()
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#         
# 
# =============================================================================
        
# =============================================================================
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
# =============================================================================


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


def distortionFinder(X):
    # X = X[:,[0,1]]
    distortions = []
    for i in range(1,40):
        km = KMeans(n_clusters=i, random_state=1).fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1, 40), distortions, marker='o')
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
    
# input df
def centroids_finder(arr, K):
    # print(arr)
    distortionFinder(arr)
    distances, iVal, varianceVal, retVal , retCount = inertiaFinder(arr)
    # print(K)
    kmeans = KMeans(init='k-means++', n_clusters=K, random_state=1).fit(arr)
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    # print(inertia)
    centroids = kmeans.cluster_centers_
    return centroids , labels , inertia, distances, iVal, varianceVal , retVal , retCount

# =============================================================================
# def listToArray(df):
#     df = df['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']
#     coef_list = []
#     for e in range(0,len(df.index)):
# 
#         coef_list.append(np.corrcoef(x,df2['triad'][e])[0][1])
#         # print(e)
#         # print(df2[])
#     # print(coef_list)
#     max_coef = max(coef_list)
#     label_index = coef_list.index(max_coef)
#     max_coef_label = df2['label'][label_index]
#     return max_coef_label
# =============================================================================
def strToArr(df):
    # df = df['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']
    arr = []
    df['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]'] = df['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]'].apply(literal_eval)
    for index, row in df.iterrows():
        arr.append(row['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]'])
        # print(type(row['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']))
    return arr

    



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
    type(entireVP)
    

    vectorPointE = np.asarray(entireVP)
    #check optimized number of centroid
    type(vectorPointE)
    #the distance measure to be used. This must be one of "euclidean", "maximum", "manhattan", "canberra", "binary", "pearson" , "abspearson" , "abscorrelation", "correlation", "spearman" or "kendall". Any unambiguous substring can be given.
    #X_precomputed = pairwise_distances(vectorPointE, metric='manhattan')
    #clustering = SpectralClustering(n_clusters=20, affinity='linear', assign_labels="kmeans", random_state=1)
    #clustering = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels="discretize",random_state=1)
    #clusters_manhattan = clustering.fit(X_precomputed)
    #ak  =clustering.labels_
    #clustering.centroids_ 
    #seriesE_m = pd.Series(ak)
    #entireDF_m = entireDF
    ##entireDF_m.insert(2, "CentroidLabelIndex_entireUnits", seriesE_m)
    #entireDF_m
    #entireDF_m = entireDF_m.drop(['CentroidLabelIndex_individualUnit'],axis=1)
    #fn = 'cluster_manhattan.csv'
    #fn = os.path.join(cwd,fn)
    #entireDF_m.to_csv(fn)
    entireDF = entireDF.drop(['CentroidLabelIndex_individualUnit'],axis=1)
    
    centroidsVectorE, labelsArrayE, inertiaValueE, distancesE, iValE, varianceValE, retVal, retCount = centroids_finder(vectorPointE,20)
    for x in range(0,20):
        retVal[x] = retVal[x] / retCount[x]
        print(x)
    makeCSV(centroidsVectorE,"centroidsVectorE")
    makeCSV(labelsArrayE,"labelsArrayE")
    #makeCSV(inertiaValueE,"inertiaValueE")
    makeCSV(distancesE,"distancesE")
    #makeCSV(iValE,"iValE")
    #makeCSV(varianceValE,"varianceValE")
    makeCSV(retVal,"retVal")
    makeCSV(retCount,"retCount")
    makeCSV(entireDF, "entireDF")
    
    
        
        
    
    #myorder = [11, 19, 8, 15, 2, 17, 10, 9, 4, 3, 16, 18, 13, 6, 12, 0, 1, 5, 7, 14]
    myorder = [12,14,18,6,17,7,19,5,10,1,13,2,16,3,0,8,15,11,4,9]
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
      
    
    
    #oldCentroid = [11, 19, 8, 15, 2, 17, 10, 9, 4, 3, 16, 18, 13, 6, 12, 0, 1, 5, 7, 14]
    oldCentroid = [12,14,18,6,17,7,19,5,10,1,13,2,16,3,0,8,15,11,4,9]
    
    # chordName=['iv_ofDmin',
    #              '4ofC',
    #              'Cmaj-root',
    #              'Cmaj1-4tet',
    #              '2/4ofCmaj',
    #              'CmajTriad',
    #              'Cmaj2-5tet',
    #              '6-1ofC',
    #              'Cmaj3rd5th',
    #              'Gmaj-root',
    #              'chr',
    #              'V-IofC',
    #              'Cmaj5-1tet',
    #              'GmajTriad',
    #              'AminTriad',
    #              'VofG',
    #              'Gmaj-IV/vii',
    #              'Gmaj3rd5th',
    #              'V7ofG',
    #              'VofAmin']
    chordName=['','','','','','','','','','','','','','','','','','','','']
    pc_labels = ['C-EbF#','GBb-C#D','F-D','FAC','FG-BD','CE-G','EFG-A','FA-C','CEG','DEFG-BC','G-F#','CEG','D','GB-D','ABC-DG','BD-G','A-CE','DF#A-C','E-GC','B-DG']
    # pc_labels = ['GBb-C#D',
    #              'F-AC',
    #              'C-EG',
    #              'C-DEF',
    #              'DF',
    #              'CEG',
    #              'DEFG',
    #              'A-CF',
    #              'EG-C',
    #              'G-BD',
    #              'FGAB',
    #              'D-GBC',
    #              'BC-AG',
    #              'GBD',
    #              'ACE',
    #              'D-F#A',
    #              'F#G-ACE',
    #              'BD-G',
    #              'DF#AC',
    #              'E-BCD']

    
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
    makeCSV(entireDF,"entireDF_manhattan")
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
   
    
        

    
    
    #entireDF = entireDF.drop(['CentroidLabelIndex_individualUnit'],axis=1)
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
    
    
    
    
    
    

    #cwd = os.getcwd()
    #fn2 = 'originalDataSet.csv'
    #fn2 = os.path.join(cwd,fn2)
    #original_df = pd.read_csv(fn2,index_col=0)
    #original_df
    #entireDF = original_df
    
    #makeCSV(entireDF, "entireDF_m_ps")
    
    
    entireDF = pd.read_csv("entireDF_m_ps.csv",index_col=0)
    entireDF
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
    #filenameGroup = anovaDF.groupby(['filename']).values
    fn = 'clusterGroup.csv'
    fn = os.path.join(cwd,fn)
    clusterGroup.to_csv(fn)
    clusterCounts = clusterGroup.iloc[:,0:1].copy().values
    len(clusterCounts)
    clusterCount = []
    for l in clusterCounts:
        clusterCount.append(l[0])
        #filling up the missing data. 
        if l[0]==106:
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
    
    st_total = [st1,st2,st3,st4,st5]
    st_arr = np.asarray(st_total)
    makeCSV(st_arr,'probofClusterbyStage')
    
    
    type(st1)
    
    
        
    ''' 
    Euclidean
    HK1: Home key – non-development (frequent in stages 1 and 5, infrequent in 3)
    2, 3, 5, 6, 8,
    
    HK2: Home key – non-ST (frequent in stages 4, 5; infrequent in 2)
    1, 4, 7
    
    SK: Subordinate key areas (frequent in stage 2)
    9, 13, 15, 17, 18
    
    Dev: Developmental (frequent in stage 3)
    0, 10, 14, 16, 19
    
    XD: Expo-Dev. (frequent in stages 1, 2, 3)
    11, 12
    ''' 
            
    ''' 
    Manhattan
    Dev:
    HK: 
    Exposition:
    ST-Dev:
    Recap:
    
    
    
    HK1: Home key – non-development (frequent in stages 1 and 5, infrequent in 3)
    2, 3, 5, 6, 8,
    
    HK2: Home key – non-ST (frequent in stages 4, 5; infrequent in 2)
    1, 4, 7
    
    SK: Subordinate key areas (frequent in stage 2)
    9, 13, 15, 17, 18
    
    Dev: Developmental (frequent in stage 3)
    0, 10, 14, 16, 19
    
    XD: Expo-Dev. (frequent in stages 1, 2, 3)
    11, 12
    ''' 
    
    
    

#======================== stage by cluster prob analysis begins================================
    newDF=pd.DataFrame({'stage':range(1,6)})
    newDF
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
        newDF.set_index('stage')
        
    #newDF = newDF.round(1)
    newDF
    #HK1
    
    
    xx = np.arange(len(newDF))
    yy= newDF['stage'].values.tolist()
    width = 0.15
    fig, ax = plt.subplots()
    
    temp = newDF[2].values.tolist()
    r2 = ax.bar(xx - width*2, temp, width, label='c2')
    temp = newDF[3].tolist()
    r3 = ax.bar(xx - width, temp, width, label='c3')
    temp = newDF[5].tolist()
    r5 = ax.bar(xx , temp, width, label='c5')
    temp = newDF[6].tolist()
    r6 = ax.bar(xx + width, temp, width, label='c6')
    temp = newDF[8].tolist()
    r8 = ax.bar(xx + width*2, temp, width, label='c8')
    
    ax.set_ylabel('prob(%)')
    ax.set_title('stageByClusterProb')
    ax.set_xticks(xx)
    ax.set_xticklabels(yy)
    ax.legend()
    
    for rect in r2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for rect in r3:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for rect in r5:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for rect in r6:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for rect in r8:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    fig.tight_layout()
    
    #plt.show()
    plt.savefig(firDir+'HK1.png',dpi=500)
    
    #HK2
    xx = np.arange(len(newDF))
    yy= newDF['stage'].tolist()
    width = 0.15
    fig, ax = plt.subplots()
    temp = newDF[1].tolist()
    r1 = ax.bar(xx - width, temp, width, label='c1')
    temp = newDF[4].tolist()
    r4 = ax.bar(xx , temp, width, label='c4')
    temp = newDF[7].tolist()
    r7 = ax.bar(xx +width, temp, width, label='c7')
    ax.set_ylabel('prob(%)')
    ax.set_title('stageByClusterProb')
    ax.set_xticks(xx)
    ax.set_xticklabels(yy)
    ax.legend()
    for rect in r1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    for rect in r4:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    for rect in r7:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    fig.tight_layout()
    #plt.show()
    plt.savefig(firDir+'HK2.png',dpi=500)
    
    #SK: Subordinate key areas (frequent in stage 2)
    #9, 13, 15, 17, 18
    xx = np.arange(len(newDF))
    yy= newDF['stage'].tolist()
    width = 0.15
    fig, ax = plt.subplots()
    temp = newDF[9].tolist()
    r9 = ax.bar(xx - width*2, temp, width, label='c9')
    temp = newDF[13].tolist()
    r13 = ax.bar(xx - width, temp, width, label='c13')
    temp = newDF[15].tolist()
    r15 = ax.bar(xx , temp, width, label='c15')
    temp = newDF[17].tolist()
    r17 = ax.bar(xx + width, temp, width, label='c17')
    temp = newDF[18].tolist()
    r18 = ax.bar(xx + width*2, temp, width, label='c18')
    ax.set_ylabel('prob(%)')
    ax.set_title('stageByClusterProb')
    ax.set_xticks(xx)
    ax.set_xticklabels(yy)
    ax.legend()
    
    for rect in r9:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    for rect in r13:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    for rect in r15:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    for rect in r17:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    for rect in r18:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    #plt.show()
    plt.savefig(firDir+'SK.png',dpi=500)
    
    
    
    #Dev: Developmental (frequent in stage 3)
    #0, 10, 14, 16, 19

    xx = np.arange(len(newDF))
    yy= newDF['stage'].tolist()
    width = 0.15
    fig, ax = plt.subplots()
    temp = newDF[0].tolist()
    r0 = ax.bar(xx - width*2, temp, width, label='c0')
    temp = newDF[10].tolist()
    r10 = ax.bar(xx - width, temp, width, label='c10')
    temp = newDF[14].tolist()
    r14 = ax.bar(xx , temp, width, label='c14')
    temp = newDF[16].tolist()
    r16 = ax.bar(xx + width, temp, width, label='c16')
    temp = newDF[19].tolist()
    r19 = ax.bar(xx + width*2, temp, width, label='c19')
    ax.set_ylabel('prob(%)')
    ax.set_title('stageByClusterProb')
    ax.set_xticks(xx)
    ax.set_xticklabels(yy)
    ax.legend()
    
    for rect in r0:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    for rect in r10:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    for rect in r14:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    for rect in r16:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    for rect in r19:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    fig.tight_layout()
    #plt.show()
    plt.savefig(firDir+'DEV.png',dpi=500)
    
    
    #XD: Expo-Dev. (frequent in stages 1, 2, 3)
    #11, 12
    xx = np.arange(len(newDF))
    yy= newDF['stage'].tolist()
    width = 0.15
    fig, ax = plt.subplots()
    temp = newDF[11].tolist()
    r11 = ax.bar(xx - width/2, temp, width, label='c11')
    temp = newDF[12].tolist()
    r12 = ax.bar(xx + width/2, temp, width, label='c12')

    ax.set_ylabel('prob(%)')
    ax.set_title('stageByClusterProb')
    ax.set_xticks(xx)
    ax.set_xticklabels(yy)
    ax.legend()

    for rect in r11:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for rect in r12:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    fig.tight_layout()
    #plt.show()
    plt.savefig(firDir+'XD.png',dpi=500)
    
   
    
    
    ''' 
    HK1: Home key – non-development (frequent in stages 1 and 5, infrequent in 3)
    2, 3, 5, 6, 8,
    
    HK2: Home key – non-ST (frequent in stages 4, 5; infrequent in 2)
    1, 4, 7
    
    SK: Subordinate key areas (frequent in stage 2)
    9, 13, 15, 17, 18
    
    Dev: Developmental (frequent in stage 3)
    0, 10, 14, 16, 19
    
    XD: Expo-Dev. (frequent in stages 1, 2, 3)
    11, 12
    ''' 
    
    
    
    afaf = entireDF['newCentroidIndex'].tolist()
    afaf
    m1 = transition_matrix(afaf)
    for row in m1: print(' '.join('{0:.3f}'.format(x) for x in row))
    #Sum of each row is 1(marcov chain transition matrix)
    
    m2 = transition_prob(afaf)
    for row in m2: print(' '.join('{0:.3f}'.format(x) for x in row))
    #Sum of entire probs is 1
    m2
    
    
    matrix1 =  pd.DataFrame(m1)
    ax = sns.heatmap(matrix1, annot=True, annot_kws={"size": 4.5}, fmt=".3f", cmap="YlGnBu")
    ax.figure.savefig("matrix1.png", dpi=300)
    ax.clear()

    matrix2 =  pd.DataFrame(m2)
    ax = sns.heatmap(matrix2, annot=True, annot_kws={"size": 4.5}, fmt=".3f", cmap="YlGnBu")
    ax.figure.savefig("matrix2.png", dpi=300)
    ax.clear()
    
    matrix2
    
    cwd = os.getcwd()
    fn = 'matrix2.csv'
    fn = os.path.join(cwd,fn)
    matrix2.to_csv(fn, index=True,header=True)
# ==================================test case===========================================
    testchain =[1,1,1,1,2,3,4,5,6,7,8,1,1,1,1]
    n = transition_matrix(testchain)
    for row in n: print(' '.join('{0:.2f}'.format(x) for x in row))
    
    n = transition_prob(testchain)
    for row in n: print(' '.join('{0:.2f}'.format(x) for x in row))
    axt = sns.heatmap(n, annot=True, annot_kws={"size": 4.5}, fmt=".3f", cmap="YlGnBu")
    axt.figure.savefig("axt.png", dpi=300)
    
#   Read row to col
# =============================================================================
      
    from sklearn.decomposition import PCA


    # Run The PCA
    pca = PCA(n_components=12)
    pca.fit(matrix2)
    PCA(copy=True, iterated_power='auto', n_components=12, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)

    cwd = os.getcwd()

    print('singular value :', pca.singular_values_)
    pca_sigular_values = pd.DataFrame(pca.singular_values_)
    fn = 'pca_sigular_values.csv'
    fn = os.path.join(cwd,fn)
    pca_sigular_values.to_csv(fn, index=True,header=True)
    
    print('singular vector :\n', pca.components_.T)
    sigular_vector = pd.DataFrame(pca.components_.T)
    fn = 'sigular_vector.csv'
    fn = os.path.join(cwd,fn)
    sigular_vector.to_csv(fn, index=True,header=True)
    
    
    
    print('eigen_value(variance) :', pca.explained_variance_)
    eigen_values = pd.DataFrame(pca.explained_variance_)
    print('explained variance ratio :', pca.explained_variance_ratio_)
    explained_variance_ratio = pd.DataFrame(pca.explained_variance_ratio_)
    fn = 'explained_variance_ratio.csv'
    fn = os.path.join(cwd,fn)
    explained_variance_ratio.to_csv(fn, index=True,header=True)
   
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print('The cumulative distribution ratio :', d)
# The cumulative distribution ratio 12






####==========================


    with open('afaf.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % af for af in afaf)

    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm


#    ====================================== Year analysis begins=======================
    
    yearCount = entireDF.groupby('year').count()[['index']]
    yearCount.reset_index().set_index('year')
    yearCount.rename(columns = {'index':'TotalCount'},inplace = True)
    
    yearCount.values[2][0]
    p_values = pd.Series([])
    lr_analysis = pd.Series([])
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
        yearCount = yearCount.reset_index()
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
        
    
        
        import scipy.stats as stats
        import pandas as pd
        import urllib
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
        import matplotlib.pyplot as plt
        import numpy as np
            
        entireDF
        
        
        group0 = entireDF[entireDF['newCentroidIndex']== 0][["year"]]
        group1 = entireDF[entireDF['newCentroidIndex']== 1][["year"]]
        group2 = entireDF[entireDF['newCentroidIndex']== 2][["year"]]
        group3 = entireDF[entireDF['newCentroidIndex']== 3][["year"]]
        group4 = entireDF[entireDF['newCentroidIndex']== 4][["year"]]
        group5 = entireDF[entireDF['newCentroidIndex']== 5][["year"]]
        group6 = entireDF[entireDF['newCentroidIndex']== 6][["year"]]
        group7 = entireDF[entireDF['newCentroidIndex']== 7][["year"]]
        group8 = entireDF[entireDF['newCentroidIndex']== 8][["year"]]
        group9 = entireDF[entireDF['newCentroidIndex']== 9][["year"]]
        group10 = entireDF[entireDF['newCentroidIndex']== 10][["year"]]
        group11 = entireDF[entireDF['newCentroidIndex']== 11][["year"]]
        group12 = entireDF[entireDF['newCentroidIndex']== 12][["year"]]
        group13 = entireDF[entireDF['newCentroidIndex']== 13][["year"]]
        group14 = entireDF[entireDF['newCentroidIndex']== 14][["year"]]
        group15 = entireDF[entireDF['newCentroidIndex']== 15][["year"]]
        group16 = entireDF[entireDF['newCentroidIndex']== 16][["year"]]
        group17 = entireDF[entireDF['newCentroidIndex']== 17][["year"]]
        group18 = entireDF[entireDF['newCentroidIndex']== 18][["year"]]
        group19 = entireDF[entireDF['newCentroidIndex']== 19][["year"]]
        

        # matplotlib plotting
        plot_data = [group0, group1, group2, group3, group4, group5, group6, group7, group8, group9, group10, group11, group12, group13, group14, group15, group16, group17, group18, group19]
        ax = plt.boxplot(plot_data)
        plt.show()

        F_statistic, pVal = stats.f_oneway(group0, group1, group2, group3, group4, group5, group6, group7, group8, group9, group10, group11, group12, group13, group14, group15, group16, group17, group18, group19)
        F_statistic
        pVal
            
        year = entireDF["year"].to_numpy()
        cent = entireDF["newCentroidIndex"].to_numpy()
        
        one_sample = [177.3, 182.7, 169.6, 176.3, 180.3, 179.4, 178.5, 177.2, 181.8, 176.5]
        print(mean(one_sample)   # 177.96
        result1 = stats.ttest_1samp(one_sample, 175.6)   # 비교집단, 관측치
        
        print('t검정 통계량 = %.3f, pvalue = %.3f'%(one_sample_result))  



        

        import warnings
        warnings.filterwarnings('ignore')
        
        df = pd.DataFrame(entireDF, columns=['newCentroidIndex', 'year'])    
        
        # the "C" indicates categorical data
        model = ols('newCentroidIndex ~ C(year)', df).fit()
        
        print(anova_lm(model))


        





        from scipy.stats import f_oneway
        fstat, pval = f_oneway(year, cent)
        fstat
        
        
    
    
    
        cwd = os.getcwd()
        fn2 = 'entireDF_m_ps.csv'
        fn2 = os.path.join(cwd,fn2)
        original_df = pd.read_csv(fn2,index_col=0)
        original_df
        figDir = os.path.join(cwd,"smallDF")
        
        firDir = os.path.join(figDir,'ClusterIndex_')
        smallDF = original_df
        
        yearCountCluster = smallDF.groupby(['year', 'filename']).count()[['index']]
        pleaceCount = entireDF.groupby([ 'filename']).count()[['index']]
        yearCountCluster.rename(columns = {'index':'countInCluster'}, inplace = True)
        yearCountCluster = yearCountCluster.reset_index()

        #yearCountCluster = yearCountCluster.drop(columns = ['filename'])
        yearCountCluster = yearCountCluster.sort_values(by = ['filename'])
        pleaceCount = pleaceCount.sort_values(by=['filename'])
        pleaceCount = pleaceCount.reset_index()
    
        yearCount = yearCount.set_index('year')
        yearCountCluster = yearCountCluster.set_index('filename')
        pleaceCount = pleaceCount.set_index('filename')
        
        
        dfdf = pd.concat([pleaceCount, yearCountCluster], axis=1, join='inner')
        #group by the cluster, counted number of cluster then normalized.
        #got p value for each of 'year' and cluster''
        
        dfdf['prob(%)'] = dfdf['countInCluster']/dfdf['index'] * 100
        dfdf = dfdf.reset_index()
        
        X_g = dfdf.iloc[:,2].values.reshape(-1,1)
        Y_g = dfdf.iloc[:,4].values.reshape(-1,1)
        X_l = dfdf.iloc[:,2].values
        X_l = X_l-1774
        Y_l = dfdf.iloc[:,4].values
        
        lr = LinearRegression()
        lr.fit(X_g,Y_g)
        Y_pred = lr.predict(X_g)
        
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(X_l,Y_l)
        p_value = pd.Series(p_value)
        lr_analysis = pd.concat([p_value, lr_analysis], ignore_index=True)
        #marginal significance
        
 
            
        #vis
        # plot the results
        plt.figure(figsize=(8, 8))
        ax = plt.axes()
        plt.scatter(X_g, Y_g)
        plt.plot(X_g, Y_pred, color= 'red')

        ax.set_xlabel('year')
        ax.set_ylabel('prob')
        ax.axis('tight')
        
        
        plt.savefig(firDir+str(v)+'_norm_by_year_linear_reg.png',dpi=500)
        
   
        '''
        # predict y from the data
        x_new = np.linspace(1774, 1790, 100)
        y_new = model.predict(x_new[:, np.newaxis])
        
        # plot the results
        plt.figure(figsize=(8, 8))
        ax = plt.axes()
        ax.scatter(yynp, pnp)
        ax.plot(x_new, y_new)
        
        ax.set_xlabel('year')
        ax.set_ylabel('prob')
        
        ax.axis('tight')
        
        
        plt.savefig(firDir+str(v)+'_norm_by_year_linear_reg.png',dpi=500)

        mod = sm.OLS(yynp, pnp)
        fii = mod.fit()
        p_value = fii.summary2().tables[1]['P>|t|']
        p_values = pd.concat([p_value, p_values], ignore_index=True)
        
        
        
        #xx = np.arange(len(dfdf))
        #yy= dfdf['year'].tolist()
        #width = 0.35
        #tc = dfdf['TotalCount'].tolist()
        #cic = dfdf['countInCluster'].tolist()
        #p = dfdf['prob(%)'].tolist()
        #p = list(np.around(np.array(p),1))
        #fig, ax = plt.subplots()
        #rects1 = ax.bar(xx, p, width,label = 'countInCluster/TotalCount(%)')
        
 
        
        
        #ax.set_ylabel('Probabilities%')
        #ax.set_title('Percentage of data points in the given cluster over all pieces')
        #ax.set_xticks(xx)
        #ax.set_xticklabels(yy)
        #ax.legend()
        #autolabel(rects1)
        
        #fig.tight_layout()
        #plt.show()
        
        #plt.savefig(firDir+str(v)+'_norm_by_year.png',dpi=500)     
        #print("Current working cluster index is: "+str(v))

        #cwd = os.getcwd()
        #fn = 'Index_'+str(v) + '_Cluster by year.csv'
        #fn = os.path.join(cwd,fn)
        #dfdf.to_csv(fn)
    
        #yy and p
        # create a linear regression model
        model = LinearRegression()
        yynp = np.reshape(yy, (6,-1))
        pnp = np.reshape(p, (6,-1))
        model.fit(yynp, pnp)
        # predict y from the data
        x_new = np.linspace(1774, 1790, 100)
        y_new = model.predict(x_new[:, np.newaxis])
        
        # plot the results
        plt.figure(figsize=(8, 8))
        ax = plt.axes()
        ax.scatter(yynp, pnp)
        ax.plot(x_new, y_new)
        
        ax.set_xlabel('year')
        ax.set_ylabel('prob')
        
        ax.axis('tight')
        
        
        plt.savefig(firDir+str(v)+'_norm_by_year_linear_reg.png',dpi=500)

        mod = sm.OLS(yynp, pnp)
        fii = mod.fit()
        p_value = fii.summary2().tables[1]['P>|t|']
        p_values = pd.concat([p_value, p_values], ignore_index=True)
        '''
        
    cwd = os.getcwd()
    fn = 'significance'+ '_clusters_by_year.csv'
    fn = os.path.join(cwd,fn)
    lr_analysis.to_csv(fn)

        
        #=======Year analysis ends =======================
        
        
        
    

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
    plt.show()
    
    
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
        #newDF.set_index('stage')
        
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
#======================== stage by cluster prob analysis ends ================================
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














































