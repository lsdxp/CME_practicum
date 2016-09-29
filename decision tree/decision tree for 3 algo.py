# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 01:48:38 2016

@author: beckswu
"""

# this code read input csv files in the function createDataSet(filename):
# then compute the infomation gain and split the dataset in order to get the tree, infomation gain at each node, and probability at each node as dictionary structure
# then auto graph the tree
# BE CAREFUL FOR PLOTTING: should use varaibles smaller than THREE. If bigger than three, the plot is gonna MESSY



import csv
import math
import operator
import matplotlib.pyplot as plt
import numpy as np


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob *math.log(prob,2) #log base 2
    return shannonEnt
    
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = 0
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature,bestInfoGain                      #returns an integer


def calculate_info_gain(dataSet,mylabel):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = 0
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        print(i," ",mylabel[i],"  ",infoGain )
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature,bestInfoGain  
    
def cal_prob(spli_data,feat):
    prob = [0,0,0]
    for i in range(len(spli_data)):
        if spli_data[i][feat] == 0:
            prob[0]+=1
        if spli_data[i][feat] == 1:
            prob[1]+=1
        if spli_data[i][feat] == -1:
            prob[2]+=1
    prob = [key/len(spli_data) for key in prob]
    return prob

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels,best_gain,info_list,pro_list):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0],best_gain,info_list,pro_list#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList),best_gain,info_list,pro_list
    bestFeat, gain = chooseBestFeatureToSplit(dataSet)
    best_gain += gain
    bestFeatLabel = labels[bestFeat]
    prob= cal_prob(dataSet,bestFeat)
    pro_list.append([bestFeatLabel,prob])
    info_list.append([bestFeatLabel,gain])
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value],best_gain,info_list,prob_list =\
        createTree(splitDataSet(dataSet, bestFeat, value),subLabels,\
        best_gain,info_list,pro_list)
    return myTree,best_gain ,info_list ,pro_list          
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel
  
def createDataSet1(filename):
    with open(filename, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     headline = True 
     data = []
     for row in spamreader:
        if headline:
            headline = False
        else:
            temp1 = [key for key in row[0].split(",")] 
            temp2 = [int(key) for key in temp1[0:4]]
            if temp1[-1] == "1":
                temp2.append("U")
            if temp1[-1] == "0":
               temp2.append("D")
            data.append(temp2)
     my_labels = ["I4","V4","I3","V3"]#,\
     #"I2","V2","I1","V1","bid","ask","Imba","WP","vol","imba value", "vol imba value","direction"]
     return data, my_labels
  
def autolabel(rects,ax):
        # attach some text labels
    for rect in rects:
         height = rect.get_height()
         ax.text(rect.get_x() + rect.get_width()/2., 1.0001*height,\
         '%f' % height,ha='center', va='bottom')
    
def graph(N,gain):    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.40      # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, gain, width, color='r')
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('infogain')
    ax.set_title('Compare info gain for each result')
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(('ANN','SVM'))
 
    autolabel(rects1,ax)
    plt.show()
    
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict :#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", \
    rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, \
    plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]) is dict:#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, \
            leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

    
def createDataSet(filename):
    with open(filename, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     headline = True 
     data = []
     threshold = [[-1,0],[-0.00189,0.00567],[-1,0]]
     #value = [["[,-1]","=0","[1,]"],[",0.00189]","(-0.00189,0.00567]","(,0.00567)"],["[,-1]","=0","[1,]"],[",0.00189]"]]
     for row in spamreader:
        if headline:
            headline = False
            temp1 = [key for key in row[0].split(",")] 
        else:
            try:
                temp1 = [key for key in row[0].split(",")] 
                temp = []
                temp.append(temp1[6])
                temp.append(temp1[7])
                temp.append(temp1[4])
                temp2 =[]
                ii = 0
                for key in temp:
                    if float(key)<= threshold[ii][0]:
                        temp2.append(-1)
                    if float(key)<= threshold[ii][1] and float(key)> threshold[ii][0] :
                        temp2.append(0)
                    if float(key)>threshold[ii][1]:
                        temp2.append(1)
                    ii+=1
                #temp2 = [int(key) for key in temp1[4:8]]
                #temp2.append(int(temp1[4]))
                if temp1[-2] == "\"1\"":
                    temp2.append("U")
                if temp1[-2] == "\"-1\"":
                   temp2.append("D")
                if temp1[-2] == "\"0\"":
                   temp2.append("N")
                
                data.append(temp2)
            except:
                pass
     #my_labels = ["BPC","APC","ASC","BSC","imbaC","wp","AS","BS","imba","vol"book1.imbalance,imba.lag3,vol.lag3,imba.lag2,vol.lag2,imba.lag1,vol.lag1,imba.lag3.1,vol.lag3.1,trade.ask.size.change,trade.bid.size.change,trade.imba.change,trade.weighted.price.change,ask.size,bid.size,imba,vol,vol.change,direction,pred
     my_labels = ["imba","wp","ask"]
     
     #"I4","V4","I3","V3"]#,\
     #"I2","V2","I1","V1","bid","ask","Imba","WP","vol","imba value", "vol imba value","direction"]
     return data, my_labels
     
def createDataSet2(filename):
    with open(filename, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     headline = True 
     data = []
     Best_ask_size_changeB1=[-2.0, -1.0]
     Best_bid_size_changeB1= [-2.0, -1.0]
     bid_price_changeB1=[-0.06291, -0.00012]
     ask_price_changeB1= [-0.00024, 0.21957]
     ask_size_changeB1= [-2.0, -1.0]
     bid_size_changeB1= [-2.0, -1.0]
     imbalance_changeB1=[-2.0, 0.0]
     weighted_price_changeB1= [-0.00267, 0.02126]
     best_ask_sizeB1=[15.0, 80.0]
     best_bid_sizeB1 =[15.0, 85.0]
     ask_sizeB1 = [1105.0, 2185.0]
     bid_sizeB1 = [1165.0, 2210.0]
     imbalanceB1 = [-176.0, 102.0]
     
     best_ask_sizeL1 = [35.0, 100.0]
     best_bid_sizeL1 = [35.0, 100.0]
     imbalanceL1 = [-262.0, 194.0]
     
     best_ask_size_change =  [-3.0, 1.0]
     best_bid_size_change =  [-3.0, 1.0]
     bid_price_change = [-0.25547, 0.17278]
     ask_price_change = [-0.17579, 0.257859999983339]
     ask_size_change = [-4.0, 2.0]
     bid_size_change =[-4.0, 3.0]
     imba_change = [-10.0, 8.0]
     weighted_change = [-0.09533, 0.0969]
     ask_size  = [1440.0, 2325.0]
     bid_size  =  [1515.0, 2360.0]
     best_ask_size = [35.0, 100.0]
     best_bid_size = [35.0, 100.0]
     imba = [-261.0, 192.0]
     direction = [-1.0, 0]

     threshold = [Best_ask_size_changeB1,\
     Best_bid_size_changeB1,\
     bid_price_changeB1,\
     ask_price_changeB1,\
     ask_size_changeB1,\
     bid_size_changeB1,\
     imbalance_changeB1,\
     weighted_price_changeB1,\
     best_ask_sizeB1,\
     best_bid_sizeB1 ,\
     ask_sizeB1,\
     bid_sizeB1 ,\
     imbalanceB1 ,\
     
     best_ask_sizeL1 ,\
     best_bid_sizeL1 ,\
     imbalanceL1,\
     
     best_ask_size_change,\
     best_bid_size_change,\
     bid_price_change ,\
     ask_price_change,\
     ask_size_change ,\
     bid_size_change ,\
     imba_change ,\
     weighted_change ,\
     ask_size ,\
     bid_size ,\
     best_ask_size,\
     best_bid_size ,\
     imba,direction]
     
     my_labels = ["Best_ask_size_changeB1",\
     "Best_bid_size_changeB1",\
     "bid_price_changeB1",\
     "ask_price_changeB1",\
     "ask_size_changeB1",\
     "bid_size_changeB1",\
     "imbalance_changeB1",\
     "weighted_price_changeB1",\
     "best_ask_sizeB1",\
     "best_bid_sizeB1" ,\
     "ask_sizeB1",\
     "bid_sizeB1 ",\
     "imbalanceB1" ,\
     
     "best_ask_sizeL1 ",\
     "best_bid_sizeL1 ",\
     "imbalanceL1",\
     
     "best_ask_size_change",\
     "best_bid_size_change",\
     "bid_price_change" ,\
     "ask_price_change",\
     "ask_size_change" ,\
     "bid_size_change" ,\
     "imba_change",\
     "weighted_change" ,\
     "ask_size" ,\
     "bid_size ",\
     "best_ask_size",\
     "best_bid_size" ,\
     "imba","direction"]
     #value = [["[,-1]","=0","[1,]"],[",0.00189]","(-0.00189,0.00567]","(,0.00567)"],["[,-1]","=0","[1,]"],[",0.00189]"]]
     for row in spamreader:
        if headline:
            headline = False
            temp1 = [key for key in row[0].split(",")] 
        else:
                temp =[]
                temp1 = [key for key in row[0].split(",")] 
                pos_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,22,23,24,26,27,28,\
                29,30,31,32,33,34,35,36,37,38,41]
                for key in pos_list:
                    temp.append(temp1[key])

                temp2 =[]
                ii = 0
                for key in temp:
                    if float(key)<= threshold[ii][0]:
                        temp2.append(-1)
                    if float(key)<= threshold[ii][1] and float(key)> threshold[ii][0] :
                        temp2.append(0)
                    if float(key)>threshold[ii][1]:
                        temp2.append(1)
                    ii+=1
                #temp2 = [int(key) for key in temp1[4:8]]
                #temp2.append(int(temp1[4]))
                if temp1[-2] == "1":
                    temp2.append("U")
                if temp1[-2] == "-1":
                   temp2.append("D")
                if temp1[-2] == "0":
                   temp2.append("N")

                data.append(temp2)
                
 
     #print(len(data))
     #"I4","V4","I3","V3"]#,\
     #"I2","V2","I1","V1","bid","ask","Imba","WP","vol","imba value", "vol imba value","direction"]
     return data, my_labels
     
def ANN(filename):
    with open(filename, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     headline = True 
     data = []

     best_ask_size_changeB1 =  [-2.0, 1.0]
     bid_size_changeB1  =  [-2.0, 1.0]
     direction = [-1,0]


     
     threshold = [\
     best_ask_size_changeB1 ,\
     bid_size_changeB1 ,direction]
     
     my_labels = [
    "B1 BASC",\
     "B1 BSC",\
     "direction"]
     #value = [["[,-1]","=0","[1,]"],[",0.00189]","(-0.00189,0.00567]","(,0.00567)"],["[,-1]","=0","[1,]"],[",0.00189]"]]
     for row in spamreader:
        if headline:
            headline = False
            temp1 = [key for key in row[0].split(",")] 
        else:
                temp =[]
                temp1 = [key for key in row[0].split(",")] 
                pos_list = [1,6,41]
                for key in pos_list:
                    temp.append(temp1[key])

                temp2 =[]
                ii = 0
                for key in temp:
                    if ii==2:
                        if float(key)<= threshold[ii][0]:
                            temp2.append("sell side")
                        if float(key)<= threshold[ii][1] and float(key)> threshold[ii][0] :
                            temp2.append(0)
                        if float(key)>threshold[ii][1]:
                            temp2.append("buy side")
                    
                    else:
                        if float(key)<= threshold[ii][0]:
                            temp2.append(-1)
                        if float(key)<= threshold[ii][1] and float(key)> threshold[ii][0] :
                            temp2.append(0)
                        if float(key)>threshold[ii][1]:
                            temp2.append(1)
                        ii+=1
                
                #temp2 = [int(key) for key in temp1[4:8]]
                #temp2.append(int(temp1[4]))
                if temp1[-2] == "1":
                    temp2.append("U")
                if temp1[-2] == "-1":
                   temp2.append("D")
                if temp1[-2] == "0":
                   temp2.append("N")

                data.append(temp2)
                #print(temp2)
 
     #print(len(data))
     #"I4","V4","I3","V3"]#,\
     #"I2","V2","I1","V1","bid","ask","Imba","WP","vol","imba value", "vol imba value","direction"]
     return data, my_labels

def SVM(filename):
    with open(filename, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     headline = True 
     data = []

     best_ask_size_change =  [-2.0, 1.0]
     best_bid_size_change =  [-2.0, 1.0]
     direction = [-1,0]
     
     imbalance_changeB1=[-2.0, 0.0]
     imba = [-261.0, 192.0] 
     imbalanceB1 = [-176.0, 102.0]

     threshold = [\
     imbalance_changeB1,\
     imbalanceB1,imba]
     
     my_labels = [
    "B1 iC",\
     "B1 i",\
     "i"]
     #value = [["[,-1]","=0","[1,]"],[",0.00189]","(-0.00189,0.00567]","(,0.00567)"],["[,-1]","=0","[1,]"],[",0.00189]"]]
     for row in spamreader:
        if headline:
            headline = False
            temp1 = [key for key in row[0].split(",")] 
        else:
                temp =[]
                temp1 = [key for key in row[0].split(",")] 
                pos_list = [6,11,38]
                for key in pos_list:
                    temp.append(temp1[key])

                temp2 =[]
                ii = 0
                for key in temp:
                    if ii==3:
                        if float(key)<= threshold[ii][0]:
                            temp2.append("sell side")
                        if float(key)<= threshold[ii][1] and float(key)> threshold[ii][0] :
                            temp2.append(0)
                        if float(key)>threshold[ii][1]:
                            temp2.append("buy side")
                    
                    else:
                        if float(key)<= threshold[ii][0]:
                            temp2.append(-1)
                        if float(key)<= threshold[ii][1] and float(key)> threshold[ii][0] :
                            temp2.append(0)
                        if float(key)>threshold[ii][1]:
                            temp2.append(1)
                        ii+=1
                #temp2 = [int(key) for key in temp1[4:8]]
                #temp2.append(int(temp1[4]))
                if temp1[-2] == "1":
                    temp2.append("U")
                if temp1[-2] == "-1":
                   temp2.append("D")
                if temp1[-2] == "0":
                   temp2.append("N")

                data.append(temp2)
     #print(len(data))
     #"I4","V4","I3","V3"]#,\
     #"I2","V2","I1","V1","bid","ask","Imba","WP","vol","imba value", "vol imba value","direction"]
     return data, my_labels


if __name__=='__main__':
    csv_set = ["/Users/beckswu/Dropbox/CME practicum/Final report/ANN/lag1data8020.csv",\
    "/Users/beckswu/Dropbox/CME practicum/Final report/SVM/testing_set/with1_80_201.csv"]#,\
    #"/Users/beckswu/Dropbox/CME practicum/Apr 15th/ANN/decision tree/lag1.csv"]
    gain=[]
    # find information gain
    for i in range(len(csv_set)):
        dataset, labels = createDataSet2(csv_set[i])
        a,b=calculate_info_gain(dataset,labels)
        print("\n")
        best_gain = 0.0
        probability_list = []
        info_list = []
        
    # get the tree
    for i in range(len(csv_set)):
        if i == 0:
            dataset, labels = ANN(csv_set[i])
        else:
            dataset, labels = ANN(csv_set[i])
        
        best_gain = 0.0
        probability_list = []
        info_list = []
        labels.append("decision")
        tree,infogain,info_list,probability_list = createTree(dataset,labels,best_gain,info_list,probability_list)
        print("\n")
        print(labels)
        print("\n")
        print("\n",tree,"\n")
        gain.append(infogain)
        print("total infogain :  ",infogain,'\n')
        print("Infomation Gain At Each Node: \n")
        print(info_list)
        print("\n")
        print("probability At Each Node: \n")
        print(probability_list)
        print("\n")
        decisionNode = dict(boxstyle = "sawtooth",fc='0.8')
        leafNode = dict(boxstyle='round4',fc='0.8')
        arrow_args = dict(arrowstyle="<-")
        createPlot(tree)
        print("I4: Volume imbalance lag 4, V4: volume lag 4, I3:Volume imbalance lag 4, V3: volume lag 3")
                
    
    graph(len(csv_set),gain)
    
    

    
    
            
    
    
    
    
       

    
  
 
    
    
    
    
