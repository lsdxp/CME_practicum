# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 19:21:00 2016

@author: beckswu
"""

# this code is created to read inputs from csv
# then according to the original inputs to generate a csv which has ouput in two columns. The first column has the number in orignal input but sorting in an ascending order. THe second columns has the counting number of the corresponding input variables
# Aslo, the code print the threshold and verify the threshold by counting the number between intervals 



import matplotlib as plt
import csv
import operator
import numpy as np


#fr = open("","r")
data = []
label = []
first = True
i= True
a = 6# variable position

with open("/Users/beckswu/Dropbox/CME practicum/Final report/with1BookRec.csv", newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     headline = True 
     for row in spamreader:
        if headline:
            row = row[0].split(",")
            print(row[a])
            headline = False
        else:
            row = row[0].split(",")
            temp = float(row[a])
            data.append(temp)

number_Count = len(data)
print(number_Count)
"""
for r in fr.readlines():
    if first:
        first = False 
        row = r.strip().split(",")
        label = row[2:]
    else:
        row = r.strip().split(",")
        if i:
            print(row)
            i= False
        temp = float(row[a])
 """       


data.sort()
cur = data[0]
count = []
cou = 0
imba = []

count_123 = []
imba_123 =[]
imba_123.append(0)
imba_123.append(0)
for i in range(3):
    count_123.append(0)
    
first_threshold = int(number_Count/3)
sec_threshold = int(2*number_Count/3)
thres = [first_threshold,sec_threshold]
first = True
second = True
print("threshold number amount : ", thres)
all_sum = 0

imba.append(data[0])
for i in range(len(data)):
    if cur==data[i]:
        cou+=1
        all_sum +=1 
    else:
        cur = data[i]
        imba.append(data[i])
        count.append(cou)
        if all_sum > thres[0] and first:
            first = False
            count_123[0] = all_sum-count[-1] #要减去前一个position的值
            print(all_sum,'  ', count[-1],'  ',imba[-2])
            try:
                imba_123[0] = imba[-3]
            except:
                first = False
        elif all_sum > thres[1] and second:
            second = False
            print(all_sum,'  ', count[-1],'  ',imba[-2])
            count_123[1] = all_sum-count[-1]-count_123[0]
            imba_123[1] = imba[-3]
            count_123[2] = number_Count-count_123[0]-count_123[1]
    
        cou = 1
        all_sum +=1 
       
count.append(cou)
print("\n")
print("number in each interval  ",count_123)
print("threshold :  ",imba_123)
print(len(count)) # 分成几类
print(len(imba))

        
with open("1.csv", 'w') as csvfile:
    fieldnames = ["imbalance","amount"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(count)):
        writer.writerow({'imbalance': imba[i],"amount":count[i]})

#print(i)