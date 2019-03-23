#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:30:23 2019

@author: aakash
"""
import math
import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class node:
    def __init__(self):
        self.children = []
        self.children_attribute=[]
        self.gini=0
    def add(self,name,data,level,gini):
        self.name = name
        self.data=data
        self.level=level+1
        self.gini=gini
    def add1(self,data,level):
        self.data=data
        self.name="label"
        self.value=data.loc[0,"label"]
        self.level=level+1
    def add_child(self, x):
        self.children.append(x)
    def add_attribute(self,x):
        self.children_attribute.append(x)
    def set_gini(self,x):
        self.gini=x


d = pd.read_csv('/home/aakash/Downloads/data/traindata.txt',delimiter="\t")
print(d)
data=pd.DataFrame(0,np.arange(1,1062),np.arange(1,3567))
for i in range(0,d.shape[0]):
    print(d.iloc[i,0])
    print(d.iloc[i,1])
    if data.iloc[d.iloc[i,0],d.iloc[i,1]]==0:
        data.iloc[d.iloc[i,0],d.iloc[i,1]]=1

print(data)
result=pd.read_csv('/home/aakash/Downloads/data/trainlabel.txt')

labels=list(data.columns)

def split_data1(labels,data):
    entropy=2
    for name in labels:
        
        count=[ [0 for _ in range(data.shape[0])]for _ in range(2)]
        store_label=[]
        flag2=0
        k=0
        for j in range(0,data.shape[0]):
            a=0
            flag=False
            for d in store_label:
                if (d==data.iloc[j,name]):
                    flag=True
                    if (result.iloc[k]==1) :
                        count[0][a]=count[0][a]+1
                    else :
                        count[1][a]=count[1][a]+1
                    k=k+1
                    break
                a=a+1
            if flag==False :
                store_label.append(data.loc[j,name])
                if (result.iloc[k]==1) :
                    count[0][a]=count[0][a]+1
                else :
                    count[1][a]=count[1][a]+1
                k=k+1
        
        x=gini_index(count)
        if x<entropy:
            labelled=store_label
            least=name
            entropy=x
    return labelled,least,entropy
    


def split_data2(gain,labels,data):
    
    
    profit=labels
    entropy=-1000
    for name in labels:
        
        count=[ [0 for _ in range(data.shape[0])]for _ in range(2)]
        store_label=[]
        flag2=0
        k=0
        for j in range(0,data.shape[0]):
            a=0
            flag=False
            for d in store_label:
                if (d==data.loc[j,name]):
                    flag=True
                    if (result.iloc[k]==1) :
                        count[0][a]=count[0][a]+1
                    else :
                        count[1][a]=count[1][a]+1
                    k=k+1
                    break
                a=a+1
            if flag==False :
                store_label.append(data.loc[j,name])
                if (result.iloc[k]==1) :
                    count[0][a]=count[0][a]+1
                else :
                    count[1][a]=count[1][a]+1
                k=k+1
        
        x=information_gain(gain,count)
        y=calculate_entropy(count) 
        if x>entropy:
            labelled=store_label
            least=name
            entropy=x

    return labelled,least,y

    
    


        
def tree(node1,data,level,labels):
    if (level<10):
        split_labels,attribute_split,gini=split_data2(node1.gini,labels,data)
        node1.add(attribute_split,data,level,gini)
        
        i=0
        for q in range(0,len(split_labels)):
            split_tag=pd.DataFrame()
            for t in range (0,data.shape[0]):
                print(data.shape[0])
                if (data.iloc[t,attribute_split]==split_labels[q]):
                    
                    split_tag=split_tag.append(data.iloc[t,:])
            node2=node()
            node1.add_attribute(split_labels[q])
            node1.add_child(node2)
            split_tag2=split_tag.reset_index(drop=True)
            split_tag2=split_tag2.drop([attribute_split],axis=1)
            labels2=labels.copy()
            labels2.remove(attribute_split) 
            tree(node2,split_tag2,level+1,labels2) 
            i=i+1 
    else :
        node1.add1(data,level)
    return node1


def calculate_entropy(count):
    total_point1=0
    total_point2=0
    for i in range(0,len(count[0])):
        total_point1=total_point1+count[0][i]
    for j in range(0,len(count[0])):
        total_point2=total_point2+count[1][j]
    if total_point1!=0 and total_point2!=0:
        entropy=((total_point1/(total_point1+total_point2))*math.log(total_point1/(total_point1+total_point2),2)+math.log(total_point2/(total_point1+total_point2),2)*(total_point2/(total_point1+total_point2)))
    else:
        entropy=0
    return -entropy

    
def information_gain(parent,count):
    x=0
    total=0
    for i in range(0,len(count[0])):
        total=total+count[0][i]+count[1][i]
    for i in range(0,len(count[0])):
        total_point=0
        total_point=total_point+count[0][i]+count[1][i]
        if count[0][i]!=0 and count[1][i]!=0:
            entropy=-(math.log(count[0][i]/total_point,2)*count[0][i]/total_point+math.log(count[1][i]/total_point,2)*count[1][i]/total_point)
        else:
            entropy=0
        
        x=x+entropy*(total_point/total)
    return parent-x



def gini_index(count):
    net=0
    size=sum(count[0])+sum(count[1])
    for i in range(len(count[0])):
        total=0
        den=count[0][i]+count[1][i]
        if den==0:
            continue
        else:
            total=total+pow(count[0][i]/den,2)+pow(count[1][i]/den,2)
        net=net+(count[0][i]+count[1][i])/size*(1-total)
    return 1-net


root=node()
root.set_gini(1)
root=tree(root,data,0,labels)
def test(attributes,it):
    m=0
    while len(it.children)>0:
        k=attributes.loc[0,it.name]
        for i in range(len(it.children_attribute)):
            if k==it.children_attribute[i]:
                break
        if m!=0:
            print('|')
        for k in range(i):
            print(' ')
        print(it.name)
        print("=")
        print(it.children_attribute[i])

        it=it.children[i]
        m=m+4
    print('|')
    print(' ')
    print(it.name)
    print("=")
    print(it.value)
    return it.value

it=root
test_data = pd.read_csv('/home/aakash/Downloads/data/testdata.csv',delimiter="\t")
test_data1=pd.DataFrame(0,np.arange(1,709),np.arange(1,3567))
for i in range(0,test_data.shape[0]):
    if test_data1.iloc[test_data.iloc[i,0],test_data.iloc[i,1]]==0:
        test_data1.iloc[test_data.iloc[i,0],test_data.iloc[i,1]]=1

test_label = pd.read_csv('/home/aakash/Downloads/data/testlabel.csv',delimiter="\t")
test_label1=pd.DataFrame(0,np.arange(1,709),np.arange(1,3567))


train_x=pd.DataFrame()
train_y=pd.DataFrame()

r=0
for r in range(test_data1.shape[0]):
    x=test(test_data1.iloc[r,:],it)
    if x==test_label1[r,:]:
        r=r+1
accuracy2=r/(test_data1.shape[0])


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(test_data1, test_label1)
predict=pd.DataFrame()
x=clf_gini.predict(predict)
print(x)
count=0
for r in range(len(x)):
    if x[r]==test_label1.iloc[0,r]:
        count=count+1
accuracy=count/len(x)


