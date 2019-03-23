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
        self.name="profitable"
        self.value=data.loc[0,"profitable"]
        self.level=level+1
        
    def add_child(self, x):
        self.children.append(x)
    def add_attribute(self,x):
        self.children_attribute.append(x)
    def set_gini(self,x):
        self.gini=x

data = pd.read_csv('/home/aakash/Downloads/data.csv')
labels=[]
for i in data.columns:
    labels.append(i)

def split_data1(labels,data):
    
    
    profit=data[["profitable"]]
    entropy=2
    for name in labels:
        if name!="profitable":
            count=[ [0 for _ in range(3)]for _ in range(2)]
        
            store_label=[]
            flag2=0
            k=0
            for j in range(0,data.shape[0]):
                a=0
                flag=False
                for d in store_label:
                    if (d==data.loc[j,name]):
                        flag=True
                        if (profit.iloc[k]=="yes").any() :
                            count[0][a]=count[0][a]+1
                        else :
                            count[1][a]=count[1][a]+1
                        k=k+1
                        break
                    a=a+1
                if flag==False :
                    store_label.append(data.loc[j,name])
                    if (profit.iloc[k]=="yes").any() :
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
    
    
    profit=data[["profitable"]]
    entropy=-1000
    for name in labels:
        if name!="profitable":
            count=[ [0 for _ in range(3)]for _ in range(2)]
            store_label=[]
            flag2=0
            k=0
            for j in range(0,data.shape[0]):
                a=0
                flag=False
                for d in store_label:
                    if (d==data.loc[j,name]):
                        flag=True
                        if (profit.iloc[k]=="yes").any() :
                            count[0][a]=count[0][a]+1
                        else :
                            count[1][a]=count[1][a]+1
                        k=k+1
                        break
                    a=a+1
                if flag==False :
                    store_label.append(data.loc[j,name])
                    if (profit.iloc[k]=="yes").any() :
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
    if (data.shape[0]>1):
        split_labels,attribute_split,gini=split_data2(node1.gini,labels,data)
        node1.add(attribute_split,data,level,gini)
        i=0
        for q in range(0,len(split_labels)):
            split_tag=pd.DataFrame()
            for t in range (0,data.shape[0]):
                if (data.loc[t,attribute_split]==split_labels[q]):
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
        

        for k in range(m):
            print('\t',end=""),
        print('|',end=""),        
        print(it.name,"=",it.children_attribute[i])
        

        it=it.children[i]
        m=m+1
    
    for k in range(m):
        print("\t",end=""),
    print('|',end="")
    print(it.name,"=",it.value)
    
it=root
test_data = pd.read_csv('/home/aakash/Downloads/Test_Data.csv')
attributes=pd.DataFrame()
attributes=attributes.append(test_data.iloc[1,:])
attributes=attributes.reset_index(drop=True)

test(attributes,root)
train_x=pd.DataFrame()
train_y=pd.DataFrame()

train_x=data.drop(["profitable"],axis=1)
train_y=train_y.append(data.loc[:,"profitable"])
train_y=train_y.reset_index(drop=True)
train_y=train_y.T
le = preprocessing.LabelEncoder()
train_x = train_x.apply(le.fit_transform)
train_y = train_y.apply(le.fit_transform)
train_y = train_y.rename(columns = {0: "profitable"})
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(train_x, train_y)
predict=pd.DataFrame()

predict=test_data.drop(["profitable"],axis=1)
predict = predict.apply(le.fit_transform)
x=clf_gini.predict(predict)
r=0
