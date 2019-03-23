#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 18:38:18 2019

@author: aakash
"""

import random

import math
import numpy as np
import matplotlib.pyplot as plt

def x_generate(points):
    x=[]
    x=np.random.uniform(0.0,1.0,points)
    return x
 ################################
def y_generate(x):
    p=math.pi*2
    y=[]
    for n in range(len(x)):
        y.append(math.sin(p*x[n]))
    
   # print (y)
        
    for n in range(len(x)):
        y[n]=y[n]+np.random.normal(0,0.3)
    
    
    return y
############################################
### a part is over
    
def min_squared_gradient(x,y,n,alpha):
    #alpha=0.05
    
    currw=[0]*(n+1)
    for i in range(100):
        sum1=0
        prevw=[0]*(n+1)
        for s in range(n+1):
            for e in range(len(x)):
                sum2=0
                for a in range(n+1):
                    sum2=sum2+currw[a]*pow(x[e],a)
                    #print(pow(x[e],a))
                sum1=sum1+(sum2-y[e])*pow(x[e],s)
            prevw[s]=currw[s]-alpha*1/len(x)*sum1
        for s in range(n+1):
            currw[s]=prevw[s]
        ##print (w)
    return currw
###########################################################
def fourth_power_gradient(x,y,n,alpha):
    #alpha=0.05
    
    currw=[0]*(n+1)
    for i in range(100):
        sum1=0.00
        prevw=[0]*(n+1)
        for s in range(n+1):
            for e in range(len(x)):
                sum2=0
                for a in range(n+1):
                    sum2=sum2+currw[a]*pow(x[e],a)
                    #print(pow(x[e],a))
                sum1=sum1+pow((sum2-y[e]),3)*pow(x[e],s)*4
            prevw[s]=currw[s]-alpha*1/len(x)*sum1
        for s in range(n+1):
            currw[s]=prevw[s]
        ##print (w)
    return currw
#############################################################           
def mean_absolute_gradient(x,y,n,alpha):
    #alpha=0.05
    
    currw=[0]*(n+1)
    for i in range(100):
        sum1=0
        prevw=[0]*(n+1)
        for s in range(n+1):
            for e in range(len(x)):
                sum2=0
                for a in range(n+1):
                    sum2=sum2+currw[a]*pow(x[e],a)
                sum2=sum2-y[e]
                if sum2>=0 :
                    sum1=sum1+pow(x[e],s)
                else:
                    sum1=sum1-pow(x[e],s)
            prevw[s]=currw[s]-alpha*1/len(x)*sum1
        for s in range(n+1):
            currw[s]=prevw[s]
        ##print (w)
    return currw

####################################################################
    
def min_square_error(x,degree,theta,y):
    error=0.00000
    degree=degree+1
    for a in range (len(x)):
        add=0
        for w in range(degree):
            add=add+theta[w]*pow(x[a],w)
        add=add-y[a]
        add=add*add
        error=error+add
    print("test_error",error/(2*len(x)))
    error=error/(2*len(x))
    return error
###################################################################


def mean_absolute_error(x,degree,theta,y):
    error=0.0000
    degree=degree+1
    for a in range (len(x)):
        add=0.0000
        for w in range(degree):
            add=add+theta[w]*pow(x[a],w)
        add=add-y[a]
        if add<0:
            add=-add    
        error=error+add
    print("test_error",error/(2*len(x)))
    error=error/(2*len(x))
    return error
###############################################################


def fourth_power_error(x,degree,theta,y):
    error=0.0000
    degree=degree+1
    for a in range (len(x)):
        add=0.0000
        for w in range(degree):
            add=add+theta[w]*pow(x[a],w)
        add=add-y[a]
        error=error+add**4
        #print 
           
    print("test_error",error/(2*len(x)))
    error=error/(2*len(x))
    return error
            
#################################################### 


def calculate_y(x):
    y=[0]*len(x)
    for q in range(len(x)):
        add=0.000
        for w in range(i):
            add=add+theta[w]*pow(x[q],w)
        y[q]=add
    return y
        

#################################


test_error=[[[ 0.000 for _ in range(9)] for _ in range(4)] for _ in range(3)]
training_error=[[[ 0.0000 for _ in range(9)] for _ in range(4)] for _ in range(3)]



for iteration in range(3):
    number=[10,100,1000,10000]
    number_point=0
    for data_points in number:
        
        ####Generate x random values in range 0,1#################
        x=[]    
        x=x_generate(data_points)   
        
        
        ##### Generate values of y
        random.shuffle(x)           
        training_x=x[:int(0.8*len(x))]
        test_x=x[int(0.8*len(x)):]
        y=y_generate(x)
        
        
        #####Save y
        training_y=y[:int(0.8*len(y))]
        test_y=y[int(0.8*len(y)):]
        ### B PART DONE
        
        
        #####plot original data
        plt.plot(x,y,'ro')
        plt.savefig('/home/aakash/document/original_data.jpg')
        plt.show()
       
        
        ###### calculate cost function and error
        for i in range(0,9):
            estimated_y=[0]*len(test_x)
            
            
            if iteration==0:
                theta=min_squared_gradient(training_x,training_y,i+1,0.05)
                test_error[iteration][number_point][i]=min_square_error(test_x,i,theta,test_y)
                training_error[iteration][number_point][i]=min_square_error(training_x,i,theta,training_y)
            elif iteration==1:
                theta=fourth_power_gradient(training_x,training_y,i+1,0.05)
                test_error[iteration][number_point][i]=fourth_power_error(test_x,i,theta,test_y)
                training_error[iteration][number_point][i]=fourth_power_error(training_x,i,theta,training_y)
            else:
                theta=mean_absolute_gradient(training_x,training_y,i+1,0.05)
                test_error[iteration][number_point][i]=mean_absolute_error(test_x,i,theta,test_y)
                training_error[iteration][number_point][i]=mean_absolute_error(training_x,i,theta,training_y)
                
            estimated_y=calculate_y(x)
            
            plt.plot(x,estimated_y, 'ro')
            plt.savefig('/home/aakash/document--test--'+str(iteration)+'--'+str(data_points)+str(i)+'.jpg')
            plt.show()
        
        ## plot errors for each data point set
        arr=[1,2,3,4,5,6,7,8,9]
        plt.plot(arr,test_error[iteration][number_point],'ro',arr,training_error[iteration][number_point],'bs')
        plt.savefig('/home/aakash/document--error--'+str(iteration)+'--'+ str(data_points)+'.jpg')
        plt.show()
    iteration=iteration+1
    
    
#    for b in range(4):
#        for c in range(9):
#            arr=[1,2,3,4,5,6,7,8,9]
#            plt.plot(arr,test_error[0][b][c],'ro',arr,training_error[1][b][c],'bs',arr,training_error[2][b][c],'rs')
#            plt.savefig('/home/aakash/document--errorcumulative--'+str(iteration)+'--'+ str(data_points)+'.jpg')
#            plt.show()
#                
        ##### 2 QUESTION DONE
    

