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
        sum1=0.0000000
        prevw=[0]*(n+1)
        for s in range(n+1):
            for e in range(len(x)):
                sum2=0.0000
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
        sum1=0.00000000
        prevw=[0]*(n+1)
        for s in range(n+1):
            for e in range(len(x)):
                sum2=0.00000
                for a in range(n+1):
                    sum2=sum2+currw[a]*pow(x[e],a)
                    #print(pow(x[e],a))
                sum1=sum1+pow((sum2-y[e]),3)*pow(x[e],s)*4
                ##print (sum1)
            prevw[s]=currw[s]-alpha*1/(len(x))*sum1
            ##print (prevw[s])
        for s in range(n+1):
            currw[s]=prevw[s]
        ##print (w)
    return currw
#############################################################           
def mean_absolute_gradient(x,y,n,alpha):
    alpha=0.05
    
    currw=[0]*(n+1)
    for i in range(100):
        sum1=0.00000000
        prevw=[0]*(n+1)
        for s in range(n+1):
            for e in range(len(x)):
                sum2=0.000000
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
    error=0
    degree=degree+1
    for a in range (len(x)):
        add=0
        for w in range(degree):
            add=add+theta[w]*pow(x[a],w)
        add=add-y[a]
        add=add*add
        error=error+add
    #print("test_error",error/(2*len(x)))
    error=error/(2*len(x))
    return error
###################################################################


def mean_absolute_error(x,degree,theta,y):
    error=0
    degree=degree+1
    for a in range (len(x)):
        add=0
        for w in range(degree):
            add=add+theta[w]*pow(x[a],w)
        add=add-y[a]
        if add<0:
            add=-add    
        error=error+add
    #print("test_error",error/(2*len(x)))
    error=error/(2*len(x))
    return error
###############################################################


def fourth_power_error(x,degree,theta,y):
    error=0
    degree=degree+1
    for a in range (len(x)):
        add=0
        for w in range(degree):
            add=add+theta[w]*pow(x[a],w)
        add=add-y[a]
        error=pow(add,4)
           
    #print("test_error",error/(2*len(x)))
    error=error/(2*len(x))
    return error
            
#################################################### 


def calculate_y(x,theta,n):
    y=[0]*len(x)
    for q in range(len(x)):
        add=0
        for w in range(n):
            add=add+theta[w]*pow(x[q],w)
        y[q]=add
    return y
        

#################################


test_error=[[ 0 for _ in range(5)] for _ in range(3)]
training_error=[[ 0 for _ in range(5)] for _ in range(3)] 


data_points=[1000]
test=[0.025, 0.05, 0.1, 0.2,0.5]

alpha_iterate=0
for alpha in test:
    for iteration in range(3):   
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
        
           
        
        ###### calculate cost function and error
        
        
        
        
        if iteration==0:
            theta=min_squared_gradient(training_x,training_y,4,alpha)
            print(alpha_iterate)
            test_error[iteration][alpha_iterate]=min_square_error(test_x,4,theta,y)
            training_error[iteration][alpha_iterate]=min_square_error(training_x,4,theta,y)
        elif iteration==1:
            theta=fourth_power_gradient(training_x,training_y,4,alpha)
            test_error[iteration][alpha_iterate]=fourth_power_error(test_x,4,theta,y)
            training_error[iteration][alpha_iterate]=fourth_power_error(training_x,4,theta,y)
        else:
            theta=mean_absolute_gradient(training_x,training_y,4,alpha)
            test_error[iteration][alpha_iterate]=mean_absolute_error(test_x,4,theta,y)
            training_error[iteration][alpha_iterate]=mean_absolute_error(training_x,4,theta,y)
        
        estimated_y=[0]*len(test_x)    
        estimated_y=calculate_y(x,theta,4)
    alpha_iterate=alpha_iterate+1;
        
plt.plot(test,test_error[0], 'ro',test,test_error[1], 'bo',test,test_error[2], 'go')
plt.savefig('/home/aakash/Documents/--alpha--'+str(alpha)+'--.jpg')
plt.show()
    
        ## plot errors for each data point set
        
        

    

    
    
#    for b in range(4):
#        for c in range(9):
#            arr=[1,2,3,4,5,6,7,8,9]
#            plt.plot(arr,test_error[0][b][c],'ro',arr,training_error[1][b][c],'bs',arr,training_error[2][b][c],'rs')
#            plt.savefig('/home/aakash/document--errorcumulative--'+str(iteration)+'--'+ str(data_points)+'.jpg')
#            plt.show()
#                
        ##### 2 QUESTION DONE
    

