#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multinomial as mlvn
from scipy.stats import bernoulli as brn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[60]:


data=pd.read_csv("MNIST_train.csv")
data2=pd.read_csv("MNIST_test.csv")


# In[61]:


data.head()


# In[15]:


#find a set of values that are your values and your data (X, y)


# In[62]:


def accuracy(y, y_hat):
    return np.mean(y == y_hat)


# In[63]:


os.getcwd()


#  # convert  from pandas to numpy  

# In[64]:


X = data.to_numpy()


# In[65]:


X


# In[66]:


data.info


# In[134]:


data2.info


# ### extract the column that contains the labels (y) which is contained in column 3, to acces col3 so you pass all rows by using (:) and identify the third column using the indes (2), so we have [ : , 2 ] all rows of column 3. the next step is to extract all the features contained in X, which has all rows and columns except the lables and first two useless columns, so to extract them we need all rows anf columns from 4 to all the way to the end. To do that, we extract all rows using the colon operator (:) and all columns from 4 to the end using (3 :) 3 then colon means all columns from 3 to the end. 

# In[67]:


y =X[:,2]


# In[69]:


y
print(y)


# In[70]:


X=X[:,3:]


# In[71]:


X.shape
print(X)


# In[72]:


print(X)


# In[73]:


#plt.iamshow

def show_me(X):
    plt.imshow(X.reshape(28,28))


# In[74]:


show_me(X[0,:])
show_me(X[99,:])


# In[75]:


y[99]


# In[76]:


y[0]


# In[77]:


data.info


# In[78]:


data2.head()


# In[79]:


X2=data2.to_numpy()


# In[80]:


y2=X2[:,2]
y2


# In[81]:


X2=X2[:,3:]
X2
X2=X2/255
X2.shape


# In[ ]:


# use X, y as the training set and y for trainig 


# In[82]:


class GaussNB():
    def fit(self, X, y, epsilon = 1e-3):
        self.likelihoods = dict()
        self.priors = dict()
        
        self.K = set(y.astype(int))
        
        for k in self.K:
            X_k = X[y == k,:]
            self.likelihoods[k] = {"mean":X_k.mean(axis=0), "cov":X_k.var(axis=0) + epsilon}
            self.priors[k] = len(X_k)/len(X)
            
    def predict(self, X):
        N, D = X.shape
        
        P_hat = np.zeros((N,len(self.K)))
        
        for k, l in self.likelihoods.items():
            P_hat[:,k] = mvn.logpdf(X, l["mean"], l["cov"]) + np.log(self.priors[k])
        return P_hat.argmax(axis = 1)


# In[85]:


gnb = GaussNB()
gnb.fit(X,y)
y_hat = gnb.predict(X2)


# In[ ]:


plt.figure()
plt.scatter(X[:,0], X[:,1], c = y_hat, alpha = 0.25)


# In[86]:


print(f"Accuracy: {accuracy(y2, y_hat):0.3f}")


# In[87]:


X=X/255
# divide by 255 to normalize(max value)


# In[88]:


X


# In[89]:


class GaussNB():
    def fit(self, X, y, epsilon = 1e-3):
        self.likelihoods = dict()
        self.priors = dict()
        
        self.K = set(y.astype(int))
        
        for k in self.K:
            X_k = X[y == k,:]
            self.likelihoods[k] = {"mean":X_k.mean(axis=0), "cov":X_k.var(axis=0) + epsilon}
            self.priors[k] = len(X_k)/len(X)
            
    def predict(self, X):
        N, D = X.shape
        
        P_hat = np.zeros((N,len(self.K)))
        
        for k, l in self.likelihoods.items():
            P_hat[:,k] = mvn.logpdf(X, l["mean"], l["cov"]) + np.log(self.priors[k])
        return P_hat.argmax(axis = 1)


# In[90]:


gnb = GaussNB()
gnb.fit(X,y)
y_hat = gnb.predict(X)


# In[91]:


plt.figure()
plt.scatter(X[:,0], X[:,1], c = y_hat, alpha = 0.25)


# In[93]:


print(f"Accuracy: {accuracy(y, y_hat):0.3f}")


# ## Naive Bayes Gaussian is crappy, the accuracy is very low

# ##  Try Gaussian Bayes but not naive, the accuracy should  be much better

# In[94]:


class GaussBayes():
    def fit(self, X, y, epsilon = 1e-3):
        self.likelihoods = dict()
        self.priors = dict()
        
        self.K = set(y.astype(int))
        
        for k in self.K:
            X_k = X[y == k,:]
            N_k, D = X_k.shape
            mu_k=X_k.mean(axis=0)
            self.likelihoods[k] = {"mean":X_k.mean(axis=0), "cov":(1/(N_k-1))*np.matmul((X_k-mu_k).T,X_k-mu_k)+ epsilon*np.identity(D)}
            self.priors[k] = len(X_k)/len(X)
            
    def predict(self, X):
        N, D = X.shape
        
        P_hat = np.zeros((N,len(self.K)))
        
        for k, l in self.likelihoods.items():
            P_hat[:,k] = mvn.logpdf(X, l["mean"], l["cov"]) + np.log(self.priors[k])
            
        return P_hat.argmax(axis = 1)


# In[95]:


gnb = GaussBayes()
gnb.fit(X,y)
y_hat = gnb.predict(X2)


# In[96]:


plt.figure()
plt.scatter(X2[:,0], X2[:,1], c = y_hat, alpha = 0.25)


# In[97]:


print(f"Accuracy: {accuracy(y2, y_hat):0.3f}")


# In[98]:


# Boolean expression takes the indices of y2 when y2 is equal to 1 and put those indices into y,
# and the other one does the same thing but puts the indces into y hat  
y2[y2==1]
y_hat[y2==1]


# In[99]:


y2[y2==1]


# In[100]:


y_hat[y2==1]


# In[101]:


accuracy(y2[y2==1],y_hat[y2==1])


# In[102]:


y2[y2==0]
y_hat[y2==0]
accuracy(y2[y2==0], y_hat[y2==0])


# In[103]:


y2[y2==2]
y_hat[y2==2]
accuracy(y2[y2==2], y_hat[y2==2])


# In[104]:


y2[y2==3]
y_hat[y2==3]
accuracy(y2[y2==3], y_hat[y2==3])


# In[105]:


y2[y2==4]
y_hat[y2==4]
accuracy(y2[y2==4], y_hat[y2==4])


# In[106]:


y2[y2==5]
y_hat[y2==5]
accuracy(y2[y2==5], y_hat[y2==5])


# In[107]:


y2[y2==6]
y_hat[y2==6]
accuracy(y2[y2==6], y_hat[y2==6])


# In[108]:


y2[y2==7]
y_hat[y2==7]
accuracy(y2[y2==7], y_hat[y2==7])


# In[109]:


y2[y2==8]
y_hat[y2==8]
accuracy(y2[y2==8], y_hat[y2==8])


# In[110]:


y2[y2==9]
y_hat[y2==9]
accuracy(y2[y2==9], y_hat[y2==9])


# In[135]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
Classifier = ['Gauss Bayes Non Naive', 'Gauss Bayes Naive ']
Accuracy = [91 , 77]
ax.bar( Classifier,Accuracy, color=["g","r"], width=[0.5,0.5])
plt.title('Accuracy of the used classifiers')
plt.xlabel('Classifier')
plt.ylabel('Accuracy %')
plt.ylim(0,100)
plt.show()


# In[133]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
DigitAccuracy = [0,1,2,3,4,5,6,7,8,9]
Accuracy = [97 ,97, 91, 90, 89,82,94,85,92,92]
ax.bar(DigitAccuracy, Accuracy)
plt.title('Accuracy of Each Digit Using Gauss Bayes Non Naive')
plt.xlabel('Digit')
plt.ylabel('Accuracy %')
ax.set_xticks(np.arange(len(DigitAccuracy)))
plt.show()


# In[ ]:




