#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENG 474 Assignment 1
Benjamin Frizzell (V00932255)
"""

import sklearn as sk
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt


# function to compute k-fold cross validation
def k_cross_validation(df,model,k=5):
    
    # array to hold validation error
    valid_err = []
    
    # separate data into k partitions
    partitions = np.array_split(df,k)
    
    # compute validation error
    for i,df_valid in enumerate(partitions):
        
        # training set = compliment of validation set
        train_partitions = [x for j,x in enumerate(partitions) if j != i]
        df_train = pd.concat(train_partitions)
        
        # get features and labels for training,validation sets
        X_train,Y_train = df_train.iloc[:, :-1],df_train.iloc[:,-1]
        X_valid,Y_valid =  df_valid.iloc[:, :-1],df_valid.iloc[:,-1]
        
        # training model
        trained_model = model.fit(X_train,Y_train)
        
        # testing model on validation set
        valid_err.append(1-trained_model.score(X_valid,Y_valid))
    
    return np.mean(valid_err)


# %% 

# set seed for reproducability
SEED = 10 

# %% 

# read in data
df_all = pd.read_csv('spambase/spambase.data',header = None)
names = open('spambase/spambase.names', 'r')


# getting feature names using regex (not necessary, purely for helping with visualization)
features = []
pat = re.compile('(.*):\s*continuous\.')
for line in names:
    m = re.match(pat,line)
    if m:
        features.append(m.group(1))

# compiling into dataframe 
features.append("label")
df_all.columns = features

# shuffle dataframe and split into training & testing set
from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(df_all,test_size = 0.2,random_state=SEED)
X_train,Y_train = df_train.iloc[:, :-1],df_train.iloc[:,-1]
X_test,Y_test =  df_test.iloc[:, :-1],df_test.iloc[:,-1]

# %% DECISION TREE TESTING

from sklearn.tree import DecisionTreeClassifier

# Testing hyperparameters on three criterion
criterion = ["Gini","Entropy","Log Loss"] 

print("Decision Tree Testing:")
# %% 1. Testing pre-pruning hyperparameter (minimum samples required at leaf)

# range of parameters
B = np.arange(1,100) 

# storing training/testinig error in a dictionary organized by split criterion names
error = {ki:np.zeros([len(B),2]) for ki in criterion}

for i,Bi in enumerate(B):
    
    # training three models of differing split criteria, varying min_samples_leaf
    DT_gini = DecisionTreeClassifier(criterion='gini',min_samples_leaf = Bi,random_state = SEED).fit(X_train,Y_train)
    DT_entropy = DecisionTreeClassifier(criterion='entropy',min_samples_leaf = Bi,random_state=SEED).fit(X_train,Y_train)
    DT_logloss =  DecisionTreeClassifier(criterion='log_loss',min_samples_leaf = Bi,random_state=SEED).fit(X_train,Y_train)
    models = [DT_gini,DT_entropy,DT_logloss]
    
    # evaluate training and testing accuracy for each model
    for j,model in enumerate(models):
        error[criterion[j]][i] = 1-model.score(X_train,Y_train),1-model.score(X_test,Y_test)
    

# %% Plotting Results

fig, axs = plt.subplots(ncols=3,figsize = (20,8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

opt_B = {} # storing optimal value of min_samples_leaf for each model

for ax,key in zip(axs,criterion):
    
    err_train = error[key].T[0]
    err_test =  error[key].T[1]
    
    ax.plot(B,err_train,'b',label="Training")
    ax.plot(B,err_test,'r',label="Test") 
    ax.set_title(key)
    
    
    ax.set_xlabel(r'Minimum Samples Per leaf ($\beta$)')
    ax.set_ylabel("Error")
    ax.set_xticks(np.arange(0,120,20))
    ax.set_ylim(-0.01,0.12)
    
    ax.grid()
    
    opt_B[key] = B[err_test == min(err_test)][0]
    
plt.legend(fontsize = 'large')
plt.savefig("figures/DT_preprune.jpeg",dpi=200)

print("Optimal pre-pruning parameters:",opt_B)


# %%  2. Testing sample size


# Increasing sample size from 1/8 original size to originial training set in increments of 10
num_samples = np.arange(len(X_train)/8,len(X_train)+10,10,dtype=int) 

# reset error dictionary
error = {ki:np.zeros([len(num_samples),2]) for ki in criterion}


for i,ni in enumerate(num_samples):

    # train only on subset of specified size (no pruning)
    DT_gini = DecisionTreeClassifier(criterion='gini',random_state=SEED).fit(X_train[:ni],Y_train[:ni])
    DT_entropy = DecisionTreeClassifier(criterion='entropy',random_state=SEED).fit(X_train[:ni],Y_train[:ni])
    DT_logloss =  DecisionTreeClassifier(criterion='log_loss',random_state=SEED).fit(X_train[:ni],Y_train[:ni])
    models = [DT_gini,DT_entropy,DT_logloss]
    
    # evaluate training and testing accuracy
    for j,model in enumerate(models):
        error[criterion[j]][i] = 1-model.score(X_train[:ni],Y_train[:ni]),1-model.score(X_test,Y_test)
        

# %% Plotting results
fig, axs = plt.subplots(ncols=3,figsize = (20,8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
for ax,key in zip(axs,criterion):
    
    err_train = error[key].T[0]
    err_test =  error[key].T[1]
    
    ax.plot(num_samples,err_train,'b',label="Training")
    ax.plot(num_samples,err_test,'r',label="Test") 
    ax.set_title(key)
    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel("Error")
    ax.set_ylim(-0.01,0.16)
    ax.grid()
    
plt.legend(fontsize = 'large')
plt.savefig("figures/DT_samplesize.jpeg",dpi=200)

# %% Repeating above analysis with pre-pruning 

# Idea: Want to show how pre-pruning effects overfitting the training data and the test error

# reset error
error = {ki:np.zeros([len(num_samples),2]) for ki in criterion}

for i,ni in enumerate(num_samples):

    
    DT_gini = DecisionTreeClassifier(criterion='gini',min_samples_leaf=opt_B['Gini'],random_state=SEED).fit(X_train[:ni],Y_train[:ni])
    DT_entropy = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=opt_B['Entropy'],random_state=SEED).fit(X_train[:ni],Y_train[:ni])
    DT_logloss =  DecisionTreeClassifier(criterion='log_loss',min_samples_leaf=opt_B['Log Loss'],random_state=SEED).fit(X_train[:ni],Y_train[:ni])
    models = [DT_gini,DT_entropy,DT_logloss]
    
    # evaluate training and testing accuracy
    for j,model in enumerate(models):
        error[criterion[j]][i] = 1-model.score(X_train[:ni],Y_train[:ni]),1-model.score(X_test,Y_test)


# %% Plotting Results

fig, axs = plt.subplots(ncols=3,figsize = (20,8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
for ax,key in zip(axs,criterion):
    
    err_train = error[key].T[0]
    err_test =  error[key].T[1]
    
    ax.plot(num_samples,err_train,'b',label="Training")
    ax.plot(num_samples,err_test,'r',label="Test") 
    ax.set_title(key)
    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel("Error")
    ax.set_ylim(-0.01,0.2)
    ax.grid()
    
plt.legend(fontsize = 'large')
plt.savefig("figures/DT_samplesize_pruning.jpeg",dpi=200)

# %% 3. Testing cost-complexity pruning (effective alpha)

# Range of effective alphas
alphas = np.linspace(0,0.035,300)

error = {ki:np.zeros([len(alphas),2]) for ki in criterion} 

for i,ai in enumerate(alphas):
    
    DT_gini = DecisionTreeClassifier(criterion='gini',ccp_alpha=ai,random_state=SEED).fit(X_train,Y_train)
    DT_entropy = DecisionTreeClassifier(criterion='entropy',ccp_alpha=ai,random_state=SEED).fit(X_train,Y_train)
    DT_logloss =  DecisionTreeClassifier(criterion='log_loss',ccp_alpha=ai,random_state=SEED).fit(X_train,Y_train)
    models = [DT_gini,DT_entropy,DT_logloss]
    
    # evaluate training and testing accuracy
    for j,model in enumerate(models):
        error[criterion[j]][i] = 1-model.score(X_train,Y_train),1-model.score(X_test,Y_test)

# %% Plotting Results

fig, axs = plt.subplots(ncols=3,figsize = (20,8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

#find optimal effective alpha for each criterion
opt_alpha = {}
    
for ax,key in zip(axs,criterion):
    
    err_train = error[key].T[0]
    err_test =  error[key].T[1]
    
    ax.plot(alphas,err_train,'b',label="Training")
    ax.plot(alphas,err_test,'r',label="Test") 
    ax.set_title(key)
    ax.set_xlabel(r'effective $\alpha$')
    ax.set_ylabel("Error")
    ax.set_ylim(-0.01,0.19)
    ax.grid()
    
    opt_alpha[key] = round(alphas[err_test == min(err_test)][0],7)

    
plt.legend(fontsize = 'large')

plt.savefig("figures/DT_ccpruning.jpeg",dpi=200)

print("Optimal Cost-Complexity Pruning Parameters:",opt_alpha)


# %% RANDOM FOREST TESTING
from sklearn.ensemble import RandomForestClassifier
print('\nRandom Forest Testing:')

# %% 1) Testing number of trees in forest

num_trees = np.arange(1,150)

# testing only with Gini split criterion as it appears to yield overall lowest error
error = np.zeros([len(num_trees),2])

for i,ni in enumerate(num_trees):
    RF = RandomForestClassifier(n_estimators = ni,random_state = SEED).fit(X_train,Y_train)
    error[i] = 1-RF.score(X_train,Y_train),1-RF.score(X_test,Y_test)



# %% Plotting Results

err_train = error.T[0]
err_test = error.T[1]

fig, ax = plt.subplots(figsize = (20,8))
ax.plot(num_trees,err_train,color = 'blue',label = "Training",zorder = -1)
ax.plot(num_trees,err_test,color = 'red',label = "Testing",zorder = -1)
ax.set_xlabel("Number of Trees")
ax.set_ylabel("Error")
ax.legend(fontsize='large')
ax.grid()
plt.title("Random Forest - Ensemble Size",fontsize = 'x-large')

plt.savefig("figures/RF_numtrees.jpeg",dpi=200)

# Finding smallest optimal number of trees 
opt_ntrees = num_trees[err_test == min(err_test)][0]
print("Optimal Number of Trees:", opt_ntrees) # will compare this to value obtained from cross validation


# %% 2) Testing number of features

d = len(X_train.columns) # total number of features
dp = round(np.sqrt(d)) # expected optimal (rule of thumb) number of features

# test error over a range of feature sizes, and compare to rule of thumb
num_features = np.arange(1,d)


error = np.zeros([len(num_features),2])

for i,di in enumerate(num_features):
    RF = RandomForestClassifier(max_features = di,random_state = SEED).fit(X_train,Y_train)
    error[i] = 1-RF.score(X_train,Y_train),1-RF.score(X_test,Y_test)


# %% Plotting 

err_train = error.T[0]
err_test = error.T[1]

fig, ax = plt.subplots(figsize = (20,8))
ax.plot(num_features,err_train,color = 'blue',label = "Training")
ax.plot(num_features,err_test,color = 'red',label = "Testing")

# Plotting special point for rule of thumb, d' = sqrt(d)
ax.scatter([dp,dp],[err_train[dp-1],err_test[dp-1]],s = 90,marker = '.',c='yellow',edgecolors = 'black',zorder = 3)

ax.set_xlabel("Number of Features $d\'$")
ax.set_ylabel("Error")
ax.legend(fontsize='large')
ax.grid()
plt.title("Random Forest - Number of Features",fontsize = 'x-large')
plt.savefig('figures/RF_numfeatures.jpeg',dpi = 200)

# Obtain optimal number of features 
opt_features = num_features[err_test==min(err_test)][0]

print("Optimal Number of Features:",opt_features)
print("Expected:",dp)


# %% 3) Testing sample size 

error = np.zeros([len(num_samples),2])

for i,ni in enumerate(num_samples):
    RF = RandomForestClassifier(n_estimators=opt_ntrees,max_features=opt_features,random_state = SEED).fit(X_train[:ni],Y_train[:ni])
    # using optimal number of estimators and features 
    error[i] = 1-RF.score(X_train[:ni],Y_train[:ni]),1-RF.score(X_test,Y_test)

# %% 


err_train = error.T[0]
err_test = error.T[1]

fig, ax = plt.subplots(figsize = (20,8))
ax.plot(num_samples,err_train,color = 'blue',label = "Training")
ax.plot(num_samples,err_test,color = 'red',label = "Testing")
ax.set_xlabel("Training Sample Size $n\'$")
ax.set_ylabel("Error")
ax.legend(fontsize = 'large')
ax.grid()
plt.title("Random Forest - Sample Size",fontsize = 'x-large')

plt.savefig('figures/RF_samplesize.jpeg',dpi = 200)

# %% ADABOOST TESTING
from sklearn.ensemble import AdaBoostClassifier

print("\nAdaBoost Testing")

# %% 1) Testing ensemble size (number of stumps)

error = np.zeros([len(num_trees),2])
for i,ni in enumerate(num_trees):
    AB = AdaBoostClassifier(n_estimators = ni,random_state = SEED).fit(X_train,Y_train)
    error[i] = 1-AB.score(X_train,Y_train),1-AB.score(X_test,Y_test)

# %% Plotting

err_train = error.T[0]
err_test = error.T[1]

fig, ax = plt.subplots(figsize = (20,8))
ax.plot(num_trees,err_train,color = 'blue',label = "Training")
ax.plot(num_trees,err_test,color = 'red',label = "Testing")
ax.set_xlabel("Number of Estimators")
ax.set_ylabel("Error")
ax.legend(fontsize = 'large')
ax.grid()
plt.title("AdaBoost - Ensemble Size",fontsize = 'x-large')
plt.savefig("figures/AB_numestimators.jpeg",dpi=200)


opt_nstumps = num_trees[err_test == min(err_test)][0]
print("Optimal Ensemble Size (depth = 1):",opt_nstumps)

# %% 2) Testing max depth

depths = np.arange(1,31)
error = np.zeros([len(depths),2])

for i,di in enumerate(depths):
    DT = DecisionTreeClassifier(max_depth=di)
    AB = AdaBoostClassifier(estimator = DT,random_state = SEED).fit(X_train,Y_train)
    error[i] = 1-AB.score(X_train,Y_train),1-AB.score(X_test,Y_test)

# %% Plotting

err_train = error.T[0]
err_test = error.T[1]

fig, ax = plt.subplots(figsize = (20,8))
ax.plot(depths,err_train,color = 'blue',label = "Training")
ax.plot(depths,err_test,color = 'red',label = "Testing")
ax.set_xlabel("Maximum Depth")
ax.set_ylabel("Error")
ax.set_xticks(depths)
ax.legend(fontsize = 'large')
ax.grid()
plt.title("AdaBoost - Max Depth")
plt.savefig("figures/AB_maxdepth.jpeg",dpi=200)

opt_maxdepth = depths[err_test == min(err_test)][0]
print("Optimal Maximum Depth:",opt_maxdepth)

# %% 3) Testing sample size
error = np.zeros([len(num_samples),2])

for i,ni in enumerate(num_samples):    
    # since we are using trees of depth 1, selecting optimal number of estimators from previous experiment
    AB = AdaBoostClassifier(n_estimators = opt_nstumps,random_state = SEED).fit(X_train[:ni],Y_train[:ni])
    error[i] = 1-AB.score(X_train[:ni],Y_train[:ni]),1-AB.score(X_test,Y_test)
    

# %% Plotting

err_train = error.T[0]
err_test = error.T[1]

fig, ax = plt.subplots(figsize = (20,8))
ax.plot(num_samples,err_train,color = 'blue',label = "Training")
ax.plot(num_samples,err_test,color = 'red',label = "Testing")
ax.set_xlabel("Sample Size")
ax.set_ylabel("Error")
ax.legend(fontsize = 'large')
ax.grid()
plt.title("AdaBoost - Sample Size")
plt.savefig("figures/AB_samplesize.jpeg",dpi=200)



# %% VALIDATION ERROR MODEL SELECTION
# Calculating validation error across varying ensemble size for Random Forest and AdaBoost
num_trees = np.arange(50,260,10) # range selected empirically over repeated trials

# storing validation error for each model
RF_err = []
AB_err = []

# Using optimal number of features rom previous analysis for Random Forest 
# using stumps for AdaBoost
for ni in num_trees:
    
    RF = RandomForestClassifier(n_estimators = ni,max_features=opt_features,random_state = SEED)
    RF_err.append(k_cross_validation(df_train,RF,5)) # k = 5 for quicker computation
    
    AB = AdaBoostClassifier(n_estimators = ni,random_state=SEED)
    AB_err.append(k_cross_validation(df_train,AB,5)) 
   
    
    
# %%  Plotting Test error & Validation Error

fig, ax = plt.subplots(figsize = (20,8))
ax.plot(num_trees,RF_err,color = 'purple',label = 'Random Forest')
ax.plot(num_trees,AB_err,color = 'green',label = 'AdaBoost')

ax.set_xlabel("Ensemble Size")
ax.set_ylabel("Validation Error")
ax.grid()
ax.legend(fontsize = 'large')
plt.title("Ensemble Model Selection - Cross Validation",fontsize = 'x-large')
plt.savefig("figures/crossvalidation.jpeg",dpi=200)

# optimal number of trees in random forest from cross validation 

print("\nOptimal Ensemble Size with Cross Validation:")
opt_ntrees_xval = num_trees[RF_err == min(RF_err)][0]
print("Random Forest:",opt_ntrees_xval)

# optimal number of stumps 
opt_nstumps_xval = num_trees[AB_err == min(AB_err)][0]
print("AdaBoost:",opt_nstumps_xval)



# %% Final model selection with test error

RF_opt = RandomForestClassifier(n_estimators=opt_ntrees_xval,max_features=opt_features,random_state=SEED).fit(X_train,Y_train)
AB_opt = AdaBoostClassifier(n_estimators=opt_nstumps_xval,random_state=SEED).fit(X_train,Y_train)

print("\nFinal Test Error of Optimized Ensemble Models:")
print("Random Forest: %.3f" % (1-RF_opt.score(X_test,Y_test)))
print("AdaBoost: %.3f" % (1-AB_opt.score(X_test,Y_test)))




