#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:12:09 2022

@author: roshanrayudu
"""
# o - attention 1-lookback 3-basic

DATASET_PATH = 'dataset'
JSON_PATH = "dataWithoutSplit.json"
CSV_PATH = "newestofnewestdata.csv"

import json
import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas
from sklearn.preprocessing import MinMaxScaler
import graphviz
import matplotlib.pyplot as plot
from mlxtend.evaluate import bias_variance_decomp
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report as cr




data = pandas.read_csv(CSV_PATH)

plt.figure(figsize=(5,5))
plt.title('Class distribution (1: attention, 2: lookback, 3: basic)')
locs, labels = plt.xticks()
sns.countplot(data['labelNumber'])



scaler = MinMaxScaler() 
Xlist = [data['team1'],data['team2'],data['team3'],data['team4'],data['team5'],data['team6'],data['team7'],data['team8'],data['team9'],data['team10'],data['team11'],data['team12'],data['team13']]
Xlist = scaler.fit_transform(Xlist)
yTrueList = data['labelNumber']
X = np.array(Xlist)
X= X.T
Y = np.array(yTrueList)
Y = Y.T


X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, shuffle = 'true',random_state= 8)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state= 8) #0.25*8

print("Number of training examples: " + str(len(X_train)))
print("Number of testing examples: " + str(len(X_test)))
#print("Number of validation examples: " + str(len(X_val)))



def runLogisticRegression():
    print("------------LOG REG-------------------")
    #Logistic regression
    logisticRegr = LogisticRegression()
    modelLog = logisticRegr.fit(X_train, y_train)
    predictLog = modelLog.predict(X_test)
    print("%0.2f accuracy for logistic regression" % accuracy_score(y_test, predictLog) )
    print(cr(y_test,predictLog))
    cm = confusion_matrix(y_test, predictLog)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion matrix for Logistic regression ')
    plt.show()


    



def runDTreeWithoutPruning():
    print("------------DTREE-------------------") # noise in data , stronger learner
    #Dtree
    scikitDescTree = tree.DecisionTreeClassifier(criterion="gini")
    dtreemodel = scikitDescTree.fit(X_train, y_train)
    #predictsDTree = scikitDescTree.predict(X_val)
    print("Depth of the Dtree(without pruning) learned: "+str(dtreemodel.get_depth()))
    print("training accuarcy of dTREE(without pruning): "+str(dtreemodel.score(X_train, y_train)))
    print("feature Importances : ")
    print(scikitDescTree.feature_importances_)
    valscore = crossValidation(scikitDescTree)
    print("Accuracy of the Dtree(without pruning) learned: " + str(valscore))
    print("-------------------------------")
    #Speculate DTree
    dot_data = tree.export_graphviz(scikitDescTree, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render("DTree Structure")
    plot.show()
    return dtreemodel.get_depth()


def dtreeAnalysis(maxdepth):
    
    print("------------DTREE ANALYSIS-------------------")
    dtreeTrainAccuracies = []
    dtreeValAccuracies = []
    depths = []
    for depth in range(1,27):
        print(" for depth"+ str(depth))
        scikitDescTree = tree.DecisionTreeClassifier(criterion="gini",max_depth=depth)
        dtreemodel = scikitDescTree.fit(X_train, y_train)
        #predictsDTree = scikitDescTree.predict(X_val)
        trainScore = dtreemodel.score(X_train, y_train)
        print("training accuarcy of dTREE : "+str(trainScore))
        dtreeTrainAccuracies.append(trainScore*100)
        validationScore = crossValidation(dtreemodel)
        print("validation accuarcy of dTREE : "+str(validationScore))
        dtreeValAccuracies.append(validationScore*100)
        depths.append(depth)
    
    plt.figure()
    plt.plot(depths, dtreeValAccuracies , marker='o')
    plt.plot(depths, dtreeTrainAccuracies, marker='s')
    plt.xlabel('depth', fontsize= 10)
    plt.ylabel('Validation/train accuracy', fontsize= 10)
    plt.legend(['Validation accuracy', 'Training accuracy'], fontsize=10)
    plt.title('Tuning depth for dTree')
    #plt.axis([2, 25, 15, 60])
    


def knnAnalysis():
    print("------------KNN-------------------")
    #Import knearest neighbors Classifier model
    knnTrainAccuracies = []
    knnValAccuracies = []
    newK = []
    for i in range (1 ,25): #480
      #Create KNN Clas n_neighbors = i
      print("k value:" + str(i))
      knn = KNeighborsClassifier(i)
      knn.fit(X_train, y_train)
      print("training accuarcy of knnmodel : "+str(knn.score(X_train, y_train)))
      knnTrainAccuracies.append(knn.score(X_train, y_train)*100)
      #knn_y_pred = knn.predict(X_val)
      valScore = crossValidation(knn)
      print("validation accuracy scores of knn : "+ str(valScore))
      knnValAccuracies.append(valScore*100)
      newK.append(i)
     
    plt.figure()
    plt.plot(newK, knnValAccuracies , marker='o')
    plt.plot(newK, knnTrainAccuracies, marker='s')
    plt.xlabel('K values')
    plt.ylabel('Validation/train accuracy')
    plt.legend(['Validation accuracy', 'train accuracy'])
    plt.title('Tuning k for K-NN')
    
   
        
    
    

    
    
    
    
def crossValidation(model):
    scores = cross_val_score(model, X_train, y_train, cv=10)
    return scores.mean()
    
    


def runRandomTreeClassifier():
    # vary depth 1 to 26  and number of estimators 10 to 100
    print("------------Random Tree Classifier-------------------")
    
    depths=[]
    rf_trainaccuracy = []
    rf_valaccuracy = []
    for depth in range(1,28,2):
        print("depth : " + str(depth))
        depths.append(depth)
        rf_depth = RandomForestClassifier(max_depth=depth,n_estimators=100,random_state=0)
        rf_depth.fit(X_train,y_train)
        print("training accuarcy of Random Forest : "+str(rf_depth.score(X_train, y_train)))
        rf_trainaccuracy.append(rf_depth.score(X_train, y_train)*100)
        valScore = crossValidation(rf_depth)
        print("validation accuracy scores of Random Forest : "+ str(valScore))
        rf_valaccuracy.append(valScore*100)
    
    plt.figure()
    plt.plot(depths, rf_valaccuracy , marker='o')
    plt.plot(depths, rf_trainaccuracy, marker='s')
    plt.xlabel('RF depth values')
    plt.ylabel('Validation/train accuracy')
    plt.legend(['Validation accuracy', 'train accuracy'])
    plt.title('Tuning maxdepth for Random Forest')
    
# =============================================================================
#     # lets fix the depth = 6  validation accuracy scores  : 0.4008333333333334 
#     nof_trainaccuracy = []
#     nof_valaccuracy = []
#     no_of_estimators = []
#     for i in range (10 ,160,10):
#         no_of_estimators.append(i)
#         nof = RandomForestClassifier(max_depth = 6,n_estimators= i,random_state=0)
#         nof.fit(X_train,y_train)
#         print("training accuarcy of Random Forest : "+str(nof.score(X_train, y_train)))
#         nof_trainaccuracy.append(nof.score(X_train, y_train)*100)
#         valScore = crossValidation(nof)
#         print("validation accuracy scores of Random Forest : "+ str(valScore))
#         nof_valaccuracy.append(valScore*100)
#         
#     plt.figure()
#     plt.plot(no_of_estimators, nof_valaccuracy , marker='o')
#     plt.plot(no_of_estimators, nof_trainaccuracy, marker='s')
#     plt.xlabel('RF no of estimator values')
#     plt.ylabel('Validation/train accuracy')
#     plt.legend(['Validation accuracy', 'train accuracy'])
#     
# =============================================================================
    # after no of estimator = 60 almost linear let' fix no of estimators = 100
    
    # trail
    
    depth = 27 
    no_of_est = 100
    rf = RandomForestClassifier(max_depth = depth,n_estimators= no_of_est,random_state=0)
    rf.fit(X_train,y_train)
    print("training accuarcy of Random Forest trail : "+str(rf.score(X_train, y_train)))
    #nof_trainaccuracy.append(rf.score(X_train, y_train)*100)
    valScore = crossValidation(rf)
    print("validation accuracy scores of Random Forest 5trail: "+ str(valScore))
    
    
    
def letsBagKNNClassifier():
    
    print("------------Baggin KNN Classifier-------------------")
    k_s=[]
    bagofknn_trainaccuracy = []
    bagofknn_valaccuracy = []
    
    for k in range(1,25,2):
        print("k : " + str(k)) 
        k_s.append(k)
        clf = BaggingClassifier(KNN(n_neighbors= k, weights='uniform', algorithm='auto', leaf_size=10, p=1, metric='minkowski', metric_params=None, n_jobs=1))
        clf.fit(X_train, y_train)
        bagofknn_trainaccuracy.append(clf.score(X_train, y_train))
        print("training accuarcy of KNN bags : "+str(clf.score(X_train, y_train)))
        valScore = crossValidation(clf)
        print("validation accuracy scores of KNN bags : "+ str(valScore))
        bagofknn_valaccuracy.append(valScore)
    
    plt.figure()
    plt.plot(k_s, bagofknn_valaccuracy , marker='o')
    plt.plot(k_s, bagofknn_trainaccuracy, marker='s')
    plt.xlabel('Bag of KNN depth values')
    plt.ylabel('Validation/train accuracy')
    plt.legend(['Validation accuracy', 'train accuracy'])
    plt.title('Tuning n_neighbours for Bag of KNN')
    
    k=4
    clf1 = BaggingClassifier(KNN(n_neighbors= k, weights='uniform', algorithm='auto', leaf_size=10, p=1, metric='minkowski', metric_params=None, n_jobs=1))
    clf1.fit(X_train, y_train)
    print("training accuarcy of KNN trial bags : "+str(clf1.score(X_train, y_train)))
    valScore = crossValidation(clf1)
    print("validation accuracy scores of KNN trial bags : "+ str(valScore))
    

def svmClassifier():
    print("----- SVM Tuning --------")
    param_grid = { 'C':[0.1,1,100,1000],'kernel':['poly'],'degree':[1,2,3,4]}
    grid = GridSearchCV(SVC(),param_grid)
    grid.fit(X_train,y_train)
    print("training accuarcy of SVC trial : "+str(grid.score(X_train, y_train)))
    print(grid.best_params_)
    valScore = crossValidation(grid)
    print("validation accuracy scores of SVC trial  : "+ str(valScore))
    
    
    
def modelTestPerformance():
    print("------------Test Performance-------------------")
    #k value:8
    #training accuarcy of knnmodel : 0.5436111111111112
    #validation accuracy scores of knn : 0.39138888888888884
    knn = KNeighborsClassifier(8)
    knn.fit(X_train, y_train)
    knnpredictTest = knn.predict(X_test)
    KNNtest_accuracy = accuracy_score(y_test, knnpredictTest)
   
    print("------------KNN with k=8 Performance-------------------")
    print("%0.2f accuracy KNN with k=8"% KNNtest_accuracy )
    print(cr(y_test,knnpredictTest))
    
    cm1 = confusion_matrix(y_test, knnpredictTest)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm1)
    disp.plot()
    plt.title('Confusion matrix for KNN with k =8 ')
    plt.show()
    #for depth6
    #training accuarcy of dTREE : 0.4513888888888889
    #validation accuarcy of dTREE : 0.38027777777777777
    pruneAt = 7
    scikitDescTree = tree.DecisionTreeClassifier(criterion="gini",max_depth= pruneAt)
    dtreemodel = scikitDescTree.fit(X_train, y_train)
    dtreePredictTest = dtreemodel.predict(X_test)
    dtreetest_accuracy = accuracy_score(y_test, dtreePredictTest)
    
    print("------------ Dtree with depth=7 Performance-------------------")
    print("%0.2f accuracy for dTree with depth =  7" % dtreetest_accuracy )
    print(cr(y_test,dtreePredictTest))
    
    cm2 = confusion_matrix(y_test, dtreePredictTest)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm2)
    disp.plot()
    plt.title('Confusion matrix for Dtree with depth = 7')
    plt.show()
    #depth : 27
    #training accuarcy of Random Forest : 0.9888888888888889
    #validation accuracy scores of Random Forest : 0.40138888888888885
    depth = 10
    no_of_est = 100
    rf = RandomForestClassifier(max_depth = depth,n_estimators= no_of_est,random_state=0)
    rf.fit(X_train,y_train)
    rfpredicty = rf.predict(X_test)
    rftestaccuracy = accuracy_score(y_test,rfpredicty)
    
    print("------------ Random Forest with depth=27 Performance-------------------")
    print("%0.2f accuracy for RF with depth = 27 " % rftestaccuracy )
    print(cr(y_test,rfpredicty))
 
    cm3 = confusion_matrix(y_test, rfpredicty)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm3)
    disp.plot()
    plt.title('Confusion matrix for Random Forest with depth = 27')
    plt.show()
    #k : 3
    #training accuarcy of KNN bags : 0.7277777777777777
    #validation accuracy scores of KNN bags : 0.41083333333333333
    
    k=3
    clf1 = BaggingClassifier(KNN(n_neighbors= k, weights='uniform', algorithm='auto', leaf_size=10, p=1, metric='minkowski', metric_params=None, n_jobs=1))
    clf1.fit(X_train, y_train)
    kPredictY = clf1.predict(X_test)
    knnAccuracy = accuracy_score(y_test,kPredictY)
    
    print("------------ Bag of KNN Forest with k=3 Performance-------------------")
    print("%0.2f accuracy for knn bag k =3" % knnAccuracy )
    print(cr(y_test,kPredictY))
    
    cm4 = confusion_matrix(y_test, kPredictY)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm4)
    disp.plot()
    plt.title('Confusion matrix for bag of KNN with k = 3')
    plt.show()
    
    
    


if __name__ == "__main__":
    

    runLogisticRegression()
    depth = runDTreeWithoutPruning()
    dtreeAnalysis(depth)
    knnAnalysis()
    runRandomTreeClassifier()
    letsBagKNNClassifier()
    modelTestPerformance()
    svmClassifier()
    #lstmNN()
    
   
    

