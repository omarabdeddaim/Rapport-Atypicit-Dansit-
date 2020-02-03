# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 21:54:13 2020

@author: ABDEDDAIM Omar
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.svm import LinearSVC

%matplotlib inline

df1 = pd.read_csv('Social_Network_Ads.csv')


"""
Pour créer une liste des listes 

newlist = []
for x in range(10):
    innerlist = []
    for y in range(10):
        innerlist.append(y)
    newlist.append(innerlist)
    
"""
# Equilibrage des données :
X = df1.iloc[:,[2,3]]
Y = df1.iloc[:,4]
X_smt, y_smt = ADASYN(random_state=42).fit_resample(X, Y)
print('Resampled dataset shape %s' % Counter(df['Purchased'])) 

df = pd.DataFrame()
df = X_smt
df["Purchased"] = y_smt 
# séparer les données en train et test
def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test
train,V,test = train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None)
"""
séparer = np.random.rand(len(df)) < 0.8
train = df[séparer]
test = df[~séparer]
"""
test.reset_index(drop=True, inplace=True)
train.reset_index(drop=True, inplace=True)

# Définition de la fonction Gaussiance pour une DataFrame
def gaussian_Matrice(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).T
    gaussienne = np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5)
                       * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)
    return gaussienne
def gamma(x0,x1, mu, cov):
    gamma = 1 / ((2 *math.pi) ** 
                 (1/ 2) * (cov[0][0]+cov[1][1]) ** 0.5)*math.exp(-(0.5 /(cov[0][0]+cov[1][1])*((x0-mu[0])**2+(x1-mu[1])**2)))
    return gamma


# C'est une fonction que calcule Ganssianne et moyenner et convariance de tout les modèles.
def GMV(X, foostring):
    Nouv_liste = []
    mu = []
    cov = []
    gaussiane =[]
    #Transformer le dataFrame en list
    for i in range(len(set(X[foostring]))):
        liste_Singulier = []
        liste_Singulier.append(np.log(X[X[foostring] == i].iloc[:,[0,1]]))
        Nouv_liste.append(liste_Singulier) 
        # Partie pour calcule les moyennes pour les types des données 
    for i in range(len(Nouv_liste)):
        for j in range(len(Nouv_liste[i])):
            mu_Singlulier =[]
            cov_Singulier = []
            mu_Singlulier.append(np.mean(Nouv_liste[i][j], axis=0))
            mu.append(mu_Singlulier)
         # Partie pour calcule les convariances pour les types des données    
    for i in range(len(Nouv_liste)):
        for j in range(len(Nouv_liste[i])):
            cov_Singlulier =[]
            cov_Singlulier.append( np.dot((Nouv_liste[i][j] - mu[i][j]).T, Nouv_liste[i][j] - mu[i][j]) / (Nouv_liste[i][j].shape[0] - 1))
            cov.append(cov_Singlulier)  
          # Partie pour calcule les gaussiennes pour les types des données   
    for i in range(len(Nouv_liste)):
        gaussiane_Singulier = []
        for j in range(len(Nouv_liste[i])):
            gaussiane_Singulier.append(gaussian_Matrice(Nouv_liste[i][j], mu[i][j], cov[i][j]))
            gaussiane.append(gaussiane_Singulier)
    return Nouv_liste, mu, cov, gaussiane

T,mu,cov, gaussiane = GMV(train, 'Purchased')

# on Applique les donnée pour en tester le modèle 

def Test_G(X_test):
    Gaussienne_Test = []
    #Transformer le dataFrame en list
    Nouv_liste = np.log(X_test.iloc[:,[0,1]])
    
    for i in range(len(mu)):
        Gaussianne_Test_Singulier = []
        for k in range(len(Nouv_liste)):
            Gaussianne_Test_Singulier.append(gamma(Nouv_liste.iloc[k,0],Nouv_liste.iloc[k,1], mu[i][0], cov[i][0]))
        Gaussienne_Test.append(Gaussianne_Test_Singulier)
    return Nouv_liste, Gaussienne_Test

test_liste, Gaussienne_Test=Test_G(test)  
     
def Atypicite(gaussienne,gaussienne_test):
    atypicite = []       
    for i in range(len(gaussienne_test)):
        for k in range(len(gaussienne)):
            if(i==k): 
                atypicite_list = []
                for j in range(len(gaussienne_test[k])):
                    atypicite_Singulier = 0
                    for y in range(len(gaussienne[k][0])):
                        if(gaussienne[k][0][y]>gaussienne_test[k][j]):
                            atypicite_Singulier+= gaussienne[k][0][y]
                    atypicite_list.append(atypicite_Singulier)    
        atypicite.append(atypicite_list) 
    return atypicite    
atypicite =  Atypicite(gaussiane,Gaussienne_Test)

def RAD(Atypicite, Gaussienne):
    Rap_Aty_Den_lis = []
    RAD_list = []
    for i in range(min(len(Atypicite),len(Gaussienne))):
        Rad_Aty_Den = []
        for j in range(len(Atypicite[i])):
            Rad_Aty_Den.append(Atypicite[i][j]/Gaussienne[i][j])
        Rap_Aty_Den_lis.append(Rad_Aty_Den) 
    for i in range(len(Rap_Aty_Den_lis[1])):
        if(Rap_Aty_Den_lis[0][i]>Rap_Aty_Den_lis[1][i]):
            RAD_list.append(0)
        else:
            RAD_list.append(1)
    return Rap_Aty_Den_lis, RAD_list
Rap_Aty_Den_lis, RAD_list= RAD(atypicite, Gaussienne_Test)    

test['Purchased_Examen'] =  RAD_list              
C_M= confusion_matrix(test['Purchased'], test['Purchased_Examen'])
C_R= classification_report(test['Purchased'], test['Purchased_Examen'])
ACC= round(100*accuracy_score(test['Purchased'], test['Purchased_Examen']))  

       
        
        
    