# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:25:30 2024

@author: tbora
"""

# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl    
from sklearn.model_selection import train_test_split

#%%
def read_data(n_mfcc=13, trigger=False, domain='temporal', seed=None):
  
    sns.set_context('talk')
    
    y = pd.read_csv('./label.csv', sep=';', header=None)
    y.columns=['Resonance']
    
    if domain == 'temporal':
        X = pd.read_csv('./temporal_features_tsfel'+'_trigger_'+str(trigger)+'.csv', header=None)
               
    elif domain == 'mfcc':
        X = pd.read_csv('./mfcc'+str(n_mfcc)+'_trigger_'+str(trigger)+'.csv', header=None)
        
    elif domain == 'statistical':
        X = pd.read_csv('./statistical_features'+'_trigger_'+str(trigger)+'.csv', header=None)
    
    df = X.copy()  

    idx = df.std(axis=0)!=0 # remove constant values
    df=df[df.columns[idx]]
    
    
    df.columns=['$X_{'+str(i+1)+'}$' for i in df.columns]
    df['Class'] = y.values

    
    
    X.columns=['X'+str(i+1) for i in X.columns] 
    variable_names = X.columns
    target_names=y.columns
     
    X = X[variable_names]
    y = y[target_names]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    
    n_samples, n_features = X_train.shape 
         
    classification_data =  {
      'task'            : 'Classification',
      'name'            : 'Voice_Resonance',
      'feature_names'   : np.array(variable_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train.values,
      'X_test'          : X_test.values,
      'y_train'         : y_train.values.reshape(1,-1),
      'y_test'          : y_test.values.reshape(1,-1),      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'normalize'       : None,
      }

    return classification_data


#%%
def lhsu(xmin,xmax,nsample):
   nvar=len(xmin); ran=np.random.rand(nsample,nvar); s=np.zeros((nsample,nvar));
   for j in range(nvar):
       idx=np.random.permutation(nsample)
       P =(idx.T-ran[:,j])/nsample
       s[:,j] = xmin[j] + P*(xmax[j]-xmin[j]);
       
   return s


