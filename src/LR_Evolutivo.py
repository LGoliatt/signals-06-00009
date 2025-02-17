# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 18:51:24 2024

@author: tbora
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-    
import time
import numpy as np
# import glob as gl
import pylab as pl
import pandas as pd
import os
from scipy.optimize import differential_evolution as de
# from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler

# from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer

from sklearn.metrics import ConfusionMatrixDisplay

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from read_data import *

basename='resonance_'
pd.options.display.float_format = '{:.3f}'.format

strategy_list=[
             'best1bin',
             'best1exp',
             'rand1exp',
             'randtobest1exp',
             'currenttobest1exp',
             'best2exp',
             'rand2exp',
             'randtobest1bin',
             'currenttobest1bin',
             'best2bin',
             'rand2bin',
             'rand1bin', 
         ]

n_runs=50
for run in range(0, n_runs):
    
    random_seed=run+10
    #-------------------------------------------------------------------------------
    datasets = [
                 read_data(domain = 'temporal', trigger=False, seed = random_seed),
               ]  
    #-------------------------------------------------------------------------------
    
    
    for dataset in datasets:
        dr=dataset['name'].replace(' ','_').replace("'","").lower()
        path='./pkl_'+dr+'/'
        # os.system('mkdir '+path.replace("-","_").lower())
        
        for tk, tn in enumerate(dataset['target_names']):
            #print (tk, tn)
            dataset_name = dataset['name']
            target                          = dataset['target_names'][tk]
            y_train, y_test                 = dataset['y_train'][tk], dataset['y_test'][tk]
            dataset_name, X_train, X_test   = dataset['name'], dataset['X_train'], dataset['X_test']
            n_samples_train, n_features     = dataset['n_samples'], dataset['n_features']
            task                            = dataset['task']
            n_samples_test                  = len(y_test)
            
            s=''+'\n'
            s+='='*80+'\n'
            s+='Dataset                    : '+dataset_name+' -- '+target+'\n'
            s+='Number of training samples : '+str(n_samples_train) +'\n'
            s+='Number of testing  samples : '+str(n_samples_test) +'\n'
            s+='Number of features         : '+str(n_features)+'\n'
            s+='Task                       : '+str(dataset['task'])+'\n'
            s+='='*80
            s+='\n'            
            
            #------------------------------------------------------------------
            lb  = [1.00]*n_features  + [ 1e-4,        0,     1e0,]
            ub  = [1.00]*n_features  + [ 1e+3,        1,     1e4, ]
            #------------------------------------------------------------------ 
            suffix='-FS' if np.mean(lb[:n_features])==0 else ''
                
                
            feature_names = dataset['feature_names']
            samples = str(n_samples_train)+'-'+str(n_samples_test)

           
            
            for strategy in strategy_list[:1]:
                
                args=(X_train, y_train, random_seed)
                                
                
                def objective_function(x,*args):
                    X,y,random_seed = args
                    n_samples, n_features=X.shape
                    ft = [ i>0.5 for i in  x[:n_features] ]
                    if sum(ft)==0:
                        return 1e12                                  

                    model = LogisticRegression(penalty='l1' if x[-2]<0.5 else 'l2',
                                               solver='sag',
                                               max_iter=1000,
                                               random_state=random_seed,
                                               C = x[-3])                    
                    try:
                        r= -cross_val_score(model,X[0:,ft], y[0:], cv=5,scoring=make_scorer(f1_score, average='weighted', greater_is_better=True),n_jobs=-1).mean()
                    except:
                        r=1e2
                        
                    return(r)
                
                #--
                start = time.process_time()
                np.random.seed(random_seed)
                init=lhsu(lb,ub,20)            
                res = de(objective_function, tuple(zip(lb,ub)), args=args,
                             strategy=strategy,
                             init=init, maxiter=30, tol=1e-5,  
                             mutation=0.8,  recombination=0.9, 
                             disp=True, polish=False,
                             seed=random_seed)
                
                end = time.process_time()
                print(end-start)
                #--
                z=res['x']
                ft = [ i>0.5 for i in  z[:n_features] ]
                model = LogisticRegression(penalty= 'l1' if z[-2]<0.5 else 'l2', 
                                           solver='liblinear',
                                               random_state=random_seed,
                                               C = z[-3]) 

                model.fit(X_train[0:,ft], y_train[0:])
                y_train_pred=model.predict(X_train[0:,ft])
                y_pred=model.predict(X_test[0:,ft])
                print(classification_report(y_test, y_pred))
                
                #%%
                y_test_pred = np.array(y_pred)
                ConfusionMatrixDisplay.from_estimator(model, X_test[0:,ft], y_test[0:],
                                  cmap=pl.cm.Blues,
                                  normalize='true')
                pl.title(dataset_name)
                pl.show()
#%%
                model_name = model.__class__.__name__+suffix
                l={
                'Y_TRAIN_TRUE':y_train, 'Y_TRAIN_PRED':y_train_pred, 
                'Y_TEST_TRUE':y_test, 'Y_TEST_PRED':y_test_pred, 'RUN':run,            
                'EST_PARAMS':model.get_params(), 
                'ESTIMATOR_NAME':model_name, 
                'PARAMS':z, 'ESTIMATOR':model, 'FEATURE_NAMES':feature_names,
                'SEED':random_seed, 'DATASET_NAME':dataset_name,
                'ALGO':'DE', 'ALGO_STRATEGY':strategy,
                'ACTIVE_VAR':ft, 'ACTIVE_VAR_NAMES':feature_names[ft],  
                'CPU_TIME':end-start, 'NFE':res['nfev']
                }
                
                pk=(path+basename+'_'+("%15s"% dataset_name).rjust(12)+
                    '_run_'+str("{:02d}".format(run))+'_'+
                    '_'+model_name+'_'+
                    ("%15s"%target).rjust(12)+'.pkl')
                pk=pk.replace(' ','_').replace("'","").lower()
                pk=pk.replace('(','_').replace(")","_").lower()
                pk=pk.replace('[','_').replace("]","_").lower()
                pk=pk.replace('-','_').replace("-","_").lower()
                pd.DataFrame([l]).to_pickle(pk)
#%%
                
