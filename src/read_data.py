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
def read_data(trigger=False, domain='temporal', seed=None):
  
    """
Reads and preprocesses data for classification tasks.

    This method reads feature and label data from CSV files, processes the features to remove constant values,
    and splits the data into training and testing sets. The processed data is structured into a dictionary 
    containing relevant information for classification tasks.

    Args:
        trigger (bool): A flag indicating whether to include trigger-specific features. Defaults to False.
        domain (str): The domain of the features to read. Can be 'temporal' or 'statistical'. Defaults to 'temporal'.
        seed (int, optional): The random seed for splitting the data into training and testing sets. Defaults to None.

    Returns:
        dict: A dictionary containing the classification data, including:
            - task (str): The type of task (e.g., 'Classification').
            - name (str): The name of the dataset (e.g., 'Voice_Resonance').
            - feature_names (np.ndarray): An array of feature names.
            - target_names (pd.DataFrame): The names of the target variables.
            - n_samples (int): The number of samples in the training set.
            - n_features (int): The number of features in the training set.
            - X_train (np.ndarray): The training feature data.
            - X_test (np.ndarray): The testing feature data.
            - y_train (np.ndarray): The training target data.
            - y_test (np.ndarray): The testing target data.
            - targets (pd.DataFrame): The target variable names.
            - true_labels (None): Placeholder for true labels (not set).
            - predicted_labels (None): Placeholder for predicted labels (not set).
            - descriptions (str): Placeholder for descriptions (not set).
            - items (None): Placeholder for items (not set).
            - normalize (None): Placeholder for normalization (not set).
    """
  
    sns.set_context('talk')
    
    y = pd.read_csv('./label.csv', sep=';', header=None)
    y.columns=['Resonance']
    
    if domain == 'temporal':
        X = pd.read_csv('./temporal_features_tsfel'+'_trigger_'+str(trigger)+'.csv', header=None)
        
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
   """
Generate Latin Hypercube Samples.

    This method generates Latin Hypercube samples for a given range defined by 
    `xmin` and `xmax`. It creates a sample of size `nsample` for each variable 
    defined in `xmin` and `xmax`.

    Args:
        xmin (list or array-like): The minimum values for each variable. 
                                    Must be of the same length as `xmax`.
        xmax (list or array-like): The maximum values for each variable. 
                                    Must be of the same length as `xmin`.
        nsample (int): The number of samples to generate.

    Returns:
        None: This function does not return a value. It modifies the state 
              of the object or performs an action (e.g., storing the samples).
    """
   nvar=len(xmin); ran=np.random.rand(nsample,nvar); s=np.zeros((nsample,nvar));
   for j in range(nvar):
       idx=np.random.permutation(nsample)
       P =(idx.T-ran[:,j])/nsample
       s[:,j] = xmin[j] + P*(xmax[j]-xmin[j]);
       
   return s


