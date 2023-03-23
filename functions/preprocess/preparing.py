import pandas as pd
#import numpy as np

#from classes.Column import Column

import time
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

'''Anaconda commands in conda promt''''''
   
conda create -n py37 python=3.7  #for version 3.7
conda activate py37
conda install pip
conda install wheel
conda install pandas
conda install -c conda-forge imbalanced-learn
conda install matplotlib
pip install matplotlib

Confusion Matrix Visualization:
conda install seaborn
pip install seaborn

Visualizing Decision Trees:
pip install graphviz
pip install pydotplus

Installing XGBoost Regression:
pip install xgboost

'''



'''Cleaning Dataset Functions'''

def df_preprocessing(rel_path,df_file_name_common,df_prev_file_path_name,file_extension,df_type,scaler_name,mm_param):
    
    dataframe = pd.read_csv(df_prev_file_path_name)
    
    num_scaler,scaler_name = scaling(scaler_name,mm_param)
    
    dataframe[['AGE']] = num_scaler.fit_transform(dataframe[['AGE']])
    dataframe[['DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION']] = num_scaler.fit_transform(dataframe[['DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION']])
    
    dataframe = pd.get_dummies(dataframe, columns=['INTUBATED'], drop_first=True)
    dataframe = pd.get_dummies(dataframe, columns=['PREGNANCY'], drop_first=True)
    dataframe = pd.get_dummies(dataframe, columns=['ICU'], drop_first=True)
    
    
    col_list = ['SEX','TYPE_OF_PATIENT','PNEUMONIA','DIABETIC','COPD','ASTHMA',
                   'IMMUNOSUPPRESSED','HYPERTENSION','OTHER_CHRONIC_DISEASE','CARDIOVASCULAR','OBESITY',
                   'CHRONIC_KIDNEY_FAILURE','SMOKER','CONTACT_WITH_COVID-19_CASE','SURVIVED']
    
    le = LabelEncoder()
    
    for col in col_list:
        
        dataframe[col] = le.fit_transform(dataframe[col])
    
    new_file_path_name =  rel_path + df_file_name_common + '_8_23_en_pos_' + df_type + '_valid_lb_' + scaler_name + file_extension
    
    dataframe.to_csv(new_file_path_name,index = False)
    
    #dataset = dataframe.values
    
    return dataframe,new_file_path_name


#Data Scaling (StandardScaler, MinMaxScaler or none)
def scaling(scaler_name,mm_param):
    
    if scaler_name != 'none':
        
        if scaler_name == 'std':
            num_scaler = StandardScaler()
    
        elif scaler_name == 'mm':
            num_scaler = MinMaxScaler(feature_range=(0,mm_param))
            num_scaler_name = scaler_name + '_0-' + str(mm_param)
        
    else:
        
        num_scaler = LabelEncoder()
    
    return num_scaler,num_scaler_name


#Create the Train & Test sets fron a dataframe/dataset
def prepare_dataset(df_file_path,x_column_list,y_column_list,rand_state,df_frac,train_sz,over_s_val,under_s_val,dataset_name):
    
    start = time.time()
    
    dataframe = pd.read_csv(df_file_path)
    
    dataframe = dataframe.sample(frac = df_frac)
    
    dataset = dataframe.values
    
    X = dataset[:,x_column_list]
    #X = X.astype('int')
    
    y = dataset[:,y_column_list]
    y = y.astype('int')
    
    X,y = over_under_sampling(over_s_val,under_s_val,X,y)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=rand_state,train_size=train_sz)
    
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    y_train_shape = y_train.shape
    y_test_shape = y_test.shape
    
    timestamp = datetime.now()
    end = time.time()
    duration = end-start
    
    report_text = f'\n\n\n================[{dataset_name}(frac={df_frac*100}%)]================\nOverSampler Ratio: {over_s_val*100}%\nUnderSampler Ratio: {under_s_val*100}%\n\ntrain size X :{X_train_shape}\ntest size X :{X_test_shape}\ntrain size y :{y_train_shape}\ntest size y :{y_test_shape}\n\nRuntime: {duration} seconds\n\n[{timestamp}]\n========================================================\n'
    
    print(report_text)
    
    return X_train,X_test,y_train,y_test


#Data Over & Under Sampling
def over_under_sampling(over_sampling_val,under_sampling_val,X_vals,y_vals):
    
    over_sampler = SMOTE(sampling_strategy=over_sampling_val)
    
    under_sampler = RandomUnderSampler(sampling_strategy=under_sampling_val)
    
    steps = [('over_s', over_sampler), ('under_s', under_sampler)]
    
    pipeline = Pipeline(steps=steps)
    
    X_vals,y_vals = pipeline.fit_resample(X_vals,y_vals)
    
    return X_vals,y_vals


''''''
