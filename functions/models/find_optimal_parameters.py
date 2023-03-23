#import pandas as pd
#import numpy as np

import csv

from os.path import exists

#from classes.Column import Column

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

#from functions.preprocess.preparing import prepare_dataset

from functions.models.create_train_predict_analyze import get_model_name
from functions.models.create_train_predict_analyze import create_model

from functions.write.write_to_file import append_list_as_row
from functions.write.write_to_file import append_dict_as_row
from functions.write.write_to_file import write_string_in_text_file

import time

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



'''Model Create-Train-Predict-Analyze Functions'''

#Extract the Classification Report's Primary metrics
def opt_params_metrics(estimator,y_test,optimum_model_y_pred):
    
    cl_report = classification_report(y_test,optimum_model_y_pred)
    
    precision_sc = precision_score(y_test,optimum_model_y_pred,pos_label='positive',average='micro')
    recall_sc = recall_score(y_test,optimum_model_y_pred)
    accur_sc = accuracy_score(y_test, optimum_model_y_pred)
    f1_sc = f1_score(y_test,optimum_model_y_pred)
    
    conf_mtrx = confusion_matrix(y_test,optimum_model_y_pred)
    
    bst_params = estimator.best_params_
    
    return cl_report,precision_sc,recall_sc,accur_sc,f1_sc,conf_mtrx,bst_params


#Find the Optimal Hyperparameters for a Model
def model_opt_params_metrics_report(X_train,X_test,y_train,y_test,method,param_grid_dict,dataset_name,rep_num,n_jobs):
    
    start = time.time()
    
    model,model_name = create_model(method,{})

    pipe = Pipeline(steps=[(method, model)], memory='tmp')

    new_param_grid_dict = {}

    for key in param_grid_dict:
        
        curr_new_key_name = ''
        curr_new_key_name = method + '__' + str(key)
        new_param_grid_dict[curr_new_key_name] = param_grid_dict[key]
    
    estimator = GridSearchCV(pipe,param_grid=new_param_grid_dict,cv=10,n_jobs=n_jobs)
    #estimator = GridSearchCV(model,param_grid=param_grid_dict,cv=10,n_jobs=-1)
    
    estimator.fit(X_train,y_train)
    optimum_model_y_pred = estimator.predict(X_test)

    cl_report,precision_sc,recall_sc,accur_sc,f1_sc,conf_mtrx,bst_params = opt_params_metrics(estimator,y_test,optimum_model_y_pred)
    
    report_text = f'\n[{rep_num}]====[Optimum {model_name} Results & Hyperparameters({dataset_name})]====\n\nParameter Grid Used:\n{param_grid_dict}\n\n--------------------[Classification Report]--------------------\n{cl_report}\n---------------------------------------------------------------\nPrecision score:{precision_sc}\nRecall score:{recall_sc}\nAccuracy score:{accur_sc}\nF1 Score:{f1_sc}\n---------------------------------------------------------------\nConfusion Matrix:\n{conf_mtrx}\n---------------------------------------------------------------\nOptimal Parameters:\n{bst_params}\n---------------------------------------------------------------\n'
    
    return bst_params,precision_sc,recall_sc,accur_sc,f1_sc,report_text,start



#Find the Optimal Hyperparameters of a Model for multi repeats
def find_model_opt_param(X_train,X_test,y_train,y_test,method,param_grid_dict,dataset_name,n_jobs):
    
    #cloud_dir = 'C:/Users/NIKOS/Dropbox'
    rep_num = 1
    model_name = get_model_name(method)
    field_names_list = ['Dataset','Precision','Recall','Accuracy','F1','Runtime']
    csv_file_path_name = 'files/csv/opt_params/' + model_name + '/' + method + '_optimal_metrics.csv'
    #csv_cloud_file_path_name = cloud_dir + '/' + method + '_optimal_metrics.csv'
    txt_file_path_name = 'files/txt/opt_params/' + model_name + '/' + method + '_optimal_metrics'
    #txt_cloud_file_path_name = cloud_dir + '/' + method + '_optimal_metrics'
    
    file_exists = exists(csv_file_path_name)
    #cloud_exist = exists(csv_cloud_file_path_name)
    
    for param in param_grid_dict:
        
        curr_param_name = ''
        curr_param_name = method + '__' + str(param)
        
        field_names_list.append(curr_param_name)
    
    if file_exists == False:
        
        append_list_as_row(csv_file_path_name,field_names_list)
        #append_list_as_row(csv_cloud_file_path_name,field_names_list)
    
    else:
        
        file = open(csv_file_path_name)
        reader = csv.reader(file)
        lines= len(list(reader))
        file.close()
        
        rep_num = lines
        
        '''
        file_cloud = open(csv_cloud_file_path_name)
        reader = csv.reader(file_cloud)
        lines= len(list(reader))
        file_cloud.close()
        
        rep_num_cloud = lines
        '''
    
    bst_params_dict,precision_sc,recall_sc,accur_sc,f1_sc,report_text,start = model_opt_params_metrics_report(X_train,X_test,y_train,y_test,method,param_grid_dict,dataset_name,rep_num,n_jobs)
    
    report_full_text,duration = write_string_in_text_file(txt_file_path_name,report_text,start)
    #write_string_in_text_file(txt_cloud_file_path_name,report_text,start)
    print(report_full_text)
    
    new_row_dict = {}
        
    new_row_dict['Dataset'] = dataset_name; new_row_dict['Precision']=precision_sc
    new_row_dict['Recall']=recall_sc; new_row_dict['Accuracy']=accur_sc
    new_row_dict['F1']=f1_sc; new_row_dict['Runtime']=duration
        
    for param_name in bst_params_dict:
            
        new_row_dict[param_name] = str(bst_params_dict[param_name])
        
    append_dict_as_row(csv_file_path_name,new_row_dict,field_names_list)
    #append_dict_as_row(csv_cloud_file_path_name,new_row_dict,field_names_list)


''''''
