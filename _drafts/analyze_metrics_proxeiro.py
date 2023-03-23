import pandas as pd
#import numpy as np

#import csv

from os.path import exists

from functions.models.create_train_predict_analyze import get_model_name

from functions.write.write_to_file import append_list_as_row
from functions.write.write_to_file import append_dict_as_row


#Appends total metrics of models 
#for a certain method(algorithm), hyperparameters and number of features to a csv file
def dataset_model_metrics_total_to_csv(method,m_param_type_name,features_num_name,dataset_name):
    
    field_names_list = ['Dataset',
                        'Precision_mean','Precision_std','Precision_min','Precision_max',
                        'Recall_mean','Recall_std','Recall_min','Recall_max',
                        'Accuracy_mean','Accuracy_std','Accuracy_min','Accuracy_max',
                        'F1_mean','F1_std','F1_min','F1_max',
                        'ROC_AUC_mean','ROC_AUC_std','ROC_AUC_min','ROC_AUC_max',
                        'P_R_AUC_mean','P_R_AUC_std','P_R_AUC_min','P_R_AUC_max',
                        'Runtime(seconds)_mean','Runtime(seconds)_std','Runtime(seconds)_min','Runtime(seconds)_max']
    
    metrics_list=['Precision','Recall','Accuracy','F1','ROC_AUC','P_R_AUC','Runtime(seconds)']
    
    dataset_type_chuncks = dataset_name.split('_',1)
    dataset_type = dataset_type_chuncks[0]
    
    param_type_chunks = m_param_type_name.split('_',1)
    param_type = param_type_chunks[0]
    
    feats_num_chunks = features_num_name.split('_',1)
    feats_num = feats_num_chunks[0]
    
    dataset_name_final = dataset_name + '_' + param_type + '_' + feats_num
    
    model_name = get_model_name(method)
    
    csv_filepath = 'files/csv/std_reports/' + model_name + '/' + method + '_' + m_param_type_name + '_' + features_num_name + '_metrics.csv'
    #print(csv_filepath)
    df =  pd.read_csv(csv_filepath,header=0)
    data = df[df['Dataset'] == dataset_name]
    
    csv_new_filepath = 'files/csv/std_reports/' + model_name + '/_' + method + '_' + dataset_type + '_total_metrics.csv'
    
    file_exists = exists(csv_new_filepath)
    
    if file_exists == False:

        append_list_as_row(csv_new_filepath,field_names_list)
    
    new_row_dict = {}
    new_row_dict['Dataset'] = dataset_name_final
    
    for metric in metrics_list:
        
        new_row_dict[str(metric + '_mean')] = round((data[metric].mean()),5)
        new_row_dict[str(metric + '_std')] = round((data[metric].std()),5)
        new_row_dict[str(metric + '_min')] = round((data[metric].min()),5)
        new_row_dict[str(metric + '_max')] = round((data[metric].max()),5)
    
    row_exists_counter = row_exists_check(new_row_dict,csv_new_filepath)
    #print(row_exists_counter)
    
    if (row_exists_counter == 0):
        append_dict_as_row(csv_new_filepath,new_row_dict,field_names_list)
        #print(new_row_dict)


#Appends total metrics of models 
#for a certain method(algorithm), hyperparameters, 
#according to primary(number of features) & secondary factors (param_type),
#to a csv file
def dataset_model_metrics_total_to_csv_new(method,dataset_name,factor_1_title,factor_1_name,factor_2_name_list):
    
    field_names_list = ['Dataset',
                        'Precision_mean','Precision_std','Precision_min','Precision_max',
                        'Recall_mean','Recall_std','Recall_min','Recall_max',
                        'Accuracy_mean','Accuracy_std','Accuracy_min','Accuracy_max',
                        'F1_mean','F1_std','F1_min','F1_max',
                        'ROC_AUC_mean','ROC_AUC_std','ROC_AUC_min','ROC_AUC_max',
                        'P_R_AUC_mean','P_R_AUC_std','P_R_AUC_min','P_R_AUC_max',
                        'Runtime(seconds)_mean','Runtime(seconds)_std','Runtime(seconds)_min','Runtime(seconds)_max']
    
    metrics_list=['Precision','Recall','Accuracy','F1','ROC_AUC','P_R_AUC','Runtime(seconds)']
    
    model_name = get_model_name(method)
    
    data_total = pd.DataFrame()
    
    for factor_2_name in factor_2_name_list:
        
        csv_filepath = 'files/csv/std_reports/' + model_name + '/' + method + '_' + factor_2_name + '_' + factor_1_name + '_metrics.csv'
        #print(csv_filepath)
        df =  pd.read_csv(csv_filepath,header=0)
        data = df[df['Dataset'] == dataset_name]
        data_total = data_total.append(data, ignore_index = True)
        
    
    dataset_type_chuncks = dataset_name.split('_',1)
    dataset_type = dataset_type_chuncks[0]
    
    feats_num_chunks = factor_1_name.split('_',1)
    feats_num = feats_num_chunks[0]
    
    dataset_name_final = dataset_name + '_' + feats_num
    
    csv_new_filepath = 'files/csv/std_reports/' + model_name + '/' + factor_1_title + '_pr/' + method + '_' + dataset_type + '__total_metrics.csv'
    
    file_exists = exists(csv_new_filepath)
    
    if file_exists == False:

        append_list_as_row(csv_new_filepath,field_names_list)
    
    new_row_dict = {}
    new_row_dict['Dataset'] = dataset_name_final
    
    for metric in metrics_list:
        
        new_row_dict[str(metric + '_mean')] = round((data_total[metric].mean()),5)
        new_row_dict[str(metric + '_std')] = round((data_total[metric].std()),5)
        new_row_dict[str(metric + '_min')] = round((data_total[metric].min()),5)
        new_row_dict[str(metric + '_max')] = round((data_total[metric].max()),5)
    
    row_exists_counter = row_exists_check(new_row_dict,csv_new_filepath)
    #print(row_exists_counter)
    
    if (row_exists_counter == 0):
        append_dict_as_row(csv_new_filepath,new_row_dict,field_names_list)
        #print(new_row_dict)


#Checks if a row's values exists before appending it
def row_exists_check(new_row_dict,csv_new_filepath):
    
    df_new = pd.read_csv(csv_new_filepath,header=0)
    
    row_exists_counter = 0
    
    new_row_list =[]
    
    for key in new_row_dict:
        
        new_row_list.append(new_row_dict[key])
    
    for index, row in df_new.iterrows():
        
        df_new_row_list = []
        
        row_dict = row.to_dict()
        
        for k in row_dict:
            
            df_new_row_list.append(row[k])
        
        row_exists = (df_new_row_list == new_row_list)
        
        if (row_exists == True):
            
            row_exists_counter = row_exists_counter + 1
        
    return row_exists_counter


#Appends total metrics of models 
#for different methods(algorithm), hyperparameters and number of features to a csv file
def multi_dataset_model_metrics_total_to_csv(method_list,m_param_type_name_list,dataset_name_list,features_num_name_list):
    
    for method in method_list:
        
        for m_param_type_name in m_param_type_name_list:
            
            for dataset_name in dataset_name_list :
                
                for features_num_name in features_num_name_list:
                
                    dataset_model_metrics_total_to_csv(method,m_param_type_name,features_num_name,dataset_name)


#Appends total metrics of models 
#for different methods(algorithm), hyperparameters, 
#according to primary(number of features) & secondary factors (param_type),
#to a csv file
def multi_dataset_model_metrics_total_to_csv_new(method_list,dataset_name_list,factor_1_title,factor_1_name_list,factor_2_name_list):
    
    for method in method_list:
        
        for dataset_name in dataset_name_list :
            
            for factor_1_name in factor_1_name_list:
                
                dataset_model_metrics_total_to_csv_new(method,dataset_name,factor_1_title,factor_1_name,factor_2_name_list)


#Sorts multi df rows according to certain characteristics' values
def multi_df_sort_values(method,dataset_type_list,factor_1_title,characteristics_list):
    
    model_name = get_model_name(method)
    
    for dataset_type in dataset_type_list:
        
        csv_filepath = 'files/csv/std_reports/' + model_name + '/' + factor_1_title + '_pr/' + method + '_' + dataset_type + '__total_metrics.csv'
        
        df = pd.read_csv(csv_filepath,header=0)
        
        num = 1
        
        for characteristic in characteristics_list:
            
            df = df.sort_values(by=[characteristic], ascending=False)
            
            csv_new_filepath = 'files/csv/std_reports/' + model_name + '/' + factor_1_title + '_pr/' + method + '_' + dataset_type + '_' + str(num) + '_' + characteristic + '.csv'
            
            df.to_csv(csv_new_filepath, index = False)
            
            num = num + 1
            #print(df)


''''''''
