import pandas as pd
#import numpy as np
import os

#import csv

from os.path import exists

from functions.models.create_train_predict_analyze import get_model_name
from functions.plot.graph_plotting import series_graph_plot_show_save


from functions.write.write_to_file import append_list_as_row
from functions.write.write_to_file import append_dict_as_row

'''
from functions.models.analyze_metrics import export_csv_filepath_list
from functions.models.analyze_metrics import edit_dataset_type_name
from functions.models.analyze_metrics import export_metrics_to_csv
from functions.models.analyze_metrics import row_exists_check
from functions.models.analyze_metrics import dataset_model_metrics_total_to_csv
from functions.models.analyze_metrics import multi_dataset_model_metrics_total_to_csv
from functions.models.analyze_metrics import multi_df_sort_values
'''



#Appends total metrics of models 
#for a certain method(algorithm), hyperparameters, 
#according to primary & secondary factors,
#to a csv file
def dataset_model_metrics_total_to_csv(method,title,dataset_name,factor_a_name,factor_b_name):
    
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
    
    dataset_type,dataset_name_final = edit_dataset_type_name(dataset_name,factor_a_name,factor_b_name)
    
    main_folder_path = 'files/csv/std_reports/' + model_name + '/'
    
    csv_filepath_list = export_csv_filepath_list(main_folder_path,model_name,factor_a_name,factor_b_name)
    
    for csv_filepath in csv_filepath_list:
        
        df =  pd.read_csv(csv_filepath,header=0)
        #data = df[df['Dataset'] == dataset_name]
        data = df[df['Dataset'].str.contains(dataset_name)]
        data_total = data_total.append(data, ignore_index = True)
    
    export_metrics_to_csv(data_total,metrics_list,main_folder_path,title,method,dataset_type,field_names_list,dataset_name_final)
        

#Returns filepath list from a folder, that contains certain phrases
def export_csv_filepath_list(main_folder_path,model_name,factor_a_name,factor_b_name):
    
    filepath_list = os.listdir(main_folder_path)
    csv_filepath_list = []
    
    for filepath in filepath_list:
        
        if factor_a_name in filepath and factor_b_name in filepath:
            
            csv_filepath_list.append(main_folder_path + filepath)
            #print(filepath)
            
    return csv_filepath_list


#Returns the edited dataset type & name
def edit_dataset_type_name(dataset_name,factor_a_name,factor_b_name):
    
    dataset_type_chuncks = dataset_name.split('_',1)
    dataset_type = dataset_type_chuncks[0]

    factor_a_chunks = factor_a_name.split('_',1)
    factor_a = factor_a_chunks[0]

    factor_b_chunks = factor_b_name.split('_',1)
    factor_b = factor_b_chunks[0]

    dataset_name_final = dataset_name + '_' + factor_a + '_' + factor_b
    
    return dataset_type,dataset_name_final


#Exports metrics to csv file
def export_metrics_to_csv(data_total,metrics_list,main_folder_path,title,method,dataset_type,field_names_list,dataset_name_final):
    
    csv_new_filepath = main_folder_path + title + '/' + method + '_' + dataset_type + '__total_metrics.csv'
        
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


#append total metrics of models 
#for a list of methods(algorithms), hyperparameters, 
#according to primary & secondary factors,
#to a csv file
def multi_dataset_model_metrics_total_to_csv(method_list,title,dataset_name_list,factor_a_name_list,factor_b_name_list):
    
    for method in method_list:
        
        for dataset_name in dataset_name_list :
            
            for factor_a_name in factor_a_name_list:
                
                for factor_b_name in factor_b_name_list:
                
                    dataset_model_metrics_total_to_csv(method,title,dataset_name,factor_a_name,factor_b_name)


#Sorts multi df rows according to certain characteristics' values
def multi_df_sort_values(method,dataset_type_list,title,characteristics_list):
    
    model_name = get_model_name(method)
    
    for dataset_type in dataset_type_list:
        
        csv_filepath = 'files/csv/std_reports/' + model_name + '/' + title + '/' + method + '_' + dataset_type + '__total_metrics.csv'
        
        df = pd.read_csv(csv_filepath,header=0)
        
        num = 1
        
        for characteristic in characteristics_list:
            
            df = df.sort_values(by=[characteristic], ascending=False)
            
            csv_new_filepath = 'files/csv/std_reports/' + model_name + '/' + title + '/' + method + '_' + dataset_type + '_' + str(num) + '_' + characteristic + '.csv'
            
            df.to_csv(csv_new_filepath, index = False)
            
            num = num + 1
            #print(df)
            
            
''''''''

'''Analyze Total Model Metrics'''

#Total Metrics according to: Preprocessing(12), Feature Number(3[22,15,12]) & Hypeparameters (3[default,opt-01,opt-02])
multi_dataset_model_metrics_total_to_csv(['lgr'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'prep_features_params',
                                         ['fc_none','fc_std',
                                          'fc_mm_0-1','fc_mm_0-10','fc_mm_0-100','fc_mm_0-1000',
                                          'lr_none','lr_std',
                                          'lr_mm_0-1','lr_mm_0-10','lr_mm_0-100','lr_mm_0-1000'],
                                         ['22_features','15_features','10_features'],
                                         ['default_params','opt-01_params','opt-02_params'])

multi_df_sort_values('lgr',['fc','lr'],'prep_features_params',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])


#Total Metrics according to: Preprocessing(12) & Feature Number(3[22,15,12])
multi_dataset_model_metrics_total_to_csv(['lgr'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'prep_features',
                                         ['fc_none','fc_std',
                                          'fc_mm_0-1','fc_mm_0-10','fc_mm_0-100','fc_mm_0-1000',
                                          'lr_none','lr_std',
                                          'lr_mm_0-1','lr_mm_0-10','lr_mm_0-100','lr_mm_0-1000'],
                                         ['22_features','15_features','10_features'],
                                         ['_'])

multi_df_sort_values('lgr',['fc','lr'],'prep_features',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])


#Total Metrics according to: Preprocessing(12) & Hypeparameters (3[default,opt-01,opt-02])
multi_dataset_model_metrics_total_to_csv(['lgr'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'prep_params',
                                         ['fc_none','fc_std',
                                          'fc_mm_0-1','fc_mm_0-10','fc_mm_0-100','fc_mm_0-1000',
                                          'lr_none','lr_std',
                                          'lr_mm_0-1','lr_mm_0-10','lr_mm_0-100','lr_mm_0-1000'],
                                         ['default_params','opt-01_params','opt-02_params'],
                                         ['_'])

multi_df_sort_values('lgr',['fc','lr'],'prep_params',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])


#Total Metrics according to: Feature Number(3[22,15,12]) & Hypeparameters (3[default,opt-01,opt-02])
multi_dataset_model_metrics_total_to_csv(['lgr'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'features_params',
                                         ['fc_','lr_'],
                                         ['22_features','15_features','10_features'],
                                         ['default_params','opt-01_params','opt-02_params'])

multi_df_sort_values('lgr',['fc','lr'],'features_params',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])


print('=====NEW Logistic Regression - Analyze Total Model Metrics [Completed]=====')



'''Feature Importance Diagrams for MLPs & KNN'''

rel_path = 'files/csv/f_importance/'
df_original = pd.read_csv(rel_path + 'MLPs.and.KNN.csv',header=0)


feat_importances_all = df_original.squeeze()


feat_importances_all = feat_importances_all.reset_index(drop=True)

#new_feat_importances_all = feat_importances_all.copy()

feat_importances_all.set_index('Feature_Name',drop=True,inplace=True)

print(feat_importances_all)

model_dict={'mlp':'Multi_Layer_Perceptrons','knc':'KNeighbors_Classifier'}

for model in model_dict:
    method = get_model_name(model)
    graph_title = method + ' All Features\' Importance Overall' 
    graph_file_path = 'files/png/f_importance/' + method + '/' + model + '_all_feats_total.png'
    series_graph_plot_show_save(feat_importances_all,graph_title,'barh',200,graph_file_path)


''''''''