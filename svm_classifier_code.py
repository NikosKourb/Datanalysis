#import pandas as pd
#import numpy as np

#from classes.Column import Column

from functions.analyzing import analyze_report_metrics_df

#from functions.preprocess.preparing import df_preprocessing
#from functions.preprocess.preparing import scaling
from functions.preprocess.preparing import prepare_dataset
#from functions.preprocess.preparing import over_under_sampling
#from functions.preprocess.preparing import over_under_sampling

#from functions.models.create_train_predict_analyze import get_model_name
#from functions.models.create_train_predict_analyze import create_model
#from functions.models.create_train_predict_analyze import primary_report_metrics
#from functions.models.create_train_predict_analyze import secondary_report_metrics
#from functions.models.create_train_predict_analyze import model_create_train_pred_analysis
from functions.models.create_train_predict_analyze import multi_df_model_create_train_pred_analysis

#from functions.models.analyze_metrics import export_csv_filepath_list
#from functions.models.analyze_metrics import edit_dataset_type_name
#from functions.models.analyze_metrics import export_metrics_to_csv
#from functions.models.analyze_metrics import row_exists_check
#from functions.models.analyze_metrics import dataset_model_metrics_total_to_csv
from functions.models.analyze_metrics import multi_dataset_model_metrics_total_to_csv
from functions.models.analyze_metrics import multi_df_sort_values

#from functions.models.find_optimal_parameters import opt_params_metrics
#from functions.models.find_optimal_parameters import model_opt_params_metrics_report
from functions.models.find_optimal_parameters import find_model_opt_param

import warnings
warnings.filterwarnings("ignore")



svc_param_dict = {'Default_Params':{'probability':True},
                  'Optimal_Params_01':{},
                  'Optimal_Params_02':{}}

svc_features_dict = {'All_22_Features':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25],
                     'Top_15_Features':[2,3,4,5,8,9,12,13,15,18,20,21,22,24,25],
                     'Top_10_Features':[2,3,4,13,18,20,21,22,24,25]}

'''
rel_path = 'files/csv/data/'
df_file_name_c_prefix = 'Covid19MPD_8_23_en_pos_'
df_pos_type_dict = {'fc':[0.1,0.5],'lr':[0.5,0.8]}
df_file_name_c_suffix = '_valid_lb_'
df_scaler_type_list = ['none','std','mm_0-1','mm_0-10','mm_0-100','mm_0-1000']
'''

df_file_dict = {'Df_01':['files/csv/data/Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv','fc_valid_lb_none',[0.1,0.5]],
                'Df_02':['files/csv/data/Covid19MPD_8_23_en_pos_fc_valid_lb_std.csv','fc_valid_lb_std',[0.1,0.5]],
                'Df_03':['files/csv/data/Covid19MPD_8_23_en_pos_fc_valid_lb_mm_0-1.csv','fc_valid_lb_mm_0-1',[0.1,0.5]],
                'Df_04':['files/csv/data/Covid19MPD_8_23_en_pos_fc_valid_lb_mm_0-10.csv','fc_valid_lb_mm_0-10',[0.1,0.5]],
                'Df_05':['files/csv/data/Covid19MPD_8_23_en_pos_fc_valid_lb_mm_0-100.csv','fc_valid_lb_mm_0-100',[0.1,0.5]],
                'Df_06':['files/csv/data/Covid19MPD_8_23_en_pos_fc_valid_lb_mm_0-1000.csv','fc_valid_lb_mm_0-1000',[0.1,0.5]],
                'Df_07':['files/csv/data/Covid19MPD_8_23_en_pos_lr_valid_lb_none.csv','lr_valid_lb_none',[0.5,0.8]],
                'Df_08':['files/csv/data/Covid19MPD_8_23_en_pos_lr_valid_lb_std.csv','lr_valid_lb_std',[0.5,0.8]],
                'Df_09':['files/csv/data/Covid19MPD_8_23_en_pos_lr_valid_lb_mm_0-1.csv','lr_valid_lb_mm_0-1',[0.5,0.8]],
                'Df_10':['files/csv/data/Covid19MPD_8_23_en_pos_lr_valid_lb_mm_0-10.csv','lr_valid_lb_mm_0-10',[0.5,0.8]],
                'Df_11':['files/csv/data/Covid19MPD_8_23_en_pos_lr_valid_lb_mm_0-100.csv','lr_valid_lb_mm_0-100',[0.5,0.8]],
                'Df_12':['files/csv/data/Covid19MPD_8_23_en_pos_lr_valid_lb_mm_0-1000.csv','lr_valid_lb_mm_0-1000',[0.5,0.8]]}



'''Support Vector Machines Classifier'''

'''
Create Train & Test sets
Train & Test Model 
Analyze Results
'''


'''Default Model Params'''
'''
#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(df_file_dict,'svc','default_params','22_features',
                                          svc_param_dict['Default_Params'],
                                          svc_features_dict['All_22_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====svc_default_params_22_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Support_Vector_Classifier/svc_default_params_22_features_metrics.csv')
print('================================================')

'''
#Top 15 features [2,3,4,5,8,9,12,13,15,18,20,21,22,24,25]
multi_df_model_create_train_pred_analysis(df_file_dict,'svc','default_params','15_features',
                                          svc_param_dict['Default_Params'],
                                          svc_features_dict['Top_15_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====svc_default_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Support_Vector_Classifier/svc_default_params_15_features_metrics.csv')
print('================================================')

'''
#Top 10 features [2,3,4,13,18,20,21,22,24,25]
multi_df_model_create_train_pred_analysis(df_file_dict,'svc','default_params','10_features',
                                          svc_param_dict['Default_Params'],
                                          svc_features_dict['Top_10_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====svc_default_params_10_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Support_Vector_Classifier/svc_default_params_10_features_metrics.csv')
print('================================================')



'''Optimal Model Params 01'''
'''
#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(df_file_dict,'svc','opt-01_params','22_features',
                                          svc_param_dict['Optimal_Params_01'],
                                          svc_features_dict['All_22_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====svc_opt-01_params_22_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Support_Vector_Classifier/svc_opt-01_params_22_features_metrics.csv')
print('================================================')

'''
#Top 15 features [2,3,4,5,8,9,12,13,15,18,20,21,22,24,25]
multi_df_model_create_train_pred_analysis(df_file_dict,'svc','opt-01_params','15_features',
                                          svc_param_dict['Optimal_Params_01'],
                                          svc_features_dict['Top_15_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====svc_opt-01_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Support_Vector_Classifier/svc_opt-01_params_15_features_metrics.csv')
print('================================================')

'''
#Top 10 features [2,3,4,13,18,20,21,22,24,25]
multi_df_model_create_train_pred_analysis(df_file_dict,'svc','opt-01_params','10_features',
                                          svc_param_dict['Optimal_Params_01'],
                                          svc_features_dict['Top_10_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====svc_opt-01_params_10_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Support_Vector_Classifier/svc_opt-01_params_10_features_metrics.csv')
print('================================================')



'''Optimal Model Params 02'''

#Tester
'''
#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(df_file_dict,'svc','opt-02_params','22_features',
                                          svc_param_dict['Optimal_Params_02'],
                                          svc_features_dict['All_22_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====svc_opt-02_params_22_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Support_Vector_Classifier/svc_opt-02_params_22_features_metrics.csv')
print('================================================')

'''
#Top 15 features [2,3,4,5,8,9,12,13,15,18,20,21,22,24,25]
multi_df_model_create_train_pred_analysis(df_file_dict,'svc','opt-02_params','15_features',
                                          svc_param_dict['Optimal_Params_02'],
                                          svc_features_dict['Top_15_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====svc_opt-02_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Support_Vector_Classifier/svc_opt-02_params_15_features_metrics.csv')
print('================================================')

'''
#Top 10 features [2,3,4,13,18,20,21,22,24,25]
multi_df_model_create_train_pred_analysis(df_file_dict,'svc','opt-02_params','10_features',
                                          svc_param_dict['Optimal_Params_02'],
                                          svc_features_dict['Top_10_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====svc_opt-02_params_10_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Support_Vector_Classifier/svc_opt-02_params_10_features_metrics.csv')
print('================================================')

print('=====Support Vector Machines Classifier - Create Train & Test sets| Train & Test Model | Analyze Results [Completed]=====')

'''----------------------------------------------------------------------------------------------------------------'''



'''Analyze Total Model Metrics'''

#Total Metrics according to: Preprocessing(12), Feature Number(3[22,15,12]) & Hypeparameters (3[default,opt-01,opt-02])
multi_dataset_model_metrics_total_to_csv(['svc'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'prep_features_params',
                                         ['fc_none','fc_std',
                                          'fc_mm_0-1','fc_mm_0-10','fc_mm_0-100','fc_mm_0-1000',
                                          'lr_none','lr_std',
                                          'lr_mm_0-1','lr_mm_0-10','lr_mm_0-100','lr_mm_0-1000'],
                                         ['22_features','15_features','10_features'],
                                         ['default_params','opt-01_params','opt-02_params'])

multi_df_sort_values('svc',['fc','lr'],'prep_features_params',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])


#Total Metrics according to: Preprocessing(12) & Feature Number(3[22,15,12])
multi_dataset_model_metrics_total_to_csv(['svc'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'prep_features',
                                         ['fc_none','fc_std',
                                          'fc_mm_0-1','fc_mm_0-10','fc_mm_0-100','fc_mm_0-1000',
                                          'lr_none','lr_std',
                                          'lr_mm_0-1','lr_mm_0-10','lr_mm_0-100','lr_mm_0-1000'],
                                         ['22_features','15_features','10_features'],
                                         ['_'])

multi_df_sort_values('svc',['fc','lr'],'prep_features',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])


#Total Metrics according to: Preprocessing(12) & Hypeparameters (3[default,opt-01,opt-02])
multi_dataset_model_metrics_total_to_csv(['svc'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'prep_params',
                                         ['fc_none','fc_std',
                                          'fc_mm_0-1','fc_mm_0-10','fc_mm_0-100','fc_mm_0-1000',
                                          'lr_none','lr_std',
                                          'lr_mm_0-1','lr_mm_0-10','lr_mm_0-100','lr_mm_0-1000'],
                                         ['default_params','opt-01_params','opt-02_params'],
                                         ['_'])

multi_df_sort_values('svc',['fc','lr'],'prep_params',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])


#Total Metrics according to: Feature Number(3[22,15,12]) & Hypeparameters (3[default,opt-01,opt-02])
multi_dataset_model_metrics_total_to_csv(['svc'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'features_params',
                                         ['fc_','lr_'],
                                         ['22_features','15_features','10_features'],
                                         ['default_params','opt-01_params','opt-02_params'])

multi_df_sort_values('svc',['fc','lr'],'features_params',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])


print('=====Support Vector Machines Classifier - Analyze Total Model Metrics [Completed]=====')

'''----------------------------------------------------------------------------------------------------------------'''



'''Find Optimal Hyperparameters'''

for df_file in df_file_dict:
    
    for repeat in range(5):
        
        X_train,X_test,y_train,y_test = prepare_dataset(df_file_dict[df_file][0],
                                                        svc_features_dict['All_22_Features'],
                                                        19,42,0.2,0.7,
                                                        df_file_dict[df_file][2][0],
                                                        df_file_dict[df_file][2][1],
                                                        df_file_dict[df_file][1])
        
        
        #Tester
        find_model_opt_param(X_train,X_test,y_train,y_test,'svc',
                             {},
                             df_file_dict[df_file][1],-1)
        
        '''
        find_model_opt_param(X_train,X_test,y_train,y_test,'svc',
                             {},
                             df_file_dict[df_file][1],-1)
        '''

print('=====Support Vector Machines Classifier - Find Optimal Hyperparameters [Completed]=====')

'''----------------------------------------------------------------------------------------------------------------'''

''''''
