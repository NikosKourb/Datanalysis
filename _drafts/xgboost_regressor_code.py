#import pandas as pd
#import numpy as np

#from classes.Column import Column

#from functions.preprocess.preparing import df_preprocessing
#from functions.preprocess.preparing import scaling
#from functions.preprocess.preparing import prepare_dataset
#from functions.preprocess.preparing import over_under_sampling
#from functions.preprocess.preparing import over_under_sampling

#from functions.models.create_train_predict_analyze import create_model
#from functions.models.create_train_predict_analyze import model_create_train_pred_analysis
from functions.models.create_train_predict_analyze import multi_df_model_create_train_pred_analysis
from functions.models.create_train_predict_analyze import analyze_report_metrics_df

#from functions.models.find_optimal_parameters import classification_report_opt_metrics
from functions.models.find_optimal_parameters import find_model_opt_param

import warnings
warnings.filterwarnings("ignore")



rel_path = 'files/csv/data/'
df_file_name_c_prefix = 'Covid19MPD_8_23_en_pos_'
df_pos_type_dict = {'fc':[0.1,0.5],'lr':[0.5,0.8]}
df_file_name_c_suffix = '_valid_lb_'
df_scaler_type_list = ['none','std','mm_0-1','mm_0-10','mm_0-100','mm_0-1000']

xbr_param_list = {'Default_Params':{},
                  'Optimal_Params_01':{},
                  'Optimal_Params_02':{}}

xbr_features_dict = {'All_22_Features':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25],
                     'Top_15_Features':[1,2,3,4,5,9,10,12,13,14,15,18,20,22,24],
                     'Top_10_Features':[1,2,3,4,5,9,13,15,20,22]}


'''XGBoost Regressor'''

'''
Create Train & Test sets
Train & Test Model 
Analyze Results
'''

'''Default Model Params'''

#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'xbr',xbr_param_list['Default_Params'],
                                          'default_params','22_features',xbr_features_dict['All_22_Features'],
                                          19,42,0.2,0.7)

print('=====xbr_default_params_22_features_metrics=====')
analyze_report_metrics_df('files/csv/std_reports/XGBoost_Regressor/xbr_default_params_22_features_metrics.csv')
print('================================================')

#Top 15 features [1,2,3,4,8,10,12,13,15,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'xbr',xbr_param_list['Default_Params'],
                                           'default_params','15_features',xbr_features_dict['Top_15_Features'],
                                            19,42,0.2,0.7)

print('=====xbr_default_params_15_features_metrics=====')
analyze_report_metrics_df('files/csv/std_reports/XGBoost_Regressor/xbr_default_params_15_features_metrics.csv')
print('================================================')

#Top 10 features [2,3,4,8,13,15,20,21,22,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'xbr',xbr_param_list['Default_Params'],
                                           'default_params','10_features',xbr_features_dict['Top_10_Features'],
                                            19,42,0.2,0.7)

print('=====xbr_default_params_10_features_metrics=====')
analyze_report_metrics_df('files/csv/std_reports/XGBoost_Regressor/xbr_default_params_10_features_metrics.csv')
print('================================================')


'''----------------------------------------------------------------------------------------------------------------'''