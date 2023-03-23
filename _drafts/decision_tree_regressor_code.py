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

dtr_param_list = {'Default_Params':{},
                  'Optimal_Params_01':{},
                  'Optimal_Params_02':{}}

dtr_features_dict = {'All_22_Features':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25],
                     'Top_15_Features':[2,3,4,5,6,9,10,11,12,14,18,20,21,24,25],
                     'Top_10_Features':[2,3,4,5,9,12,18,20,21,25]}


'''Decision Tree Regressor'''

'''
Create Train & Test sets
Train & Test Model 
Analyze Results
'''

'''Default Model Params'''

#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'dtr',dtr_param_list['Default_Params'],
                                          'default_params','22_features',dtr_features_dict['All_22_Features'],
                                          19,42,0.2,0.7)

print('=====dtr_default_params_22_features_metrics=====')
analyze_report_metrics_df('files/csv/std_reports/Decision_Tree_Regressor/dtr_default_params_22_features_metrics.csv')
print('================================================')

#Top 15 features [1,2,3,4,8,10,12,13,15,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'dtr',dtr_param_list['Default_Params'],
                                           'default_params','15_features',dtr_features_dict['Top_15_Features'],
                                            19,42,0.2,0.7)

print('=====dtr_default_params_15_features_metrics=====')
analyze_report_metrics_df('files/csv/std_reports/Decision_Tree_Regressor/dtr_default_params_15_features_metrics.csv')
print('================================================')

#Top 10 features [2,3,4,8,13,15,20,21,22,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'dtr',dtr_param_list['Default_Params'],
                                           'default_params','10_features',dtr_features_dict['Top_10_Features'],
                                            19,42,0.2,0.7)

print('=====dtr_default_params_10_features_metrics=====')
analyze_report_metrics_df('files/csv/std_reports/Decision_Tree_Regressor/dtr_default_params_10_features_metrics.csv')
print('================================================')


'''----------------------------------------------------------------------------------------------------------------'''


'''Optimal_Selected_features_01'''
'''
model_create_train_pred_analysis(X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01,'dt',
                                 {'ccp_alpha':0.0,'class_weight':None,'criterion':'gini',
                                  'max_depth':20,'max_features':'auto','max_leaf_nodes':None,
                                  'min_impurity_decrease':0.0,'min_samples_leaf':1, 
                                  'min_samples_split':2,'min_weight_fraction_leaf':0.0, 
                                  'random_state':42,'splitter':'best'},
                                 'Optimal_Selected_features_01')
'''
'''Find Optimal Hyperparameters'''

#Tester
'''
find_model_opt_param(X_train,X_test,y_train,y_test,'dtc',
                     {'ccp_alpha':[0.0],'class_weight':[None],'criterion':['gini','entropy'],
                      'max_depth':[15,25],'max_features':['auto','sqrt'],'max_leaf_nodes':[None],
                      'min_impurity_decrease':[0.0],'min_samples_leaf':[1], 
                      'min_samples_split':[2],'min_weight_fraction_leaf':[0.0], 
                      'random_state':[42],'splitter':['best']},
                     'Tester')
'''
#Default_features
'''
find_model_opt_param(X_train,X_test,y_train,y_test,'dt',
                     {'ccp_alpha':[0.0],'class_weight':[None],'criterion':['gini','entropy'],
                      'max_depth':[15,25,35],'max_features':['auto','sqrt','log2',None],'max_leaf_nodes':[None],
                      'min_impurity_decrease':[0.0],'min_samples_leaf':[1], 
                      'min_samples_split':[2],'min_weight_fraction_leaf':[0.0], 
                      'random_state':[42],'splitter':['best','random']},
                     'Default')
'''
#All_features
'''
find_model_opt_param(X_train_all,X_test_all,y_train_all,y_test_all,'dt',
                     {'ccp_alpha':[0.0],'class_weight':[None],'criterion':['gini','entropy'],
                      'max_depth':[15,25,35],'max_features':['auto','sqrt','log2',None],'max_leaf_nodes':[None],
                      'min_impurity_decrease':[0.0],'min_samples_leaf':[1], 
                      'min_samples_split':[2],'min_weight_fraction_leaf':[0.0], 
                      'random_state':[42],'splitter':['best','random']},
                     'All_features')


'''
#Selected_features_01 [1,3,5,8,15]
'''
find_model_opt_param(X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01,'dt',
                     {'ccp_alpha':[0.0],'class_weight':[None],'criterion':['gini','entropy'],
                      'max_depth':[15,25,35],'max_features':['auto','sqrt','log2',None],'max_leaf_nodes':[None],
                      'min_impurity_decrease':[0.0],'min_samples_leaf':[1], 
                      'min_samples_split':[2],'min_weight_fraction_leaf':[0.0], 
                      'random_state':[42],'splitter':['best','random']},
                     'Selected_features_01')
'''
''''''
