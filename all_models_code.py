#import pandas as pd
#import numpy as np
from numpy import nan

#from classes.Column import Column

from functions.analyzing import analyze_report_metrics_df

#from functions.preprocess.preparing import df_preprocessing
#from functions.preprocess.preparing import scaling
#from functions.preprocess.preparing import prepare_dataset
#from functions.preprocess.preparing import over_under_sampling
#from functions.preprocess.preparing import over_under_sampling

#from functions.models.create_train_predict_analyze import create_model
#from functions.models.create_train_predict_analyze import model_create_train_pred_analysis
from functions.models.create_train_predict_analyze import multi_df_model_create_train_pred_analysis

#from functions.models.find_optimal_parameters import classification_report_opt_metrics
from functions.models.find_optimal_parameters import find_model_opt_param

import warnings
warnings.filterwarnings("ignore")



rel_path = 'files/csv/data/'
df_file_name_c_prefix = 'Covid19MPD_8_23_en_pos_'
df_pos_type_dict = {'fc':[0.1,0.5],'lr':[0.5,0.8]}
df_file_name_c_suffix = '_valid_lb_'
df_scaler_type_list = ['none','std','mm_0-1','mm_0-10','mm_0-100','mm_0-1000']


'''
Create Train & Test sets
Train & Test Model 
Analyze Results
'''


'''Logistic Regression'''

lgr_param_dict = {'Default_Params':{'C':1.0,'class_weight':None,'dual':False,'fit_intercept':True,
                                    'intercept_scaling':1,'l1_ratio':None,'max_iter':100,
                                    'multi_class':'auto','penalty':'l2','random_state':42,
                                    'solver':'lbfgs','tol':0.0001,'verbose':0,'warm_start':False},
                  'Optimal_Params_01':{'C':1.0,'class_weight':None,'dual':False,'fit_intercept':True,
                                       'intercept_scaling':1,'l1_ratio':None,'max_iter':100,
                                       'multi_class':'auto','penalty':'l2','random_state':42,
                                       'solver':'lbfgs','tol':0.0001,'verbose':0,'warm_start':False},
                  'Optimal_Params_02':{'C':1.0,'class_weight':None,'dual':False,'fit_intercept':True,
                                       'intercept_scaling':1,'l1_ratio':None,'max_iter':100,
                                       'multi_class':'auto','penalty':'l2','random_state':42,
                                       'solver':'lbfgs','tol':0.0001,'verbose':0,'warm_start':False}}

lgr_features_dict = {'All_22_Features':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25],
                     'Top_15_Features':[1,2,3,4,8,10,12,13,15,20,21,22,23,24,25],
                     'Top_10_Features':[2,3,4,8,13,15,20,21,22,25]}

#Default Model Params
'''
#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'lgr',lgr_param_dict['Default_Params'],
                                          'default_params','22_features',lgr_features_dict['All_22_Features'],
                                          19,42,0.2,0.7)
'''
print('\n\n=====lgr_default_params_22_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Logistic_Regression/lgr_default_params_22_features_metrics.csv')
print('================================================\n\n')
'''
#Top 15 features [1,2,3,4,8,10,12,13,15,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'lgr',lgr_param_dict['Default_Params'],
                                          'default_params','15_features',lgr_features_dict['Top_15_Features'],
                                          19,42,0.2,0.7)
'''
print('\n\n=====lgr_default_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Logistic_Regression/lgr_default_params_15_features_metrics.csv')
print('================================================\n\n')
'''
#Top 10 features [2,3,4,8,13,15,20,21,22,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'lgr',lgr_param_dict['Default_Params'],
                                          'default_params','10_features',lgr_features_dict['Top_10_Features'],
                                          19,42,0.2,0.7)
'''
print('\n\n=====lgr_default_params_10_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Logistic_Regression/lgr_default_params_10_features_metrics.csv')
print('================================================\n\n')


'''----------------------------------------------------------------------------------------------------------------'''



'''Decision Tree Classifier'''

dtc_param_list = {'Default_Params':{'ccp_alpha':0.0,'class_weight':None,'criterion':'gini',
                                    'max_depth':None,'max_features':None,'max_leaf_nodes':None,
                                    'min_impurity_decrease':0.0,'min_samples_leaf':1, 
                                    'min_samples_split':2,'min_weight_fraction_leaf':0.0, 
                                    'random_state':42,'splitter':'best'},
                  'Optimal_Params_01':{'ccp_alpha':0.0,'class_weight':None,'criterion':'gini',
                                       'max_depth':None,'max_features':None,'max_leaf_nodes':None,
                                       'min_impurity_decrease':0.0,'min_samples_leaf':1, 
                                       'min_samples_split':2,'min_weight_fraction_leaf':0.0, 
                                       'random_state':42,'splitter':'best'},
                  'Optimal_Params_02':{'ccp_alpha':0.0,'class_weight':None,'criterion':'gini',
                                       'max_depth':None,'max_features':None,'max_leaf_nodes':None,
                                       'min_impurity_decrease':0.0,'min_samples_leaf':1, 
                                       'min_samples_split':2,'min_weight_fraction_leaf':0.0, 
                                       'random_state':42,'splitter':'best'}}

dtc_features_dict = {'All_22_Features':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25],
                     'Top_15_Features':[3,4,5,6,7,9,10,11,12,14,18,20,21,24,25],
                     'Top_10_Features':[3,4,5,12,14,9,18,20,21,25]}

#Default Model Params
'''
#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'dtc',dtc_param_list['Default_Params'],
                                          'default_params','22_features',dtc_features_dict['All_22_Features'],
                                          19,42,0.2,0.7)
'''
print('\n\n=====dtc_default_params_22_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Decision_Tree_Classifier/dtc_default_params_22_features_metrics.csv')
print('================================================\n\n')
'''
#Top 15 features [3,4,5,6,7,9,10,11,12,14,18,20,21,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'dtc',dtc_param_list['Default_Params'],
                                           'default_params','15_features',dtc_features_dict['Top_15_Features'],
                                            19,42,0.2,0.7)
'''
print('\n\n=====dtc_default_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Decision_Tree_Classifier/dtc_default_params_15_features_metrics.csv')
print('================================================\n\n')
'''
#Top 10 features [3,4,5,12,14,9,18,20,21,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'dtc',dtc_param_list['Default_Params'],
                                           'default_params','10_features',dtc_features_dict['Top_10_Features'],
                                            19,42,0.2,0.7)
'''
print('\n\n=====dtc_default_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Decision_Tree_Classifier/dtc_default_params_10_features_metrics.csv')
print('================================================\n\n')


'''----------------------------------------------------------------------------------------------------------------'''



'''Random Forest Classifier'''

rfc_param_list = {'Default_Params':{'bootstrap':True,'ccp_alpha':0.0,'class_weight':None,'criterion':'gini',
                                    'max_depth':20, 'max_features':'auto','max_leaf_nodes':None,'max_samples':None,
                                    'min_impurity_decrease':0.0,'min_samples_leaf':5,'min_samples_split':5,
                                    'min_weight_fraction_leaf':0.0,'n_estimators':100,
                                    'oob_score':False,'random_state':42,'verbose':0,'warm_start':False},
                  'Optimal_Params_01':{'bootstrap':True,'ccp_alpha':0.0,'class_weight':'balanced','criterion':'entropy',
                                       'max_depth':None, 'max_features':'sqrt','max_leaf_nodes':35,'max_samples':None,
                                       'min_impurity_decrease':0.0,'min_samples_leaf':1,'min_samples_split':2,
                                       'min_weight_fraction_leaf':0.0,'n_estimators':150,
                                       'oob_score':False,'random_state':42,'verbose':0,'warm_start':True},
                  'Optimal_Params_02':{'bootstrap':True,'ccp_alpha':0.0,'class_weight':'balanced','criterion':'entropy',
                                       'max_depth':None, 'max_features':'sqrt','max_leaf_nodes':35,'max_samples':None,
                                       'min_impurity_decrease':0.0,'min_samples_leaf':3,'min_samples_split':3,
                                       'min_weight_fraction_leaf':0.0,'n_estimators':75,
                                       'oob_score':False,'random_state':42,'verbose':0,'warm_start':True}}

rfc_features_dict = {'All_22_Features':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25],
                     'Top_15_Features':[2,3,4,5,9,12,11,13,14,15,18,20,21,24,25],
                     'Top_10_Features':[2,3,4,5,9,18,20,21,24,25]}

#Default Model Params
'''
#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'rfc',rfc_param_list['Default_Params'],
                                          'default_params','22_features',rfc_features_dict['All_22_Features'],
                                          19,42,0.2,0.7)
'''
print('\n\n=====rfc_default_params_22_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Random_Forest_Classifier/rfc_default_params_22_features_metrics.csv')
print('================================================\n\n')
'''
#Top 15 features [2,3,4,5,9,12,11,13,14,15,18,20,21,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'rfc',rfc_param_list['Default_Params'],
                                           'default_params','15_features',rfc_features_dict['Top_15_Features'],
                                            19,42,0.2,0.7)
'''
print('\n\n=====rfc_default_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Random_Forest_Classifier/rfc_default_params_15_features_metrics.csv')
print('================================================\n\n')
'''
#Top 10 features [2,3,4,5,9,18,20,21,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'rfc',rfc_param_list['Default_Params'],
                                           'default_params','10_features',rfc_features_dict['Top_10_Features'],
                                            19,42,0.2,0.7)
'''
print('\n\n=====rfc_default_params_10_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Random_Forest_Classifier/rfc_default_params_10_features_metrics.csv')
print('================================================\n\n')


'''----------------------------------------------------------------------------------------------------------------'''



'''XGBoost Classifier'''

xbc_param_list = {'Default_Params':{'objective':'binary:logistic','use_label_encoder': True,
                                    'base_score':0.5, 'booster':'gbtree','colsample_bylevel':1,
                                    'colsample_bynode':1,'colsample_bytree':1,'enable_categorical':False,
                                    'gamma':0,'gpu_id':-1,'importance_type':None,'interaction_constraints':'',
                                    'learning_rate':0.300000012,'max_delta_step':0,'max_depth':6,
                                    'min_child_weight':1,'missing':nan,'monotone_constraints':'()',
                                    'n_estimators':100,'n_jobs':8,'num_parallel_tree':1,'predictor':'auto',
                                    'random_state':42,'reg_alpha':0,'reg_lambda':1,'scale_pos_weight':1,
                                    'subsample':1,'tree_method':'exact','validate_parameters':1,
                                    'verbosity':None},
                  'Optimal_Params_01':{'objective':'binary:logistic','use_label_encoder': True,
                                       'base_score':0.5, 'booster':'gbtree','colsample_bylevel':1,
                                       'colsample_bynode':1,'colsample_bytree':1,'enable_categorical':False,
                                       'gamma':0,'gpu_id':-1,'importance_type':None,'interaction_constraints':'',
                                       'learning_rate':0.300000012,'max_delta_step':0,'max_depth':6,
                                       'min_child_weight':1,'missing':nan,'monotone_constraints':'()',
                                       'n_estimators':100,'n_jobs':8,'num_parallel_tree':1,'predictor':'auto',
                                       'random_state':42,'reg_alpha':0,'reg_lambda':1,'scale_pos_weight':1,
                                       'subsample':1,'tree_method':'exact','validate_parameters':1,
                                       'verbosity':None},
                  'Optimal_Params_02':{'objective':'binary:logistic','use_label_encoder': True,
                                       'base_score':0.5, 'booster':'gbtree','colsample_bylevel':1,
                                       'colsample_bynode':1,'colsample_bytree':1,'enable_categorical':False,
                                       'gamma':0,'gpu_id':-1,'importance_type':None,'interaction_constraints':'',
                                       'learning_rate':0.300000012,'max_delta_step':0,'max_depth':6,
                                       'min_child_weight':1,'missing':nan,'monotone_constraints':'()',
                                       'n_estimators':100,'n_jobs':8,'num_parallel_tree':1,'predictor':'auto',
                                       'random_state':42,'reg_alpha':0,'reg_lambda':1,'scale_pos_weight':1,
                                       'subsample':1,'tree_method':'exact','validate_parameters':1,
                                       'verbosity':None}}

xbc_features_dict = {'All_22_Features':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25],
                     'Top_15_Features':[1,2,3,4,5,8,9,10,12,13,14,15,20,22,24],
                     'Top_10_Features':[1,2,3,4,5,9,13,15,20,22]}

#Default Model Params
'''
#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'xbc',xbc_param_list['Default_Params'],
                                          'default_params','22_features',xbc_features_dict['All_22_Features'],
                                          19,42,0.2,0.7)
'''
print('\n\n=====xbc_default_params_22_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/XGBoost_Classifier/xbc_default_params_22_features_metrics.csv')
print('================================================\n\n')
'''
#Top 15 features [1,2,3,4,5,8,9,10,12,13,14,15,20,22,24]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'xbc',xbc_param_list['Default_Params'],
                                           'default_params','15_features',xbc_features_dict['Top_15_Features'],
                                            19,42,0.2,0.7)
'''
print('\n\n=====xbc_default_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/XGBoost_Classifier/xbc_default_params_15_features_metrics.csv')
print('================================================\n\n')
'''
#Top 10 features [1,2,3,4,5,9,13,15,20,22]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'xbc',xbc_param_list['Default_Params'],
                                           'default_params','10_features',xbc_features_dict['Top_10_Features'],
                                            19,42,0.2,0.7)
'''
print('\n\n=====xbc_default_params_10_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/XGBoost_Classifier/xbc_default_params_10_features_metrics.csv')
print('================================================\n\n')


'''----------------------------------------------------------------------------------------------------------------'''



'''Multi-Layer Perceptrons'''

mlp_param_list = {'Default_Params':{'activation':'relu','alpha':0.0001,'batch_size':'auto','beta_1':0.9,
                                    'beta_2':0.999,'early_stopping':False,'epsilon':1e-08, 
                                    'hidden_layer_sizes':(100,),'learning_rate':'constant',
                                    'learning_rate_init':0.001,'max_fun':15000,'max_iter':200,
                                    'momentum':0.9,'n_iter_no_change':10,'nesterovs_momentum':True,
                                    'power_t':0.5,'random_state':42,'shuffle':True,'solver':'adam',
                                    'tol':0.0001,'validation_fraction':0.1,'verbose':False,
                                    'warm_start':False},
                  'Optimal_Params_01':{'activation':'logistic','alpha':0.0001,'batch_size':'auto','beta_1':0.9,
                                       'beta_2':0.999,'early_stopping':False,'epsilon':1e-08, 
                                       'hidden_layer_sizes':(50,50,),'learning_rate':'adaptive',
                                       'learning_rate_init':0.001,'max_fun':15000,'max_iter':100,
                                       'momentum':0.9,'n_iter_no_change':10,'nesterovs_momentum':True,
                                       'power_t':0.5,'random_state':42,'shuffle':True,'solver':'adam',
                                       'tol':0.0001,'validation_fraction':0.1,'verbose':False,
                                       'warm_start':True},
                  'Optimal_Params_02':{'activation':'identity','alpha':0.0001,'batch_size':'auto','beta_1':0.9,
                                       'beta_2':0.999,'early_stopping':False,'epsilon':1e-08, 
                                       'hidden_layer_sizes':(100,100,),'learning_rate':'invscaling',
                                       'learning_rate_init':0.001,'max_fun':15000,'max_iter':300,
                                       'momentum':0.9,'n_iter_no_change':10,'nesterovs_momentum':True,
                                       'power_t':0.5,'random_state':42,'shuffle':True,'solver':'lbfgs',
                                       'tol':0.0001,'validation_fraction':0.1,'verbose':False,
                                       'warm_start':True}}

mlp_features_dict = {'All_22_Features':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25],
                     'Top_15_Features':[2,3,4,5,8,9,12,13,15,18,20,21,22,24,25],
                     'Top_10_Features':[2,3,4,13,18,20,21,22,24,25]}

#Default Model Params
'''
#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'mlp',mlp_param_list['Default_Params'],
                                          'default_params','22_features',mlp_features_dict['All_22_Features'],
                                          19,42,0.2,0.7)
'''
print('\n\n=====mlp_default_params_22_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Multi_Layer_Perceptrons/mlp_default_params_22_features_metrics.csv')
print('================================================\n\n')
'''
#Top 15 features [2,3,4,5,8,9,12,13,15,18,20,21,22,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'mlp',mlp_param_list['Default_Params'],
                                           'default_params','15_features',mlp_features_dict['Top_15_Features'],
                                            19,42,0.2,0.7)
'''
print('\n\n=====mlp_default_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Multi_Layer_Perceptrons/mlp_default_params_15_features_metrics.csv')
print('================================================\n\n')
'''
#Top 10 features [2,3,4,13,18,20,21,22,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'mlp',mlp_param_list['Default_Params'],
                                           'default_params','10_features',mlp_features_dict['Top_10_Features'],
                                            19,42,0.2,0.7)
'''
print('\n\n=====mlp_default_params_10_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/Multi_Layer_Perceptrons/mlp_default_params_10_features_metrics.csv')
print('================================================\n\n')



'''----------------------------------------------------------------------------------------------------------------'''



'''KNeighbors Classifier'''

knc_param_list = {'Default_Params':{'algorithm':'auto','leaf_size':30,'metric':'minkowski',
                                    'metric_params':None,'n_neighbors':5,'p':2,'weights':'uniform'},
                  'Optimal_Params_01':{'algorithm':'brute','leaf_size':30,'metric':'minkowski',
                                       'metric_params':None,'n_neighbors':2,'p':1,'weights':'distance'},
                  'Optimal_Params_02':{'algorithm':'kd_tree','leaf_size':30,'metric':'minkowski',
                                       'metric_params':None,'n_neighbors':10,'p':2,'weights':'distance'}}

knc_features_dict = {'All_22_Features':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25],
                     'Top_15_Features':[2,3,4,5,8,9,12,13,15,18,20,21,22,24,25],
                     'Top_10_Features':[2,3,4,13,18,20,21,22,24,25]}

#Default Model Params
'''
#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'knc',knc_param_list['Default_Params'],
                                          'default_params','22_features',knc_features_dict['All_22_Features'],
                                          19,42,0.2,0.7)
'''
print('\n\n=====knc_default_params_22_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/KNeighbors_Classifier/knc_default_params_22_features_metrics.csv')
print('================================================\n\n')
'''
#Top 15 features [2,3,4,5,8,9,12,13,15,18,20,21,22,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'knc',knc_param_list['Default_Params'],
                                           'default_params','15_features',knc_features_dict['Top_15_Features'],
                                            19,42,0.2,0.7)
'''
print('\n\n=====knc_default_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/KNeighbors_Classifier/knc_default_params_15_features_metrics.csv')
print('================================================\n\n')
'''
#Top 10 features [2,3,4,13,18,20,21,22,24,25]
multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,
                                          df_scaler_type_list,10,'knc',knc_param_list['Default_Params'],
                                           'default_params','10_features',knc_features_dict['Top_10_Features'],
                                            19,42,0.2,0.7)
'''
print('\n\n=====knc_default_params_10_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/KNeighbors_Classifier/knc_default_params_10_features_metrics.csv')
print('================================================\n\n')

'''----------------------------------------------------------------------------------------------------------------'''

''''''
