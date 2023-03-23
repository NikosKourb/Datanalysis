#import pandas as pd
#import numpy as np
from numpy import nan

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



xbc_param_dict = {'Default_Params':{'objective':'binary:logistic','use_label_encoder': True,
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
                                       'gamma':0,'gpu_id':-1,'importance_type':'gain','interaction_constraints':'',
                                       'learning_rate':0.300000012,'max_delta_step':0,'max_depth':5,
                                       'min_child_weight':1,'missing':nan,'monotone_constraints':'()',
                                       'n_estimators':100,'n_jobs':8,'num_parallel_tree':1,'predictor':'auto',
                                       'random_state':42,'reg_alpha':0,'reg_lambda':1,'scale_pos_weight':1,
                                       'subsample':1,'tree_method':'auto','validate_parameters':1,
                                       'verbosity':None},
                  'Optimal_Params_02':{'objective':'binary:logistic','use_label_encoder': True,
                                       'base_score':0.5, 'booster':'gbtree','colsample_bylevel':1,
                                       'colsample_bynode':1,'colsample_bytree':1,'enable_categorical':False,
                                       'gamma':0,'gpu_id':-1,'importance_type':'gain','interaction_constraints':'',
                                       'learning_rate':0.300000012,'max_delta_step':0,'max_depth':11,
                                       'min_child_weight':1,'missing':nan,'monotone_constraints':'(1,0)',
                                       'n_estimators':125,'n_jobs':8,'num_parallel_tree':1,'predictor':'auto',
                                       'random_state':42,'reg_alpha':0,'reg_lambda':1,'scale_pos_weight':1,
                                       'subsample':1,'tree_method':'auto','validate_parameters':1,
                                       'verbosity':None}}

xbc_features_dict = {'All_22_Features':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25],
                     'Top_15_Features':[1,2,3,4,5,8,9,10,12,13,14,15,20,22,24],
                     'Top_10_Features':[1,2,3,4,5,9,13,15,20,22]}

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



'''XGBoost Classifier'''

'''
Create Train & Test sets
Train & Test Model 
Analyze Results
'''


'''Default Model Params'''

'''
#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(df_file_dict,'xbc','default_params','22_features',
                                          xbc_param_dict['Default_Params'],
                                          xbc_features_dict['All_22_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====xbc_default_params_22_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/XGBoost_Classifier/xbc_default_params_22_features_metrics.csv')
print('================================================')

'''
#Top 15 features [1,2,3,4,5,8,9,10,12,13,14,15,20,22,24]
multi_df_model_create_train_pred_analysis(df_file_dict,'xbc','default_params','15_features',
                                          xbc_param_dict['Default_Params'],
                                          xbc_features_dict['Top_15_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====xbc_default_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/XGBoost_Classifier/xbc_default_params_15_features_metrics.csv')
print('================================================')

'''
#Top 10 features [1,2,3,4,5,9,13,15,20,22]
multi_df_model_create_train_pred_analysis(df_file_dict,'xbc','default_params','10_features',
                                          xbc_param_dict['Default_Params'],
                                          xbc_features_dict['Top_10_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====xbc_default_params_10_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/XGBoost_Classifier/xbc_default_params_10_features_metrics.csv')
print('================================================')



'''Optimal Model Params 01'''

'''
#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(df_file_dict,'xbc','opt-01_params','22_features',
                                          xbc_param_dict['Optimal_Params_01'],
                                          xbc_features_dict['All_22_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====xbc_opt-01_params_22_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/XGBoost_Classifier/xbc_opt-01_params_22_features_metrics.csv')
print('================================================')

'''
#Top 15 features [1,2,3,4,5,8,9,10,12,13,14,15,20,22,24]
multi_df_model_create_train_pred_analysis(df_file_dict,'xbc','opt-01_params','15_features',
                                          xbc_param_dict['Optimal_Params_01'],
                                          xbc_features_dict['Top_15_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====xbc_opt-01_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/XGBoost_Classifier/xbc_opt-01_params_15_features_metrics.csv')
print('================================================')

'''
#Top 10 features [1,2,3,4,5,9,13,15,20,22]
multi_df_model_create_train_pred_analysis(df_file_dict,'xbc','opt-01_params','10_features',
                                          xbc_param_dict['Optimal_Params_01'],
                                          xbc_features_dict['Top_10_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====xbc_opt-01_params_10_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/XGBoost_Classifier/xbc_opt-01_params_10_features_metrics.csv')
print('================================================')



'''Optimal Model Params 02'''

'''
#All 22 features [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,22,23,24,25]
multi_df_model_create_train_pred_analysis(df_file_dict,'xbc','opt-02_params','22_features',
                                          xbc_param_dict['Optimal_Params_02'],
                                          xbc_features_dict['All_22_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====xbc_opt-02_params_22_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/XGBoost_Classifier/xbc_opt-02_params_22_features_metrics.csv')
print('================================================')

'''
#Top 15 features [1,2,3,4,5,8,9,10,12,13,14,15,20,22,24]
multi_df_model_create_train_pred_analysis(df_file_dict,'xbc','opt-02_params','15_features',
                                          xbc_param_dict['Optimal_Params_02'],
                                          xbc_features_dict['Top_15_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====xbc_opt-02_params_15_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/XGBoost_Classifier/xbc_opt-02_params_15_features_metrics.csv')
print('================================================')

'''
#Top 10 features [1,2,3,4,5,9,13,15,20,22]
multi_df_model_create_train_pred_analysis(df_file_dict,'xbc','opt-02_params','10_features',
                                          xbc_param_dict['Optimal_Params_02'],
                                          xbc_features_dict['Top_10_Features'],
                                          19,42,0.2,0.7,10)
'''
print('=====xbc_opt-02_params_10_features_metrics=====')
#analyze_report_metrics_df('files/csv/std_reports/XGBoost_Classifier/xbc_opt-02_params_10_features_metrics.csv')
print('================================================')

print('=====XGBoost Classifier - Create Train & Test sets| Train & Test Model | Analyze Results [Completed]=====')


'''----------------------------------------------------------------------------------------------------------------'''



'''Analyze Total Model Metrics'''

#Total Metrics according to: Preprocessing(12), Feature Number(3[22,15,12]) & Hypeparameters (3[default,opt-01,opt-02])
multi_dataset_model_metrics_total_to_csv(['xbc'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'prep_features_params',
                                         ['fc_none','fc_std',
                                          'fc_mm_0-1','fc_mm_0-10','fc_mm_0-100','fc_mm_0-1000',
                                          'lr_none','lr_std',
                                          'lr_mm_0-1','lr_mm_0-10','lr_mm_0-100','lr_mm_0-1000'],
                                         ['22_features','15_features','10_features'],
                                         ['default_params','opt-01_params','opt-02_params'])

multi_df_sort_values('xbc',['fc','lr'],'prep_features_params',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])


#Total Metrics according to: Preprocessing(12) & Feature Number(3[22,15,12])
multi_dataset_model_metrics_total_to_csv(['xbc'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'prep_features',
                                         ['fc_none','fc_std',
                                          'fc_mm_0-1','fc_mm_0-10','fc_mm_0-100','fc_mm_0-1000',
                                          'lr_none','lr_std',
                                          'lr_mm_0-1','lr_mm_0-10','lr_mm_0-100','lr_mm_0-1000'],
                                         ['22_features','15_features','10_features'],
                                         ['_'])

multi_df_sort_values('xbc',['fc','lr'],'prep_features',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])


#Total Metrics according to: Preprocessing(12) & Hypeparameters (3[default,opt-01,opt-02])
multi_dataset_model_metrics_total_to_csv(['xbc'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'prep_params',
                                         ['fc_none','fc_std',
                                          'fc_mm_0-1','fc_mm_0-10','fc_mm_0-100','fc_mm_0-1000',
                                          'lr_none','lr_std',
                                          'lr_mm_0-1','lr_mm_0-10','lr_mm_0-100','lr_mm_0-1000'],
                                         ['default_params','opt-01_params','opt-02_params'],
                                         ['_'])

multi_df_sort_values('xbc',['fc','lr'],'prep_params',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])


#Total Metrics according to: Feature Number(3[22,15,12]) & Hypeparameters (3[default,opt-01,opt-02])
multi_dataset_model_metrics_total_to_csv(['xbc'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'features_params',
                                         ['fc_','lr_'],
                                         ['22_features','15_features','10_features'],
                                         ['default_params','opt-01_params','opt-02_params'])

multi_df_sort_values('xbc',['fc','lr'],'features_params',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])


print('=====XGBoost Classifier - Analyze Total Model Metrics [Completed]=====')


'''----------------------------------------------------------------------------------------------------------------'''



'''Find Optimal Hyperparameters'''

for df_file in df_file_dict:
    
    for repeat in range(5):
        
        X_train,X_test,y_train,y_test = prepare_dataset(df_file_dict[df_file][0],
                                                        xbc_features_dict['All_22_Features'],
                                                        19,42,0.2,0.7,
                                                        df_file_dict[df_file][2][0],
                                                        df_file_dict[df_file][2][1],
                                                        df_file_dict[df_file][1])
        
        
        #Tester
        find_model_opt_param(X_train,X_test,y_train,y_test,'xbc',
                             {'objective':['binary:logistic'],'use_label_encoder':[True],
                              'base_score':[0.5],'booster':['gbtree','gblinear'],'colsample_bylevel':[1],
                              'colsample_bynode':[1],'colsample_bytree':[1],'enable_categorical':[False],
                              'gamma':[0],'gpu_id':[-1],'importance_type':[None],'interaction_constraints':[''],
                              'learning_rate':[0.300000012],'max_delta_step':[0],'max_depth':[6],
                              'min_child_weight':[1],'missing':[nan],'monotone_constraints':['()'],
                              'n_estimators':[100],'n_jobs':[8],'num_parallel_tree':[1],'predictor':['auto'],
                              'random_state':[42],'reg_alpha':[0],'reg_lambda':[1],'scale_pos_weight':[1],
                              'subsample':[1],'tree_method':['exact'],'validate_parameters':[1],
                              'verbosity':[None]},
                             df_file_dict[df_file][1],-1)
        
        '''
        find_model_opt_param(X_train,X_test,y_train,y_test,'xbc',
                             {'objective':['binary:logistic'],'use_label_encoder':[True],
                              'base_score':[0.5],'booster':['gbtree','gblinear','dart'],'colsample_bylevel':[1],
                              'colsample_bynode':[1],'colsample_bytree':[1],'enable_categorical':[True,False],
                              'gamma':[0],'gpu_id':[-1],'importance_type':['gain','weight','cover','total_gain','total_cover',None],
                              'interaction_constraints':[''],'learning_rate':[0.300000012],
                              'max_delta_step':[0],'max_depth':[5,6,10,12,15,20],'min_child_weight':[1],
                              'missing':[nan],'monotone_constraints':['()','(1,0)'],'n_estimators':[50,75,100,125,150],
                              'n_jobs':[8],'num_parallel_tree':[1],'predictor':['auto'],
                              'random_state':[42],'reg_alpha':[0],'reg_lambda':[1],'scale_pos_weight':[1],
                              'subsample':[1],'tree_method':['auto','exact','approx','hist'],
                              'validate_parameters':[1],'verbosity':[None]},
                             df_file_dict[df_file][1],-1)
        '''
        '''
        find_model_opt_param(X_train,X_test,y_train,y_test,'xbc',
                             {'objective':['binary:logistic'],'use_label_encoder':[True],
                              'base_score':[0.5],'booster':['gbtree'],'colsample_bylevel':[1],
                              'colsample_bynode':[1],'colsample_bytree':[1],'enable_categorical':[False],
                              'gamma':[0],'gpu_id':[-1],'importance_type':['gain'],
                              'interaction_constraints':[''],'learning_rate':[0.300000012],
                              'max_delta_step':[0],'max_depth':[5,11],'min_child_weight':[1],
                              'missing':[nan],'monotone_constraints':['()','(1,0)'],'n_estimators':[75,100,125,150],
                              'n_jobs':[8],'num_parallel_tree':[1],'predictor':['auto'],
                              'random_state':[42],'reg_alpha':[0],'reg_lambda':[1],'scale_pos_weight':[1],
                              'subsample':[1],'tree_method':['auto'],
                              'validate_parameters':[1],'verbosity':[None]},
                             df_file_dict[df_file][1],-1)
        '''

print('=====XGBoost Classifier - Find Optimal Hyperparameters [Completed]=====')


'''----------------------------------------------------------------------------------------------------------------'''

''''''
