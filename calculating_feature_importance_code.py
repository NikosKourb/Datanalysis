#import pandas as pd
#import numpy as np

#import time

#from classes.Column import Column

#from functions.calculating_feature_importance import feature_correlation
from functions.calculating_feature_importance import multi_df_feature_correlation
#from functions.calculating_feature_importance import feature_importance
from functions.calculating_feature_importance import multi_df_feature_importance

import warnings
warnings.filterwarnings("ignore")



rel_path = 'files/csv/data/'
df_file_name_c_prefix = 'Covid19MPD_8_23_en_pos_'
df_pos_type_list = ['fc']
df_file_name_c_suffix = '_valid_lb_'
#df_scaler_type_list = ['none','std','mm_0-1','mm_0-10','mm_0-100','mm_0-1000']
df_scaler_type_list = ['none','std','mm_0-1','mm_0-10','mm_0-100','mm_0-1000']

'''Calculate Feature Importance'''
#Feature Correlation(Pearson-Spearman-Kendall)
multi_df_feature_correlation(rel_path,df_file_name_c_prefix,df_pos_type_list,
                            df_file_name_c_suffix,df_scaler_type_list,
                            ['REGISTRATION_ID','CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION'],
                            'SURVIVED')

#Logistic Regression
multi_df_feature_importance(rel_path,df_file_name_c_prefix,df_pos_type_list,
                            df_file_name_c_suffix,df_scaler_type_list,
                            ['REGISTRATION_ID','CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION'],
                            'SURVIVED','lgr',{},22)

#Decision Tree Classifier
multi_df_feature_importance(rel_path,df_file_name_c_prefix,df_pos_type_list,
                            df_file_name_c_suffix,df_scaler_type_list,
                            ['REGISTRATION_ID','CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION'],
                            'SURVIVED','dtc',{},22)

#Decision Tree Regressor
#multi_df_feature_importance(rel_path,df_file_name_c_prefix,df_pos_type_list,
#                            df_file_name_c_suffix,df_scaler_type_list,
#                            ['REGISTRATION_ID','CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION'],
#                            'SURVIVED','dtr',{},22)

#Random Forest Classifier
multi_df_feature_importance(rel_path,df_file_name_c_prefix,df_pos_type_list,
                            df_file_name_c_suffix,df_scaler_type_list,
                            ['REGISTRATION_ID','CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION'],
                            'SURVIVED','rfc',{},22)

#Random Forest Regressor
#multi_df_feature_importance(rel_path,df_file_name_c_prefix,df_pos_type_list,
#                            df_file_name_c_suffix,df_scaler_type_list,
#                            ['REGISTRATION_ID','CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION'],
#                            'SURVIVED','rfr',{},22)

#XGBoost Classifier
multi_df_feature_importance(rel_path,df_file_name_c_prefix,df_pos_type_list,
                            df_file_name_c_suffix,df_scaler_type_list,
                            ['REGISTRATION_ID','CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION'],
                            'SURVIVED','xbc',{},22)

#XGBoost Regressor
#multi_df_feature_importance(rel_path,df_file_name_c_prefix,df_pos_type_list,
#                            df_file_name_c_suffix,df_scaler_type_list,
#                            ['REGISTRATION_ID','CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION'],
#                            'SURVIVED','xbr',{},22)

'''
#KNeighbors Classifier
multi_df_feature_importance(rel_path,df_file_name_c_prefix,df_pos_type_list,
                            df_file_name_c_suffix,df_scaler_type_list,
                            ['REGISTRATION_ID','LAB_RESULT','FINAL_CLASSIFICATION'],
                            'SURVIVED','knc',{},22)

#KNeighbors Regressor
multi_df_feature_importance(rel_path,df_file_name_c_prefix,df_pos_type_list,
                            df_file_name_c_suffix,df_scaler_type_list,
                            ['REGISTRATION_ID','LAB_RESULT','FINAL_CLASSIFICATION'],
                            'SURVIVED','knr',{},22)
'''
''''''


