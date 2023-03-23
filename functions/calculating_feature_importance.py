import pandas as pd
#import numpy as np
#import matplotlib.pyplot as mtp
#from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure

#from classes.Column import Column

from sklearn.inspection import permutation_importance

from functions.models.create_train_predict_analyze import model_name
from functions.models.create_train_predict_analyze import create_model

from functions.write.write_to_file import write_string_in_text_file

from functions.plot.graph_plotting import series_graph_plot_show_save

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



'''Calculating Feature Importance Functions'''

#Calculate Feature Correlation for a df(Pearson-Spearman-Kendall)
def feature_correlation(df_file_path,x_column_drop_list,y_column_name):
    
    start = time.time()
    
    dataframe = pd.read_csv(df_file_path)
    
    df = dataframe.drop(columns=x_column_drop_list)
    
    pearson_corr = df.corr(method ='pearson')
    p_analysis = pearson_corr[y_column_name].sort_values(ascending=False)
    
    spearman_corr = df.corr(method ='spearman')
    s_analysis = spearman_corr[y_column_name].sort_values(ascending=False)
    
    kendall_corr = df.corr(method ='kendall')
    k_analysis = kendall_corr[y_column_name].sort_values(ascending=False)
    
    feat_corr_report_text = f'\n===============[All Features Correlation]===============\n\nFile: {df_file_path}\n\n\n---------------[Pearson Correllation]---------------\n{p_analysis}\n----------------------------------------------------\n\n---------------[Spearman Correllation]---------------\n{s_analysis}\n----------------------------------------------------\n\n---------------[Kendall Correllation]---------------\n{k_analysis}\n----------------------------------------------------\n\n'
    
    feat_corr_report_full_text,duration = write_string_in_text_file('files/txt/f_importance/Feature_Correlation',feat_corr_report_text,start)
    
    print(feat_corr_report_full_text)


#Calculate Feature Correlation for multple dfs
def multi_df_feature_correlation(rel_path,df_file_name_c_prefix,df_pos_type_list,df_file_name_c_suffix,df_scaler_type_list,x_column_drop_list,y_column_name):
    
    for pos_type in df_pos_type_list:
        
        for scaler_type in df_scaler_type_list:
            
            df_file_path = rel_path + df_file_name_c_prefix + pos_type + df_file_name_c_suffix + scaler_type + '.csv'
            
            feature_correlation(df_file_path,x_column_drop_list,y_column_name)



#Calculate Feature Importance of a Method (Logistic Regression, Decision Tree, Random Forest, XGBoost, KNeighbors)
def feature_importance(df_file_path,pos_scaler,x_column_drop_list,y_column_name,method_name,param_dict,top_feature_num):
    
    start = time.time()
    
    dataframe = pd.read_csv(df_file_path)
    
    x_column_drop_list.append(y_column_name)
    
    model,method = create_model(method_name,param_dict)
    
    features=dataframe.drop(columns=x_column_drop_list)
    
    model.fit(features,dataframe[y_column_name])
    
    if method_name == 'lgr':
        
        feat_importances = pd.Series(model.coef_[0], index=features.columns)
    
    elif ((method_name == 'dtc' or method_name == 'dtr') or (method_name == 'rfc' or method_name == 'rfr') or (method_name == 'xbc' or method_name == 'xbr')):
        
        feat_importances = pd.Series(model.feature_importances_, index=features.columns)
    
    elif (method_name == 'knc' or method_name == 'knr'):
        
        results = permutation_importance(model,features,dataframe[y_column_name], scoring='accuracy')
        
        feat_importances = pd.Series(results.importances_mean, index=features.columns)
    
    feat_importances,df_export = feat_imp_series_modification(feat_importances,'Feature_Name','Importance_Value')
    
    feat_imp_report_text = f'\n=======[{method} All Features Importance]=======\n\nFile: {df_file_path}\n\n{feat_importances}\n\n'
    
    feat_imp_report_full_text,duration = write_string_in_text_file('files/txt/f_importance/' + method + '/' + method_name + '_feat_imp',feat_imp_report_text,start)
    
    graph_title = method + ' Top ' + str(top_feature_num) + ' Features\' Importance (' + pos_scaler +')'
    
    graph_file_path = 'files/png/f_importance/' + method + '/' + method_name + '_Top_' + str(top_feature_num) + '_(' + pos_scaler +').png'
    
    series_graph_plot_show_save(feat_importances,graph_title,'barh',200,graph_file_path)
    
    print(feat_imp_report_full_text)
    
    return df_export


#Calculate Ffeature Importance of a Method for multple dfs
def multi_df_feature_importance(rel_path,df_file_name_c_prefix,df_pos_type_list,df_file_name_c_suffix,df_scaler_type_list,x_column_drop_list,y_column_name,method_name,param_dict,top_feature_num):
    
    start = time.time()
    
    features_dict = {'SEX':0,'TYPE_OF_PATIENT':0,'PNEUMONIA':0,'AGE':0,'DIABETIC':0,'COPD':0,'ASTHMA':0,
                     'IMMUNOSUPPRESSED':0,'HYPERTENSION':0,'OTHER_CHRONIC_DISEASE':0,'CARDIOVASCULAR':0,'OBESITY':0,
                     'CHRONIC_KIDNEY_FAILURE':0,'SMOKER':0,'CONTACT_WITH_COVID-19_CASE':0,
                     'DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION':0,'INTUBATED_2':0,'INTUBATED_97':0,
                     'PREGNANCY_2':0,'PREGNANCY_97':0,'ICU_2':0,'ICU_97':0}
    
    for pos_type in df_pos_type_list:
        
        for scaler_type in df_scaler_type_list:
            
            df_file_path = rel_path + df_file_name_c_prefix + pos_type + df_file_name_c_suffix + scaler_type + '.csv'
            pos_scaler = pos_type + '_' + scaler_type
            
            df_feat_imp = feature_importance(df_file_path,pos_scaler,x_column_drop_list,y_column_name,method_name,param_dict,top_feature_num)
            
            for index, row in df_feat_imp.iterrows():
                
                features_dict[row['Feature_Name']] = (features_dict[row['Feature_Name']]) + abs(row['Importance_Value'])
                
    feat_importances_all = pd.Series(features_dict)
    
    feat_importances_all,df = feat_imp_series_modification(feat_importances_all,'Feature_Name','Importance_Score(Total)')
    
    feat_importances_all['Average'] = feat_importances_all['Importance_Score(Total)']/(len(df_pos_type_list) * len(df_scaler_type_list))
    
    method = model_name(method_name)
    
    feat_importances_all.to_csv('files/csv/f_importance/' + method + '_All_Features_Importance_Overall.csv')
    
    #feat_importances_all = pd.read_csv('files/csv/f_importance/' + method + '_All_Features_Importance_Overall.csv',header=0,index_col='Feature_Name')
    
    feat_importances_all_report_text = f'\n============[{method} All Features Importance Overall]============\n\n{feat_importances_all}\n\n'
    
    feat_importances_all_report_full_text,duration = write_string_in_text_file('files/txt/f_importance/' + method + '/' + method_name + '_feat_imp_total',feat_importances_all_report_text,start)
    
    graph_title = method + ' All Features\' Importance Overall'
    
    graph_file_path = 'files/png/f_importance/' + method + '/' + method_name + '_all_feats_total.png'
    
    series_graph_plot_show_save(feat_importances_all,graph_title,'barh',200,graph_file_path)
    
    print(feat_importances_all_report_full_text)


#Modifies the Feature Importace Series
def feat_imp_series_modification(series,index_col_name,score_col_name):
    
    series.index.name = index_col_name

    series = series.reset_index(name=score_col_name)

    series = series.sort_values(by=[score_col_name], ascending=False)

    series = series.reset_index(drop=True)
    
    df = series.copy()

    series.set_index(index_col_name,drop=True,inplace=True)
    
    return series,df


''''''
