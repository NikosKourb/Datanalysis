import pandas as pd
#import numpy as np

import csv

from os.path import exists

#from classes.Column import Column

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from functions.analyzing import numerical_column_analysis

from functions.preprocess.preparing import prepare_dataset

from functions.write.write_to_file import append_list_as_row
from functions.write.write_to_file import append_dict_as_row
from functions.write.write_to_file import write_string_in_text_file

from functions.plot.graph_plotting import conf_matrix_plot
from functions.plot.graph_plotting import roc_curve_plot
from functions.plot.graph_plotting import precision_recall_curve_plot




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

#Return model name
def get_model_name(method):
    
    model_name='none'
    
    method_dict = {'lgr':'Logistic_Regression','dtc':'Decision_Tree_Classifier',
                   'rfc':'Random_Forest_Classifier','xbc':'XGBoost_Classifier',
                   'mlp':'Multi_Layer_Perceptrons','knc':'KNeighbors_Classifier',
                   'svc':'Support_Vector_Classifier','kms':'KMeans','test':'Tester'}
    
    model_name = method_dict[method]
    
    return model_name


#Create a Classifier Model with default or specific Hyperparameters
def create_model(method,param_dict):
    
    model_name='none'
    
    method_dict = {'lgr':('Logistic_Regression',LogisticRegression()),
                   'dtc':('Decision_Tree_Classifier',DecisionTreeClassifier()),
                   'rfc':('Random_Forest_Classifier',RandomForestClassifier()),
                   'xbc':('XGBoost_Classifier',XGBClassifier()),
                   'mlp':('Multi_Layer_Perceptrons',MLPClassifier()),
                   'knc':('KNeighbors_Classifier',KNeighborsClassifier()),
                   'svc':('Support_Vector_Classifier',SVC()),
                   'kms':('KMeans',KMeans()),
                   'test':('Tester',LogisticRegression())}
    
    model_name = method_dict[method][0]
    model = method_dict[method][1]
    model.set_params(**param_dict)
    
    return model,model_name


#Primary Report Variables
def primary_report_metrics(X_test,y_test,model,conf_mtrx_graph_title,conf_mtrx_graph_file_path,graph_file_dpi):
    
    y_pred = model.predict(X_test)
    
    model_cl_report = classification_report(y_test,y_pred)
    model_precission = precision_score(y_test,y_pred,pos_label='positive',average='micro')
    model_recall = recall_score(y_test,y_pred)
    model_accuracy = accuracy_score(y_test, y_pred)
    model_f1_score = f1_score(y_test,y_pred)
    model_conf_mtrx = confusion_matrix(y_test,y_pred)
    model_params = model.get_params()
    
    conf_matrix_plot(model,X_test,y_test,conf_mtrx_graph_title,conf_mtrx_graph_file_path,graph_file_dpi)
    
    return y_pred,model_cl_report,model_precission,model_recall,model_accuracy,model_f1_score,model_conf_mtrx,model_params


#Secondary Report Variables
def secondary_report_metrics(X_test,y_test,model,model_name,au_roc_graph_title,au_roc_graph_file_path,p_r_curve_graph_title,p_r_curve_graph_file_path,graph_file_dpi):
    
    no_skill_probs = [0 for i in range(len(y_test))]
    
    model_probs = model.predict_proba(X_test)
    model_probs = model_probs[:, 1]
    
    no_skill_auc_roc = roc_auc_score(y_test, no_skill_probs)
    model_auc_roc = roc_auc_score(y_test, model_probs)
    
    au_roc_graph_title = au_roc_graph_title + ' ROC AUC: ' + str(model_auc_roc)
    roc_curve_plot(y_test,no_skill_probs,model_probs,model_name,au_roc_graph_title,au_roc_graph_file_path,graph_file_dpi)
    
    model_precision, model_recall, _ = precision_recall_curve(y_test,model_probs)
    model_auc_p_r = auc(model_recall,model_precision)
    
    p_r_curve_graph_title = p_r_curve_graph_title + ' Precision-Recall AUC: ' + str(model_auc_p_r)
    precision_recall_curve_plot(y_test,model_precision,model_recall,model_name,p_r_curve_graph_title,p_r_curve_graph_file_path,graph_file_dpi)
    
    return no_skill_auc_roc,model_auc_roc,model_auc_p_r


#Create Train, Predict & Show the statistic results of a Model with default or specific Hyperparameters
def model_create_train_pred_analysis(X_train,X_test,y_train,y_test,method,param_dict,dataset_name,rep_num,m_param_type_name,features_list_name,pos_type,scaler_type):
    
    start = time.time()
    
    model,model_name = create_model(method,param_dict)
    
    model.fit(X_train,y_train)
    
    conf_mtrx_graph_title = ''
    au_roc_graph_title = ''
    p_r_curve_graph_title = ''
    
    graph_file_path_common = 'files/png/std_reports/' + model_name + '/' + m_param_type_name + '/' + features_list_name + '/'
    conf_mtrx_graph_file_path = graph_file_path_common + 'conf_matrices' + '/' + '[' + str(rep_num) + ']_' + method + '_' + pos_type + '_' + scaler_type + '.png'
    au_roc_graph_file_path = graph_file_path_common + 'au_roc' + '/' + '[' + str(rep_num) + ']_' + method + '_' + pos_type + '_' + scaler_type + '.png'
    p_r_curve_graph_file_path = graph_file_path_common + 'p_r_curve' + '/' + '[' + str(rep_num) + ']_' + method + '_' + pos_type + '_' + scaler_type + '.png'
    
    y_pred,model_cl_report,model_precission,model_recall,model_accuracy,model_f1_score,model_conf_mtrx,model_params = primary_report_metrics(X_test,y_test,model,conf_mtrx_graph_title,conf_mtrx_graph_file_path,200)
    
    no_skill_auc_roc,model_auc_roc,model_auc_p_r = secondary_report_metrics(X_test,y_test,model,model_name,au_roc_graph_title,au_roc_graph_file_path,p_r_curve_graph_title,p_r_curve_graph_file_path,200)
    
    report_text = f'\n[{rep_num}]================{model_name}({dataset_name})================\n--------------------[Classification Report]--------------------\n\n{model_cl_report}\n---------------------------------------------------------------\nPrecision score: {model_precission}\nRecall score: {model_recall}\nAccuracy score: {model_accuracy}\nF1 score: {model_f1_score}\n\nROC AUC({method}): {model_auc_roc}\nROC AUC(no skill): {no_skill_auc_roc}\n\nPrecision-Recall AUC: {model_auc_p_r}\n\n---------------------------------------------------------------\nConfusion Matrix:\n{model_conf_mtrx}\n---------------------------------------------------------------\nModel Parameters:\n{model_params}\n---------------------------------------------------------------\n'
    
    report_full_text,duration = write_string_in_text_file('files/txt/std_reports/' + model_name + '/' + m_param_type_name + '/' + features_list_name + '/' + method + '_' + pos_type + '_' + scaler_type,report_text,start)
    
    print(report_full_text)
    
    return model_precission,model_recall,model_accuracy,model_f1_score,model_auc_roc,model_auc_p_r,duration


#Multi df Create Train, Predict & Show the statistic results of a Model with default or specific Hyperparameters
def multi_df_model_create_train_pred_analysis(df_file_dict,method,m_param_type_name,features_list_name,param_dict,x_column_list,y_column_list,rand_state,df_frac,train_sz,repeats):
    
    for df_file in df_file_dict:
        
        df_file_path = df_file_dict[df_file][0]; over_s_val = df_file_dict[df_file][2][0]
        under_s_val = df_file_dict[df_file][2][1]; dataset_name = df_file_dict[df_file][1]
        
        for rep in range(repeats):
            
            rep_num = 1
            field_names_list = ['Dataset','Precision','Recall','Accuracy','F1','ROC_AUC','P_R_AUC','Runtime(seconds)']
            model_name = get_model_name(method)
            csv_file_path_name = 'files/csv/std_reports/' + model_name + '/' + method + '_' + m_param_type_name + '_' +  features_list_name + '_metrics.csv'
    
            file_exists = exists(csv_file_path_name)
    
            if file_exists == False:
        
                append_list_as_row(csv_file_path_name,field_names_list)
    
            else:
                
                file = open(csv_file_path_name)
                reader = csv.reader(file)
                lines= len(list(reader))
                file.close()
                
                rep_num = lines
            
            X_train,X_test,y_train,y_test = prepare_dataset(df_file_path,
                                                            x_column_list,y_column_list,
                                                            rand_state,df_frac,train_sz,
                                                            over_s_val,under_s_val,
                                                            dataset_name)
            
            dt_chunks = dataset_name.split('_',3); dt_chunks.pop(1);dt_chunks.pop(1)
            pos_type = dt_chunks[0]; scaler_type = dt_chunks[1]
            
            model_precission,model_recall,model_accuracy,model_f1_score,model_auc_roc,model_auc_p_r,duration = model_create_train_pred_analysis(X_train,X_test,y_train,y_test,method,param_dict,m_param_type_name + '_' + dataset_name,
                                                                                                                                                rep_num,m_param_type_name,features_list_name,pos_type,scaler_type)
                
            new_row_dict = {}
            new_row_dict['Dataset'] = pos_type + '_' + scaler_type; new_row_dict['Precision']= model_precission
            new_row_dict['Recall']= model_recall; new_row_dict['Accuracy']= model_accuracy
            new_row_dict['F1']= model_f1_score; new_row_dict['ROC_AUC']= model_auc_roc
            new_row_dict['P_R_AUC']= model_auc_p_r; new_row_dict['Runtime(seconds)']= duration
                
            append_dict_as_row(csv_file_path_name,new_row_dict,field_names_list)


''''''
