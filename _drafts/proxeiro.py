import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp

from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder #Για unordered categorical features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import warnings
warnings.filterwarnings("ignore")

#from imblearn.pipeline import Pipeline
#from matplotlib import pyplot
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split



'''
#
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
    df =  pd.read_csv(csv_filepath,header=0)
    data = df[df['Dataset'] == dataset_name]
    
    #print(data.count())
    
    csv_new_filepath = 'files/csv/std_reports/' + model_name + '/_' + method + '_' + dataset_type + '_total_metrics.csv'
    
    file_exists = exists(csv_new_filepath)
    
    if file_exists == False:

        append_list_as_row(csv_new_filepath,field_names_list)
    
    new_row_dict = {}
    new_row_dict['Dataset'] = dataset_name_final
    
    for metric in metrics_list:
        
        new_row_dict[str(metric + '_mean')] = data[metric].mean()
        new_row_dict[str(metric + '_std')] = data[metric].std()
        new_row_dict[str(metric + '_min')] = data[metric].min()
        new_row_dict[str(metric + '_max')] = data[metric].max()
    
    
    new_row_dict = {'Dataset': 'lr_mm_0-1_default_15', 'Precision_mean': 0.8988920996222299, 'Precision_std': 0.001128685784501613, 'Precision_min': 0.8966575540310517, 'Precision_max': 0.9004745855973588, 'Recall_mean': 0.9130211023886388, 'Recall_std': 0.0026470851777022786, 'Recall_min': 0.908622833605347, 'Recall_max': 0.916177297129678, 'Accuracy_mean': 0.8988920996222299, 'Accuracy_std': 0.001128685784501613, 'Accuracy_min': 0.8966575540310517, 'Accuracy_max': 0.9004745855973588, 'F1_mean': 0.8888962971369528, 'F1_std': 0.0013719021436301032, 'F1_min': 0.886111926935794, 'F1_max': 0.89060875810323, 'ROC_AUC_mean': 0.9536719342894188, 'ROC_AUC_std': 0.0005275933773926882, 'ROC_AUC_min': 0.952678133650402, 'ROC_AUC_max': 0.9544544146421692, 'P_R_AUC_mean': 0.924762802879506, 'P_R_AUC_std': 0.0013036256955442755, 'P_R_AUC_min': 0.923016897996264, 'P_R_AUC_max': 0.9273138457467756, 'Runtime(seconds)_mean': 3.575614070892334, 'Runtime(seconds)_std': 0.17588038282996987, 'Runtime(seconds)_min': 3.368624687194824, 'Runtime(seconds)_max': 3.9562368392944336}
    
    #print(new_row_dict)
    
    df_new = pd.read_csv(csv_new_filepath,header=0)
    
    #print(df_new)
    
    new_row_1 = str(new_row_dict['Dataset'])
    new_row_2 = str(new_row_dict['Precision_mean'])
    new_row_3 = str(new_row_dict['Precision_std'])
    
    df_new_1 = str(df_new['Dataset'])
    df_new_2 = str(df_new['Precision_mean'])
    df_new_3 = str(df_new['Precision_std'])
    
    values_exist = ((new_row_1 == df_new_1) and (new_row_2 == df_new_2) and (new_row_3 == df_new_3))
    print(values_exist)
    
    
    values_exist = ((df_new['Dataset'] == new_row_dict['Dataset']) & 
                    (df_new['Precision_mean'] == new_row_dict['Precision_mean']) &
                    (df_new['Precision_std'] == new_row_dict['Precision_std']) &
                    (df_new['Precision_min'] == new_row_dict['Precision_min']) &
                    (df_new['Precision_max'] == new_row_dict['Precision_max']) &
                    (df_new['Recall_mean'] == new_row_dict['Recall_mean']) &
                    (df_new['Recall_std'] == new_row_dict['Recall_std']) &
                    (df_new['Recall_min'] == new_row_dict['Recall_min']) &
                    (df_new['Recall_max'] == new_row_dict['Recall_max']) &
                    (df_new['Accuracy_mean'] == new_row_dict['Accuracy_mean']) &
                    (df_new['Accuracy_std'] == new_row_dict['Accuracy_std']) &
                    (df_new['Accuracy_min'] == new_row_dict['Accuracy_min']) &
                    (df_new['Accuracy_max'] == new_row_dict['Accuracy_max']) &
                    (df_new['F1_mean'] == new_row_dict['F1_mean']) &
                    (df_new['F1_std'] == new_row_dict['F1_std']) &
                    (df_new['F1_min'] == new_row_dict['F1_min']) &
                    (df_new['F1_max'] == new_row_dict['F1_max']) &
                    (df_new['ROC_AUC_mean'] == new_row_dict['ROC_AUC_mean']) &
                    (df_new['ROC_AUC_std'] == new_row_dict['ROC_AUC_std']) &
                    (df_new['ROC_AUC_min'] == new_row_dict['ROC_AUC_min']) &
                    (df_new['ROC_AUC_max'] == new_row_dict['ROC_AUC_max']) &
                    (df_new['P_R_AUC_mean'] == new_row_dict['P_R_AUC_mean']) &
                    (df_new['P_R_AUC_std'] == new_row_dict['P_R_AUC_std']) &
                    (df_new['P_R_AUC_min'] == new_row_dict['P_R_AUC_min']) &
                    (df_new['P_R_AUC_max'] == new_row_dict['P_R_AUC_max']) &
                    (df_new['Runtime(seconds)_mean'] == new_row_dict['Runtime(seconds)_mean']) &
                    (df_new['Runtime(seconds)_std'] == new_row_dict['Runtime(seconds)_std']) &
                    (df_new['Runtime(seconds)_min'] == new_row_dict['Runtime(seconds)_min']) &
                    (df_new['Runtime(seconds)_max'] == new_row_dict['Runtime(seconds)_max']))
    
    print(values_exist)
    
    
    #row_num = len(df_new.index)
    #print(row_num)
    #
    #if row_num > 0:
    row_exist_counter = 0
    
    for index, row in df_new.iterrows():
        #print(row['c1'], row['c2'])
        t_f_row_list = []
        
        for key in new_row_dict:
                
            t_f = (new_row_dict[key] == row[key])
            
            if (t_f == True):
                
                t_f = 't'
            
            else:
                
                t_f = 'f'
            
            t_f_row_list.append(t_f)
            
            #print(new_row_dict[key])
            #print(row[key])
            #print(t_f)
            
        print(t_f_row_list)
        
        values_exist = ('f' not in t_f_row_list)
        ##print(values_exist)
        
        if (values_exist == True):
            
            row_exist_counter = row_exist_counter + 1
        
        #print('row contains the specific values')
    #
    print(row_exist_counter)
    
    if (values_exist == False):
        
        append_dict_as_row(csv_new_filepath,new_row_dict,field_names_list)
        print('appended')
        
    #print(new_row_dict)
    
    #return new_row_dict

'''
'''
df_new_row_list.append(row['Dataset'])
df_new_row_list.append(row['Precision_mean'])
df_new_row_list.append(row['Precision_std'])
df_new_row_list.append(row['Precision_min'])
df_new_row_list.append(row['Precision_max'])
df_new_row_list.append(row['Recall_mean'])
df_new_row_list.append(row['Recall_std'])
df_new_row_list.append(row['Recall_min'])
df_new_row_list.append(row['Recall_max'])
df_new_row_list.append(row['Accuracy_mean'])
df_new_row_list.append(row['Accuracy_std'])
df_new_row_list.append(row['Accuracy_min'])
df_new_row_list.append(row['Accuracy_max'])
df_new_row_list.append(row['F1_mean'])
df_new_row_list.append(row['F1_std'])
df_new_row_list.append(row['F1_min'])
df_new_row_list.append(row['F1_max'])
df_new_row_list.append(row['ROC_AUC_mean'])
df_new_row_list.append(row['ROC_AUC_std'])
df_new_row_list.append(row['ROC_AUC_min'])
df_new_row_list.append(row['ROC_AUC_max'])
df_new_row_list.append(row['P_R_AUC_mean'])
df_new_row_list.append(row['P_R_AUC_std'])
df_new_row_list.append(row['P_R_AUC_min'])
df_new_row_list.append(row['P_R_AUC_max'])
df_new_row_list.append(row['Runtime(seconds)_mean'])
df_new_row_list.append(row['Runtime(seconds)_std'])
df_new_row_list.append(row['Runtime(seconds)_min'])
df_new_row_list.append(row['Runtime(seconds)_max'])
'''
'''
#Multi df Create Train, Predict & Show the statistic results of a Model with default or specific Hyperparameters
def multi_df_model_create_train_pred_analysis(rel_path,df_file_name_c_prefix,df_pos_type_dict,df_file_name_c_suffix,df_scaler_type_list,repeats,method_name,param_dict,m_param_type_name,features_list_name,x_column_list,y_column_list,rand_state,df_frac,train_sz):
    
    rep_num = 1
    field_names_list = ['Dataset','Precision','Recall','Accuracy','F1','ROC_AUC','P_R_AUC','Runtime(seconds)']
    method = model_name(method_name)
    file_path_name = 'files/csv/std_reports/' + method + '/' + method_name + '_' + m_param_type_name + '_' +  features_list_name + '_metrics.csv'
    file_exists = exists(file_path_name)
    
    if file_exists == False:
        
        append_list_as_row(file_path_name,field_names_list)
    
    else:
        file = open(file_path_name)

        reader = csv.reader(file)

        lines= len(list(reader))
        
        file.close()
        
        rep_num = lines
    
    for pos_type in df_pos_type_dict:
        
        for scaler_type in df_scaler_type_list:
            
            for rep in range(repeats):
                
                df_file_path = rel_path + df_file_name_c_prefix + pos_type + df_file_name_c_suffix + scaler_type + '.csv'
                dataset_name = features_list_name + '_' + pos_type + df_file_name_c_suffix + scaler_type
    
                X_train,X_test,y_train,y_test = prepare_dataset(df_file_path,x_column_list,y_column_list,
                                                                rand_state,df_frac,train_sz,
                                                                df_pos_type_dict[pos_type][0],
                                                                df_pos_type_dict[pos_type][1],
                                                                dataset_name)
            
                model_precission,model_recall,model_accuracy,model_f1_score,model_auc_roc,model_auc_p_r,duration = model_create_train_pred_analysis(X_train,X_test,y_train,y_test,
                                                                                                                                                    method_name,param_dict,
                                                                                                                                                    m_param_type_name + '_' + dataset_name,rep_num,
                                                                                                                                                    m_param_type_name,features_list_name,
                                                                                                                                                    pos_type,scaler_type)
                
                new_row_dict = {}
                
                new_row_dict['Dataset'] = pos_type + '_' + scaler_type
                new_row_dict['Precision']=model_precission
                new_row_dict['Recall']=model_recall
                new_row_dict['Accuracy']=model_accuracy
                new_row_dict['F1']=model_f1_score
                new_row_dict['ROC_AUC']=model_auc_roc
                new_row_dict['P_R_AUC']=model_auc_p_r
                new_row_dict['Runtime(seconds)']=duration
                
                append_dict_as_row(file_path_name,new_row_dict,field_names_list)
                
                rep_num = rep_num + 1
'''


#Run & Show the statistic results of Logistic Regression with certain hyperparameters
def log_reg_std(name,X_train,X_test,y_train,y_test,c_val,max_iter_val,penalty_val,solver_val,tol_val):
    
    print('\n'+'================Logistic Regression(' + name + ')================'+'\n')
    
    model = LogisticRegression(penalty=penalty_val,C=c_val,solver=solver_val,max_iter=max_iter_val,tol=tol_val)
    model.fit(X_train,y_train)
    
    predictions = model.predict(X_test)
    log_pred = model.predict(X_test)
    
    print('\n',classification_report(y_test, log_pred),'\n',confusion_matrix(y_test,predictions),'\n' + '\n' +
          '===============================================================')


#Find the Optimal Logistic Regression's Hyperparameters for a dataframe's train and test sets 
def find_log_reg_opt_param(name,X_train,X_test,y_train,y_test,pen_list,c_list,solv_list,m_itr_list,tol_list):
    
    optimum_model = LogisticRegression()
    #scaler = MinMaxScaler()
    #steps = [('sampler',ros),('scaler', scaler),('LogReg',log_clf)]
    #pipe = Pipeline(steps=steps, memory='tmp')
    pipe = Pipeline(steps=[('LogReg', optimum_model)], memory='tmp')
    penalty = pen_list #penalty = ['l1','l2','elasticnet','none']
    C = c_list #C = [0.01,0.1,1,10,100]
    solver = solv_list #solver = ['newton-cg','lbfgs','liblinear','sag','saga']
    max_iter = m_itr_list #max_iter = [50,75,100,125,150]
    tol = tol_list #tol = [1e-03,1e-02,1e-04,1e-05]
    estimator_tester = GridSearchCV(pipe,
                                    dict(LogReg__penalty=penalty, LogReg__C=C,
                                         LogReg__solver=solver, LogReg__max_iter=max_iter,
                                         LogReg__tol=tol),
                                    cv=5)

    estimator_tester.fit(X_train, y_train)
    optimum_log_pred= estimator_tester.predict(X_test)
    #with open('log_Reg.txt', 'w') as f:
    print('\n' + '====[Optimum Logistic Regression Results & Hyperparameters(' + name + ')]====' + '\n' + '\n',
          classification_report(y_test,optimum_log_pred),'\n' + '\n',
          estimator_tester.best_estimator_,'\n' + '\n',
          estimator_tester.best_params_,'\n' + '\n',
          confusion_matrix(y_test,optimum_log_pred),'\n' +
          '\n' + '======================================================================='+'\n')



covid19_df = pd.read_csv('01.Covid19MPD.(Pos_valid_bin_21).csv')

#print(covid19_df.head())
#print(covid19_df.info())
#print(covid19_df.describe())
'''
corr_pear = covid19_df.corr(method ='pearson')
print(corr_pear['21.SURVIVED'].sort_values(ascending=False))

corr_spear = covid19_df.corr(method ='spearman')
print(corr_spear['21.SURVIVED'].sort_values(ascending=False))

corr_kend = covid19_df.corr(method ='kendall')
print(corr_kend['21.SURVIVED'].sort_values(ascending=False))
'''

dataset = covid19_df.values
#X = dataset[:,[1,3,5,8,15]]
X = dataset[:,[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
y = dataset[:,20]
y=y.astype('int')

print(X , '\n')
print('\n', y)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,train_size=0.7)

print("train size X : ",X_train.shape)
print("train size y : ",y_train.shape)
print("test size X : ",X_test.shape)
print("test size y : ",y_test.shape)

#scaler = StandardScaler()
#ros = RandomOverSampler()
#scaler = MinMaxScaler()

'''Logistic Regression'''

print('\n'+'\n'+'==========Logistic Regression==========')

model_log_reg = LogisticRegression()
model_log_reg.fit(X_train,y_train)
predictions = model_log_reg.predict(X_test)
model_log_reg.fit(X_train, y_train)
log_pred = model_log_reg.predict(X_test)
print('Log_Reg')
print(classification_report(y_test, log_pred))
print(confusion_matrix(y_test,predictions))


model_log_reg_opt = LogisticRegression(penalty='none',C=0.005,solver='newton-cg',max_iter=50,tol=1e-04)
model_log_reg_opt.fit(X_train,y_train)
predictions_opt = model_log_reg_opt.predict(X_test)
model_log_reg_opt.fit(X_train, y_train)
log_pred_opt = model_log_reg_opt.predict(X_test)
print('Log_Reg_Opt')
print(classification_report(y_test, log_pred_opt))
print(confusion_matrix(y_test,predictions_opt))


model_log_tester = LogisticRegression()
#scaler = MinMaxScaler()
#steps = [('sampler',ros),
#         ('scaler', scaler),
#         ('LogReg',log_clf)]
#pipe = Pipeline(steps=steps, memory='tmp')
pipe = Pipeline(steps=[('LogReg', model_log_tester)], memory='tmp')
#penalty = ['l1', 'l2', 'elasticnet', 'none'] 
penalty = ['l1', 'l2', 'elasticnet', 'none']
#C = [0.01, 0.1, 1, 10, 100]
C = [0.1,1,10]
#solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
#max_iter = [50, 100, 200, 500]
max_iter = [50,100,150]
#tol = [1e-05, 1e-04, 1e-03, 1e-02]
tol = [1e-05, 1e-04, 1e-03, 1e-02]
estimator_tester = GridSearchCV(pipe, dict(LogReg__penalty=penalty, LogReg__C=C, LogReg__solver=solver, LogReg__max_iter=max_iter, LogReg__tol=tol), cv=5)
estimator_tester.fit(X_train, y_train)
log_pred_tester= estimator_tester.predict(X_test)
print('Log_Reg_Opt_Tester')
print(classification_report(y_test,log_pred_tester))
#print(f'Optimized Logistic Regression\n{classification_report(y_test, log_best_pred)}')
print(estimator_tester.best_estimator_)
print(estimator_tester.best_params_)
print(confusion_matrix(y_test,log_pred_tester))
#with open('log_Reg.txt', 'w') as f:
print('\n'+'======================================='+'\n')

'''
{'LogReg__C': 0.1, 
 'LogReg__max_iter': 50, 
 'LogReg__penalty': 'l2', 
 'LogReg__solver': 'sag', 
 'LogReg__tol': 0.01}



{'LogReg__C': 0.1, 
 'LogReg__max_iter': 50, 
 'LogReg__penalty': 'l2', 
 'LogReg__solver': 'sag', 
 'LogReg__tol': 0.01}

[[28613 23669]
 [15337 55490]]



{'LogReg__C': 1, 
 'LogReg__max_iter': 50, 
 'LogReg__penalty': 'none', 
 'LogReg__solver': 'sag', 
 'LogReg__tol': 0.01}

[[27664 24618]
 [14339 56488]]



{'LogReg__C': 10, 
 'LogReg__max_iter': 100, 
 'LogReg__penalty': 'none', 
 'LogReg__solver': 'sag', 
 'LogReg__tol': 0.01}

[[27253 25029]
 [13928 56899]]



{'LogReg__C': 10, 
 'LogReg__max_iter': 100, 
 'LogReg__penalty': 'l2', 
 'LogReg__solver': 'sag', 
 -'LogReg__tol': 0.001}
[[27568 24714]
 [14239 56588]]

'''

'''
3,15
For Logistic Regression
              precision    recall  f1-score   support

           0       0.72      0.26      0.39     52282
           1       0.63      0.92      0.75     70827

    accuracy                           0.64    123109
   macro avg       0.67      0.59      0.57    123109
weighted avg       0.67      0.64      0.60    123109

3,8
For Logistic Regression
              precision    recall  f1-score   support

           0       0.73      0.25      0.37     52282
           1       0.63      0.93      0.75     70827

    accuracy                           0.64    123109
   macro avg       0.68      0.59      0.56    123109
weighted avg       0.67      0.64      0.59    123109


3,5
For Logistic Regression
              precision    recall  f1-score   support

           0       0.65      0.50      0.57     52282
           1       0.68      0.80      0.74     70827

    accuracy                           0.67    123109
   macro avg       0.67      0.65      0.65    123109
weighted avg       0.67      0.67      0.66    123109
'''






'''DecisionTree'''

'''
Visualizing Decision Trees:
pip install graphviz
pip install pydotplus
'''

print('\n'+'\n'+'==========DecisionTree==========')
#OneHotEncoder
tree_clf = DecisionTreeClassifier(criterion='gini', splitter='random', max_features= 'sqrt',random_state = 42)
tree_clf.fit(X_train, y_train)
tree_pred = tree_clf.predict(X_test)
print(f'For Decision Tree classifier\n{classification_report(y_test, tree_pred)}')

tree_clf_2 = DecisionTreeClassifier(criterion='entropy', splitter='best', max_features= 'log2')
tree_clf_2.fit(X_train, y_train)
tree_pred_2 = tree_clf_2.predict(X_test)
print(f'For Decision Tree classifier\n{classification_report(y_test, tree_pred_2)}')

steps = [('DTree', tree_clf)]
pipe = Pipeline(steps=steps, memory='tmp')

strategy = ['not majority']
params = {'criterion':['gini','entropy'],'splitter': ['best','random'],'max_features':['auto','sqrt','log2']}
estimator = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=10, n_jobs=-1)
estimator.fit(X_train, y_train),
dtree_best_pred = estimator.predict(X_test)
print(estimator.best_estimator_)
print(estimator.best_params_)

#print(confusion_matrix(y_test,predictions))

# import pandas as pd
# from sklearn.datasets import load_iris
# from lightgbm import LGBMRegressor
# from category_encoders import OrdinalEncoder
# 
# X = load_iris()['data']
# y = load_iris()['target']
# 
# X = OrdinalEncoder(cols=[3]).fit_transform(X)
# 
# dt = LGBMRegressor()
# dt.fit(X, y, categorical_feature=[3])

print('\n'+'======================================='+'\n')

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder #Για unordered categorical features
from sklearn.impute import SimpleImputer
#from imblearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot

import warnings
warnings.filterwarnings("ignore")






# load the dataset
#def load_dataset(filename,x_col_num_list,y_col_num):
def load_dataset(filename,x_col_num_list,y_col_num):
	# load the dataset as a pandas DataFrame
	data = pd.read_csv(filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:,x_col_num_list]
	y = dataset[:,y_col_num]
	# format all fields as string
	X = X.astype(str);y = y.astype(str)
	return X, y

# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc
 
# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc



data_set= pd.read_csv('01.Covid19MPD.(Pos_valid_bin_21).csv')  

#correlation between features
corr_plot = sns.heatmap(data.corr(),annot = True,linewidths=3 )
plt.title("Correlation plot")
plt.show()

#Extracting Independent and dependent Variable  
X= data_set.iloc[:,[3,4]].values
y= data_set.iloc[:,[20]].values

# Splitting the dataset into training and test set.  
  
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=42)

print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)

# prepare input data

scale=StandardScaler()
X_train = scale.fit_transform(x_train)
X_test = scale.transform(x_test)

#check for distribution of labels
y_train.value_counts(normalize=True)

#Fitting Logistic Regression to the training set    
classifier= LogisticRegression(random_state=0)  
classifier.fit(x_train, y_train)  
y_pred= classifier.predict(x_test)  

'''
#df_21_final = pd.read_csv('01.Covid19MPD.(Pos_valid_bin_21).csv',header=0)#[410.767 valid from 2.062.829 COVID-19 Positive samples]
#dataset = df_21_final.values
#X = dataset[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
#y = dataset[:,[20]]
#X, y = load_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv',[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18],20)
#X, y = load_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv',[5],20)
#data = np.array(data, dtype=[('06.AGE', int), ('21.SURVIVED', int)]
##print(X , '\n' + '\n', y)

# split into train and test sets
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#X_train, X_test, y_train, y_test = train_test_split(dataset.loc[:, dataset.columns != 'class'], dataset['class'], test_size=0.2, random_state=42)

'''
scaler = StandardScaler()
ros = RandomOverSampler()
'''
'''
Εφαρμόζουμε 3ο ταξινομητή, το LogisticRegression
'''
'''
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
log_pred = log_clf.predict(X_test)
print('For Logistic Regression')
print(classification_report(y_test, log_pred))

scaler = MinMaxScaler()
steps = [('sampler',ros),
        ('scaler', scaler),
         ('LogReg',log_clf)]

pipe = Pipeline(steps=steps, memory='tmp')

pipe.fit(X_train, y_train)
pipe_pred = pipe.predict(X_test)
print("Pipe Logistic Regression")
print(classification_report(y_test, pipe_pred))


C = [0.009, 0.01, 0.02]
solver = ['lbfgs']
penalty = ['l2']
strategy = ['not majority', 'all']


params = {
          'LogReg__C':C,
          'LogReg__solver':solver,
          'LogReg__penalty':penalty,
        'sampler__sampling_strategy':strategy
}
estimator = GridSearchCV(pipe, param_grid=params, cv=10, n_jobs=-1)
estimator.fit(X_train, y_train),
logreg_best_pred = estimator.predict(X_test)
print('Optimal Logistic Regression')
print(classification_report(y_test, logreg_best_pred))
print(estimator.best_params_)
print(('ok'))
# with open('log_Reg.txt', 'w') as f:
'''
'''
Βελτιστοποίηση υπερπαραμέτρων μέσω GridSearch
'''

'''
# pipe = Pipeline(steps=[('LogReg', log_clf)], memory='tmp')
#
# penalty = ['l1', 'l2', 'elasticnet', 'none']
# C = [0.01, 0.1, 1, 10, 100]
# solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# max_iter = [50, 100, 200, 500]
# tol = [1e-05, 1e-04, 1e-03, 1e-02]
# estimator = GridSearchCV(pipe, dict(LogReg__penalty=penalty, LogReg__C=C, LogReg__solver=solver, LogReg__max_iter=max_iter, LogReg__tol=tol), cv=5)
#
# estimator.fit(X_train, y_train)
# log_best_pred = estimator.predict(X_test)
# print(f'Optimized Logistic Regression\n{classification_report(y_test, log_best_pred)}')

print('ok')




#--------------------------------------------------------------------------------------


# example of mutual information feature selection for categorical data
#from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot

from sklearn.preprocessing import StandardScaler
'''
''' 
#load the dataset
def load_dataset(filename,x_col_num_list,y_col_num_list):
	# load the dataset as a pandas DataFrame
	data = pd.read_csv(filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
    y = dataset[:,-1]
	#X = dataset[x_col_num_list]
    #y = dataset[y_col_num_list]
	# format all fields as string
	X = X.astype(str)
	return X, y
'''

'''
# load the dataset
#def load_dataset(filename,x_col_num_list,y_col_num):
def load_dataset(filename,x_col_num_list,y_col_num):
	# load the dataset as a pandas DataFrame
	data = pd.read_csv(filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:,x_col_num_list]
	y = dataset[:,y_col_num]
	# format all fields as string
	X = X.astype(str);y = y.astype(str)
	return X, y

 
# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc
 
# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc
 
# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_classif, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs


dataset = pd.read_csv('01.Covid19MPD.(Pos_valid_bin_21).csv')
# dataset2 = pd.read_csv('covid_jpn_total.csv')
# dataset3 = pd.read_csv('all-states-history.csv')
'''
'''
Μία γρήγορη ματιά στο dataset με functions των pandas
'''
'''
print(dataset.head())
print(dataset.info())
print(dataset.describe())
    
#scale=StandardScaler()
#X_train = scale.fit_transform(X_train)
#X_test = scale.transform(X_test)

# load the dataset except the columns 06.AGE,20.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION
#X, y = load_dataset('01.Covid19MPD.(Pos_transf_bin_21).csv',[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18],20)
X, y = load_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv',[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18],20)
print(X , '\n' + '\n', y)

# split into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,train_size=0.8)
#get shape of train and test data

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#X_train, X_test, y_train, y_test = train_test_split(dataset.loc[:, dataset.columns != 'class'], dataset['class'], test_size=0.2, random_state=42)

# summarize
print("train size X : ",X_train.shape)
print("train size y : ",y_train.shape)
print("test size X : ",X_test.shape)
print("test size y : ",y_test.shape)
#print('Train', X_train.shape, y_train.shape)
#print('Test', X_test.shape, y_test.shape)

# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)

# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)

# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))

# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
'''


'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
import time

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder #Για unordered categorical features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report,confusion_matrix,precision_recall_fscore_support
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

import warnings
warnings.filterwarnings("ignore")
'''

#from my_functions import *
from my_functions import feature_correlation_std
from my_functions import prepare_dataset
from my_functions import model_create_train_pred_analysis
from my_functions import find_model_opt_param

'''Functions'''
'''
#Show a Dataframe's Feature Correlation(Pearson-Spearman-Kendall)
def feature_correlation_std(df_csv_file,y_column_title):
    
    dataframe = pd.read_csv(df_csv_file)
    
    pearson_corr = dataframe.corr(method ='pearson')
    spearman_corr = dataframe.corr(method ='spearman')
    kendall_corr = dataframe.corr(method ='kendall')
    
    print('\n' + '===============[Pearson Correllation]===============' + '\n',
          pearson_corr[y_column_title].sort_values(ascending=False),'\n' +
          '====================================================' + '\n' +
          '\n' + '==============[Spearman Correllation]===============' + '\n',
          spearman_corr[y_column_title].sort_values(ascending=False),'\n' +
          '====================================================' + '\n' +
          '\n' + '===============[Kendall Correllation]===============' + '\n',
          kendall_corr[y_column_title].sort_values(ascending=False),'\n' +
          '====================================================' + '\n',)


#Create the Train & Test sets fron a dataframe
def prepare_dataset(df_csv_file,x_column_list,y_column_list,rand_state,train_sz,dataset_name):
    
    dataframe = pd.read_csv(df_csv_file)
    dataset = dataframe.values
    
    X = dataset[:,x_column_list]
    X = X.astype('int')
    
    y = dataset[:,y_column_list]
    y = y.astype('int')
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=rand_state,train_size=train_sz)
    
    print('\n'+'================[' + dataset_name + ']================' + '\n' +
          'train size X :',X_train.shape,'\n' + 
          'test size X :',X_test.shape,'\n' +
          'train size y :',y_train.shape,'\n' + 
          'test size y :',y_test.shape,'\n' +
          '==============================================' + '\n')
    
    return X_train,X_test,y_train,y_test


#Create a Classifier Model with default or specific Hyperparameters
def create_model(method_name,param_dict):
    
    method='none'
    
    if  method_name == 'lr':
        model = LogisticRegression(**param_dict)
        method = 'Logistic Regression'
    
    elif method_name == 'dt':
        model = DecisionTreeClassifier(**param_dict)
        method = 'Decision Tree'
    
    elif method_name == 'rf':
        model = RandomForestClassifier(**param_dict)
        method = 'Random Forest'
    
    elif method_name == 'kn':
        model = KNeighborsClassifier(**param_dict)
        method = 'KNeighbors'
    
    elif method_name == 'kms':
        model = KMeans(**param_dict)
        method = 'KMeans'
    
    elif method_name == 'mlp':
        model = MLPClassifier(**param_dict)
        method = 'MLPClassifier'
    
    elif method_name == 'svm':
        model = SVC(**param_dict)
        method = 'Support Vector Machine Classifier'
    
    return model,method


#Create Train, Predict & Show the statistic results of a Model with default or specific Hyperparameters
def model_create_train_pred_analysis(X_train,X_test,y_train,y_test,method_name,param_dict,dataset_name):
    
    start = time.time()
    
    model,method = create_model(method_name,param_dict)
    
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    
    print('\n'+'================' + method + '(' + dataset_name + ')================' + '\n' +
          '--------------------[Classification Report]--------------------' + '\n' + '\n' ,
          classification_report(y_test,y_pred), '\n' +
          '---------------------------------------------------------------' + '\n' +
          'Precision score:',precision_score(y_test,y_pred,pos_label='positive',average='micro'), '\n' +
          'Recall score:',recall_score(y_test,y_pred), '\n' +
          'Accuracy score:',accuracy_score(y_test, y_pred), '\n' +
          'F1 Score:',f1_score(y_test,y_pred), '\n' +
          '---------------------------------------------------------------' + '\n' +
          'Confusion Matrix:' + '\n' ,confusion_matrix(y_test,y_pred), '\n' + 
          '---------------------------------------------------------------' + '\n' +
          'Model Parameters:' + '\n' ,model.get_params(), '\n' +
          '---------------------------------------------------------------' + '\n')

    end = time.time()
    print(f'Runtime: {end-start} seconds\n' + 
          '===============================================================' + '\n')


#Extract the Classification Report's metrics
def classification_report_opt_metrics(estimator,y_test,optimum_model_y_pred):
    
    cl_report = classification_report(y_test,optimum_model_y_pred)
    
    precision_sc = precision_score(y_test,optimum_model_y_pred,pos_label='positive',average='micro')
    recall_sc = recall_score(y_test,optimum_model_y_pred)
    accur_sc = accuracy_score(y_test, optimum_model_y_pred)
    f1_sc = f1_score(y_test,optimum_model_y_pred)
    
    conf_mtrx = confusion_matrix(y_test,optimum_model_y_pred)
    
    bst_params = estimator.best_params_
    
    return cl_report,precision_sc,recall_sc,accur_sc,f1_sc,conf_mtrx,bst_params


#Extract the Optimal Hyperparameter Classification Report's text as String
def create_opt_params_text(method_name,dataset_name,param_grid_dict,cl_report,precision_sc,recall_sc,accur_sc,f1_sc,conf_mtrx,bst_params):
    
    report_text = f'====[Optimum {method_name} Results & Hyperparameters({dataset_name})]====\n\nParameter Grid Used:\n{param_grid_dict}\n\n--------------------[Classification Report]--------------------\n{cl_report}\n---------------------------------------------------------------\nPrecision score:{precision_sc}\nRecall score:{recall_sc}\nAccuracy score:{accur_sc}\nF1 Score:{f1_sc}\n---------------------------------------------------------------\nConfusion Matrix:\n{conf_mtrx}\n---------------------------------------------------------------\nOptimal Parameters:\n{bst_params}\n---------------------------------------------------------------\n'
    
    return report_text


#Write the Optimal Hyperparameter Classification Report's String to a *.txt File(and return the duration{end-start})
def create_write_in_text_file(method_name,report_text,start):
    
    duration_text = ''
    
    with open(method_name + '.txt', 'a+') as file_object:
        
        file_object.seek(0)
        
        data = file_object.read(100)
        
        if len(data) > 0 :
            
            file_object.write('\n'+'\n')
            
        file_object.write(report_text)
        
        end = time.time()
        
        duration_text = f'Runtime: {end-start} seconds\n===============================================================\n'
        
        file_object.write(duration_text)
    
    return duration_text


#Find the Optimal Hyperparameters for a Model
def find_model_opt_param(X_train,X_test,y_train,y_test,method,param_grid_dict,dataset_name):
    
    start = time.time()
    
    model,method_name = create_model(method,{})

    pipe = Pipeline(steps=[(method, model)], memory='tmp')

    new_param_grid_dict = {}

    for key in param_grid_dict:
        
        curr_new_key_name = ''
        curr_new_key_name = method+'__'+str(key)
        new_param_grid_dict[curr_new_key_name] = param_grid_dict[key]
    
    estimator = GridSearchCV(pipe,param_grid=new_param_grid_dict,cv=10,n_jobs=-1)
    #estimator = GridSearchCV(model,param_grid=param_grid_dict,cv=10,n_jobs=-1)
    
    estimator.fit(X_train, y_train)
    optimum_model_y_pred = estimator.predict(X_test)

    cl_report,precision_sc,recall_sc,accur_sc,f1_sc,conf_mtrx,bst_params = classification_report_opt_metrics(estimator,y_test,optimum_model_y_pred)
    
    report_text = create_opt_params_text(method_name,dataset_name,param_grid_dict,cl_report,precision_sc,recall_sc,accur_sc,f1_sc,conf_mtrx,bst_params)
    
    print(report_text)
    
    duration_text = create_write_in_text_file(method_name,report_text,start)
    
    print(duration_text)
'''
''''''




'''Functions'''
'''
#Show the number of NaN and Non Unique values of a column
def show_column_info(df,column_name):
    
    row_number = len(df.index)
    null_count = df[column_name].isnull().sum()
    unique_count = df[column_name].nunique()
    non_unique_count = row_number - (null_count + unique_count)
    data_type = df[column_name].dtypes
        
    print('\n' + column_name + ' [' + str(data_type) + ']' +
          '\n' + 'has ' + str(row_number) + ' rows, containing:' +
          '\n' + str(null_count) + ' NaN Values' +
          '\n' + str(unique_count) + ' Unique Values' +
          '\n' + str(non_unique_count) + ' Non Unique Values' +
          '\n')


#Show the number of NaN and Non Unique values in all the columns of a dataframe
def show_all_column_info(df):
    
    column_name_list = list(df.columns.values)

    for column_name in column_name_list:
    
        show_column_info(df,column_name)


#Analysis of a dataframe column with binary values
def binary_column_analysis(df,column_name,binary_name1,binary_name2):
    
    binary_name1_count= (df[column_name]==1).sum()
    binary_name2_count= (df[column_name]==2).sum()
    does_not_apply_count = (df[column_name]==97).sum()
    ignored_count = (df[column_name]==98).sum()
    not_specified_count = (df[column_name]==99).sum()
    row_number = len(df.index)
    null_count = df[column_name].isnull().sum()
    data_type = df[column_name].dtypes

    print('\n' + column_name + ' [' + str(data_type) + ']' +
          '\n' + 'has ' + str(row_number) + ' rows, containing:' +
          '\n' + str(binary_name1_count) + ' '+ binary_name1 +
          '\n' + str(binary_name2_count) + ' '+ binary_name2 +
          '\n' + str(does_not_apply_count) + ' Do not apply' +
          '\n' + str(ignored_count) + ' Ignored' +
          '\n' + str(not_specified_count) + ' Not Specified' +
          '\n' + str(null_count) + ' NaN values' +
          '\n')


#Analysis of a dataframe column with categorical values
def categorical_column_analysis(df,column_name,var_name_value_list,boolean_extra):
    
    data_type = df[column_name].dtypes
    row_number = len(df.index)
    null_count = df[column_name].isnull().sum()
    result_list = []
    extra_list = [['Do not apply',97],['Ignored',98],['Not Specified',99]]
    
    if boolean_extra == 1:
        var_name_value_list.append(extra_list[0]); var_name_value_list.append(extra_list[1]); var_name_value_list.append(extra_list[2])
                                                              
    for i in range(len(var_name_value_list)):
        
        var_name = var_name_value_list[i][0]
        var_value=var_name_value_list[i][1]
        var_name_count = (df[column_name]== var_value).sum()
        result_list.append([var_name, var_value, var_name_count])

    print('\n' + column_name + ' [' + str(data_type) + ']' +
          '\n' + 'has ' + str(row_number) + ' rows, containing:')

    for j in range(len(result_list)):

        var_name_r = result_list[j][0]
        var_value_r =result_list[j][1]
        var_name_count_r =result_list[j][2]

        print(str(var_name_count_r) + ' '+ str(var_name_r) + '(' + str(var_value_r) + ')')

    print(str(null_count) + ' NaN values' +
          '\n')


#Create a barchart from a dataframe column with categorical values
def create_categorical_bar_chart(df,column_name,var_value_list,var_name_list,chart_title):

    all_values_quantity_list=[]
    
    for i in range(len(var_value_list)):

        #value_quantity = int(df.loc[df[column_name]==var_value_list[i], :][column_name].count())
        
        df_filter = df[df[column_name]==var_value_list[i]]

        filter_index = df_filter.index
        value_quantity = len(filter_index)
        
        value_quantity_list = [value_quantity]
        all_values_quantity_list.append(value_quantity_list)
    
    index = [column_name]
    
    my_dictionary={}
    
    for j in range(len(all_values_quantity_list)):
        
        my_dictionary[var_name_list[j]] =  all_values_quantity_list[j]
    
    
    df = pd.DataFrame(my_dictionary, index=index)
    
    bar_chart = df.plot.bar(title=chart_title)
    stacked_bar_chart = df.plot.bar(title=chart_title,stacked=True)


#Create a barchart from a dataframe column with categorical values (df,column_name,var_value_list,var_name_list):
def create_a_df_categorical_column_bar_chart(df,column,chart_title):
    
    my_dictionary={}
    index = [column.my_name]
    
    for key in column.characteristics_dict:

        #value_quantity = int(df.loc[df[column.name]==column.characteristics_dict[key], :][column.name].count())
        
        df_filter = df[df[column.name]==column.characteristics_dict[key]]

        filter_index = df_filter.index
        value_quantity = len(filter_index)
        
        my_dictionary[key] = value_quantity
    
    my_dictionary['NaN'] = df[column.name].isnull().sum()
    df = pd.DataFrame(my_dictionary, index=index)

    
    bar_chart = df.plot.bar(title=chart_title)
    stacked_bar_chart = df.plot.bar(title=chart_title,stacked=True)


#Create a barchart from a dataframe list of columns with categorical values (df,column_name,var_value_list,var_name_list):
def create_multi_df_categorical_column_bar_chart(df,column_list,col_val_list,chart_title):
    
    my_dictionary = dict.fromkeys(col_val_list)
    
    for key in my_dictionary:
        curr_list=[]
        my_dictionary[key]=curr_list

    index = []

    for col in column_list:

        index.append(col.my_name)
        
        for key in col.characteristics_dict:
            
            #value_quantity = int(df.loc[df[col.name]==col.characteristics_dict[key], :][col.name].count())
            
            df_filter = df[df[col.name]==col.characteristics_dict[key]]

            filter_index = df_filter.index
            value_quantity = len(filter_index)
            
            my_dictionary[key].append(value_quantity)
            
    df = pd.DataFrame(my_dictionary, index=index)

    bar_chart = df.plot.bar(title=chart_title)
    stacked_bar_chart = df.plot.bar(title=chart_title,stacked=True)


#Create a barchart from a dataframe list of columns with categorical values as an addage to the above function
def create_multi_df_categorical_column_list_chart(df,col_dict,chart_title):
    my_column_values_list=['yes','no','not apply','ignored','not specified']
    dict_common ={'yes':1,'no':2,'not apply':97,'ignored':98,'not specified':99}
    my_col_list=[]
    
    for col in col_dict:
        my_col_list.append(Column(str(col),str(col_dict[col]),dict_common))
    
    create_multi_df_categorical_column_bar_chart(df,my_col_list,my_column_values_list,chart_title)


#Create a barchart from a dataframe list of columns with numerical & categorical values (df,column_list_a,col_val_list_a,column_list_b,col_val_list_b):
def create_multi_df_numerical_column_bar_chart(df,column_list_a,col_val_list_a,column_list_b,col_val_list_b,chart_title):
    
    my_dict = dict.fromkeys(col_val_list_b)
    
    for key in my_dict:
        curr_list=[]
        my_dict[key]=curr_list

    index = []
    
    for col_a in column_list_a:

        index.append(col_a.my_name)
        
        for key_a in col_a.characteristics_dict:
            
            for col_b in column_list_b:
                
                for key_b in col_b.characteristics_dict:
                    
                    df_filter = df[(df[col_a.name] >= col_a.characteristics_dict[key_a][0]) & 
                                            (df[col_a.name] <= col_a.characteristics_dict[key_a][1]) &
                                            (df[col_b.name]==col_b.characteristics_dict[key_b])]

                    filter_index = df_filter.index
                    value_quantity = len(filter_index)

                    my_dict[key_b].append(value_quantity)
            
    df = pd.DataFrame(my_dict, index=index)

    bar_chart = df.plot.bar(title=chart_title)
    stacked_bar_chart = df.plot.bar(title=chart_title,stacked=True)

#Ages Column Analysis
def ages_column_analysis(df,col_name):
    print(df[[col_name]].describe())
    print(((df[col_name])< 0).sum(),'Invalid values')
    print(df[col_name].isnull().sum(),'NaN values')

#Create Ages & Sexes Barchart
def ages_column_barchart(df,col_ages_name,col_sexes_name,chart_title):
    column_sexes_values_list= ['women','men','not apply','ignored','not specified']
    column_ages_values_list= ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99','100 and over']
    
    sexes_column_list =[Column(col_sexes_name,'Sexes',{'women':1,'men':2,'not apply':97,'ignored':98,'not specified':99})]
    ages_column_list =[Column(col_ages_name,'0-9',{'0-9':[0,9]}),Column(col_ages_name,'10-19',{'10-19':[10,19]}),
                       Column(col_ages_name,'20-29',{'20-29':[20,29]}),Column(col_ages_name,'30-39',{'30-39':[30,39]}),
                       Column(col_ages_name,'40-49',{'40-49':[40,49]}),Column(col_ages_name,'50-59',{'50-59':[50,59]}),
                       Column(col_ages_name,'60-69',{'60-69':[60,69]}),Column(col_ages_name,'70-79',{'70-79':[70,79]}),
                       Column(col_ages_name,'80-89',{'80-89':[80,89]}),Column(col_ages_name,'90-99',{'90-99':[90,99]}),
                       Column(col_ages_name,'100 and over',{'100 and over':[100,500]})]
    
    create_multi_df_numerical_column_bar_chart(df,ages_column_list,column_ages_values_list,sexes_column_list,column_sexes_values_list,chart_title)

#Ages Column Analysis & Create Ages & Sexes Barchart
def ages_column_analysis_and_barchart(df_first,df_last,col_ages_name_first,col_ages_name_last,col_sexes_name_first,col_sexes_name_last,chart_title):
    ages_column_analysis(df_first,col_ages_name_first)
    ages_column_barchart(df_first,col_ages_name_first,col_sexes_name_first,chart_title+' 2M(29_transf)')
    ages_column_analysis(df_last,col_ages_name_last)
    ages_column_barchart(df_last,col_ages_name_last,col_sexes_name_last,chart_title+' 400K(21_valid)')
    

#Analyze a categorical values df column and create the corresponding barchart 
def categorical_column_analysis_and_bar_chart(df_first,df_last,col_first,col_last,column_values_keys_list,chart_title):
    categorical_column_analysis(df_first,col_first.name,column_values_keys_list,1)
    categorical_column_analysis(df_last,col_last.name,column_values_keys_list,1)
    create_a_df_categorical_column_bar_chart(df_first,col_first,chart_title+' 2M(29_transf)')
    create_a_df_categorical_column_bar_chart(df_last,col_last,chart_title+' 400K(21_valid)')
    
    
#Return a Dataframe with all the rows than contain all the valid and Not NaN values for a single column
def filter_values_of_a_column(df,df_col_name,invalid_val_list):
    
    df_not_null =df
    
    df_not_null = df[df[df_col_name].notnull()]
    
    df_all_valid = df_not_null[df_not_null[df_col_name].isin(invalid_val_list)== False]
    
    return df_all_valid


#Return a Dataframe with all the rows than contain all the valid and Not NaN values for a column list
def filter_values_of_multi_columns(df,df_col_name_list,invalid_val_list):
    
    final_df = df
    
    for col_name in df_col_name_list:
        
        final_df = filter_values_of_a_column(final_df,col_name,invalid_val_list)

    return final_df

'''
''''''




'''
df_sars_cov_2_p_all_transf_valid_21_en_num = pd.read_csv('01.Covid19MPD.(Pos_valid_21).csv',header=0)

#Create a csv with the Dataframe that contains all the COVID-19 Positive samples
#and without the columns 19.LAB_RESULT,20.FINAL_CLASSIFICATION
#---21.TRANFORMED.ALL.(2.062.829)-----------------------------
df_sars_cov_2_p_all_transf_21_en_num = pd.read_csv('01.Covid19MPD.(Pos_transf_21).csv',header=0)#[2.062.829 COVID-19 Positive samples from 12.425.181 total samples]
'''
'''Create a barchart for all binary columns for Valid samples[410.362 Valid COVID-19 Positive samples]'''
'''
col_21_dict={'03.OUTPATIENT':'outpatient','04.INTUBATED':'intubated','05.PNEUMONIA':'pneumonia','07.PREGNANCY':'pregnant',
 '08.DIABETIC':'diabetic','09.COPD':'copd','10.ASTHMA':'asthma','11.IMMUNOSUPPRESSED':'immunosuppressed',
 '12.HYPERTENSION':'hypertension','13.OTHER_CHRONIC_DISEASE':'other chronic disease','14.CARDIOVASCULAR':'cardiovascular',
 '15.OBESITY':'obesity','16.CHRONIC_KIDNEY_FAILURE':'chronic kidney failure','17.SMOKER':'smoker',
 '18.CONTACT_WITH_COVID-19_CASE':'contact with COVID-19 case','19.ICU':'icu','21.SURVIVED':'survived'}

create_multi_df_categorical_column_list_chart(df_sars_cov_2_p_all_transf_21_en_num,col_21_dict,'All Distributions 2M (Valid)')
'''
''''''

'''03.TYPE_OF_PATIENT(Renamed 03.OUTPATIENT)'''
'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_21_en_num,df_sars_cov_2_p_all_transf_21_en_num,
                                          Column('03.OUTPATIENT','Type of Patient', {'Outpatient':1,'Inpatient':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('03.OUTPATIENT','Type of Patient', {'Outpatient':1,'Inpatient':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Outpatient-Ambulatory',1],['Inpatient-Hospitalized',2]],'03.Types of Patients')
'''
'''07.INTUBATED(Renamed 04.INTUBATED)'''
'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_21_en_num,df_sars_cov_2_p_all_transf_21_en_num,
                                          Column('04.INTUBATED','Intubations', {'Intubated':1,'Not Intubated':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('04.INTUBATED','Intubations', {'Intubated':1,'Not Intubated':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Intubated',1],['Not Intubated',2]],'04.Intubated Distribution')
'''
'''27.ICU(Renamed 19.ICU)'''
'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_21_en_num,df_sars_cov_2_p_all_transf_21_en_num,
                                          Column('19.ICU','ICU', {'ICU':1,'Not ICU':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('19.ICU','ICU', {'ICU':1,'Not ICU':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['ICU',1],['Not ICU',2]],'19.ICU Distribution')
'''

'''Feature Correlation(Pearson-Spearman-Kendall)'''
'''
feature_correlation_std('01.Covid19MPD.(Pos_valid_bin_21).csv','21.SURVIVED')
'''


'''Train & Test sets'''

'''Default_features[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]'''
'''
X_train,X_test,y_train,y_test = prepare_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv',
                                                                [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                                                                 20,42,0.7)
'''
'''All_features[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]'''
'''
X_train_all,X_test_all,y_train_all,y_test_all = prepare_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv',
                                                                [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                                                                 20,42,0.7)
'''
'''Selected_features_01 [1,3,5,8,15]'''
'''
X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01 = prepare_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv',
                                                                            [1,3,5,8,15],
                                                                            20,42,0.7)
'''


'''Find Optimal Hyperparameters'''

'''Logistic Regression'''
'''
#Tester
find_log_reg_opt_param('Tester',X_train,X_test,y_train,y_test,
                       [0.005],[50,75],['l2','none'],['sag'],[0.001])

#Default_features
find_log_reg_opt_param('Default',X_train,X_test,y_train,y_test,
                       [0.001,0.005,0.01],[50,75,100],['l2','none'],['sag'],[0.01,0.001,0.0001])

#All_features
find_log_reg_opt_param('All_features',X_train_all,X_test_all,y_train_all,y_test_all,
                       [0.001,0.005,0.01],[50,75,100],['l2','none'],['sag'],[0.01,0.001,0.0001])

#Selected_features_01 [1,3,5,8,15]
find_log_reg_opt_param('Selected_features_01',X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01,
                       [0.001,0.005,0.01],[50,75,100],['l2','none'],['sag'],[0.01,0.001,0.0001])

'''

'''Decision Tree Classifier'''
'''
#Tester
find_dtree_opt_param('Tester',X_train,X_test,y_train,y_test,
                     ['gini','entropy'],[15,25],['auto','sqrt'],
                     [42],['best'])

#Default_features
find_dtree_opt_param('Default',X_train,X_test,y_train,y_test,
                     ['gini','entropy'],[15,25,35],['auto','sqrt','log2',None],
                     [42],['best','random'])

#All_features
find_dtree_opt_param('All_features',X_train_all,X_test_all,y_train_all,y_test_all,
                     ['gini','entropy'],[15,25,35],['auto','sqrt','log2',None],
                     [42],['best','random'])

#Selected_features_01 [1,3,5,8,15]
find_dtree_opt_param('Selected_features_01',X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01,
                     ['gini','entropy'],[15,25,35],['auto','sqrt','log2',None],
                     [42],['best','random'])
'''

'''
#Run & Show the statistic results of Logistic Regression with certain hyperparameters
def log_reg_std(name,X_train,X_test,y_train,y_test,c_val,max_iter_val,penalty_val,solver_val,tol_val):
    
    print('\n'+'================Logistic Regression(' + name + ')================'+'\n')
    
    model = LogisticRegression(C=c_val,max_iter=max_iter_val,penalty=penalty_val,solver=solver_val,tol=tol_val)
    model.fit(X_train,y_train)
    
    predictions = model.predict(X_test)
    
    print('\n',classification_report(y_test,predictions),'\n',confusion_matrix(y_test,predictions),'\n' + '\n' +
          '===============================================================')


#Find the Optimal Logistic Regression's Hyperparameters for a dataframe's train and test sets 
def find_log_reg_opt_param(name,X_train,X_test,y_train,y_test,c_list,m_itr_list,pen_list,solv_list,tol_list):
    
    optimum_model = LogisticRegression()
    #scaler = MinMaxScaler()
    #steps = [('sampler',ros),('scaler', scaler),('LogReg',log_clf)]
    #pipe = Pipeline(steps=steps, memory='tmp')
    pipe = Pipeline(steps=[('LogReg', optimum_model)], memory='tmp')
    #C = c_list #C = [0.01,0.1,1,10,100]
    #max_iter = m_itr_list #max_iter = [50,75,100,125,150]
    #penalty = pen_list #penalty = ['l1','l2','elasticnet','none']
    #solver = solv_list #solver = ['newton-cg','lbfgs','liblinear','sag','saga']
    #tol = tol_list #tol = [1e-03,1e-02,1e-04,1e-05]
    #params = {'C':c_list,'max_iter':m_itr_list,'penalty':pen_list,'solver':solv_list,'tol':tol_list}
    #param_grid=params
    estimator = GridSearchCV(pipe,
                             dict(LogReg__C=c_list,LogReg__max_iter=m_itr_list,LogReg__penalty=pen_list, 
                                  LogReg__solver=solv_list,LogReg__tol=tol_list),
                             cv=5)

    estimator.fit(X_train, y_train)
    optimum_model_pred= estimator.predict(X_test)
    #with open('log_Reg.txt', 'w') as f:
    print('\n' + '====[Optimum Logistic Regression Results & Hyperparameters(' + name + ')]====' + '\n' + '\n',
          classification_report(y_test,optimum_model_pred),'\n' + '\n',
          estimator.best_estimator_,'\n' + '\n',
          estimator.best_params_,'\n' + '\n',
          confusion_matrix(y_test,optimum_model_pred),'\n' +
          '\n' + '======================================================================='+'\n')


#Run & Show the statistic results of Decision Tree Classifier with certain hyperparameters
def dtree_std(name,X_train,X_test,y_train,y_test,crit_val,m_depth_val,m_features_val,rand_st_val,split_val):
    
    print('\n'+'================Decision Tree(' + name + ')================'+'\n')
    
    model = DecisionTreeClassifier(criterion=crit_val,max_depth=m_depth_val,max_features=m_features_val,random_state=rand_st_val,splitter=split_val)
    model.fit(X_train,y_train)
    
    predictions = model.predict(X_test)
    
    print('\n',classification_report(y_test,predictions),'\n',confusion_matrix(y_test,predictions),'\n' + '\n' +
          '===============================================================')


#Find the Optimal Decision Tree Classifier's Hyperparameters for a dataframe's train and test sets 
def find_dtree_opt_param(name,X_train,X_test,y_train,y_test,crit_list,m_depth_list,m_features_list,rand_st_list,split_list):
    
    optimum_model = DecisionTreeClassifier()
    #scaler = MinMaxScaler()
    #steps = [('sampler',ros),('scaler', scaler),('LogReg',log_clf)]
    #pipe = Pipeline(steps=steps, memory='tmp')
    pipe = Pipeline(steps=[('DTree', optimum_model)], memory='tmp')
    #criterion = crit_list #criterion = ['gini','entropy']
    #max_depth = m_depth_list #max_depth = [15,25,35]
    #max_features = m_features_list #max_features = ['auto','sqrt','log2',None] or int, float or
    #random_state = rand_st_list #random_state = [None] or int
    #splitter = split_list #splitter = ['best','random']
    #params = {'criterion':crit_list,'max_depth':m_depth_list,'max_features':m_features_list,'random_state':rand_st_list,'splitter':split_list}
    #param_grid=params
    estimator = GridSearchCV(pipe,
                             dict(DTree__criterion=crit_list,DTree__max_depth=m_depth_list,
                                  DTree__max_features=m_features_list,DTree__random_state=rand_st_list,
                                  DTree__splitter=split_list),
                                  cv=10,n_jobs=-1)
    
    estimator.fit(X_train, y_train)
    optimum_model_pred= estimator.predict(X_test)
    #with open('log_Reg.txt', 'w') as f:
    print('\n' + '====[Optimum Decision Tree Results & Hyperparameters(' + name + ')]====' + '\n' + '\n',
          classification_report(y_test,optimum_model_pred),'\n' + '\n',
          estimator.best_estimator_,'\n' + '\n',
          estimator.best_params_,'\n' + '\n',
          confusion_matrix(y_test,optimum_model_pred),'\n' +
          '\n' + '======================================================================='+'\n')

'''

'''Random Forest'''

print('\n'+'\n'+'==========Random Forest==========')

rand_forest_clf = RandomForestClassifier()
rand_forest_clf.fit(X_train, y_train)
rand_pred = rand_forest_clf.predict(X_test)
print(f'For Random Forest\n{classification_report(y_test, rand_pred)}')

# pipe = Pipeline(steps=[('RF', rand_forest_clf)], memory='tmp')
# n_estimators = [1,2,3,4,5]
# min_samples_split = [9]
# estimator = GridSearchCV(pipe, dict(RF__n_estimators=n_estimators, RF__min_samples_split=min_samples_split), cv=5, scoring='f1_macro')
# estimator.fit(X_train, y_train)
# rf_best_pred = estimator.predict(X_test)
# print(f'Optimized Random Forest\n{classification_report(y_test, rf_best_pred)}')
# print(estimator.best_params_)

#print(confusion_matrix(y_test,predictions))

print('\n'+'\n'+'=======================================')

'''KNeighbors'''

#print('\n'+'\n'+'==========KNeighbors==========')
#knn_clf = KNeighborsClassifier(n_neighbors=1)
#knn_clf.fit(X_train, y_train)
#y_pred = knn_clf.predict(X_test)
#print('KNN Classifier')
#print(classification_report(y_test, y_pred))

# k = [9, 10, 11]
# weights = ['uniform', 'distance']
# steps = [('sampler', ros),
#          ('KNN',knn_clf)]
# pipe = Pipeline(steps=steps, memory='tmp')
# strategy = ['not majority']
#
#
# params = {'sampler__sampling_strategy':strategy,
#           'KNN__n_neighbors':k,
#           'KNN__weights':weights}
# estimator = GridSearchCV(pipe, param_grid=params, cv=10, scoring='f1_macro', n_jobs=-1)
# estimator.fit(X_train, y_train)
# y_best_pred = estimator.predict(X_test)
# print(classification_report(y_test, y_best_pred))
# print(estimator.best_estimator_)
# print(estimator.best_params_)

#print(confusion_matrix(y_test,predictions))

print('\n'+'\n'+'=======================================')

'''Support Vector Machine(SVM)'''

print('\n'+'\n'+'==========Support Vector Machine(SVM)==========')

svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
print('For SVM')
print(classification_report(y_test, svm_pred))
# scaler = MinMaxScaler()
# steps = [('scaler', scaler),
#         ('sample', ros),
#          ('SVM', svm_clf)]
# pipe = Pipeline(steps=steps, memory='tmp')
#
# pipe.fit(X_train, y_train)
# pipe_pred = pipe.predict(X_test)
# print('Pipe SVM')
# print(classification_report(y_test, pipe_pred))
#
# C = [0.1, 1, 10]
# kernel = ['linear']
# gamma= ['scale', 'auto']
# class_weight = ['balanced']
# #'SVM__gamma': gamma
# params = {'SVM__C':C,
#           'SVM__kernel':kernel,
#             'SVM__class_weight':class_weight
# }
# estimator = GridSearchCV(pipe, param_grid=params, cv=10, scoring='precision', n_jobs=-1)
#
# estimator.fit(X_train, y_train)
# svm_best_pred = estimator.predict(X_test)
# conf_matrix = confusion_matrix(y_test, svm_best_pred)
# print('Optimized SVM')
# print(classification_report(y_test, svm_best_pred))
#
# print(estimator.best_params_)
# print(conf_matrix)

print('\n'+'\n'+'=======================================')