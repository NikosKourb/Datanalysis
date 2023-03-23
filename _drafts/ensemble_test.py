import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from my_functions import feature_correlation_std
from my_functions import prepare_dataset
from my_functions import model_create_train_pred_analysis
from my_functions import find_model_opt_param
from my_functions import create_multi_df_categorical_column_list_chart,create_multi_df_categorical_column_bar_chart,create_multi_df_numerical_column_bar_chart
from my_functions import create_model,classification_report_opt_metrics



def preprocessing_dataframe(df_csv_file_name,rel_path,scaler_name):
    
    dataframe = pd.read_csv(rel_path + df_csv_file_name)
    
    le = LabelEncoder()
    
    if scaler_name != 'none':
        
        if scaler_name == 'std':
            scaler = StandardScaler()
    
        elif scaler_name == 'mm':
            scaler = MinMaxScaler(feature_range=(0,1))
        
        dataframe[['06.AGE']] = scaler.fit_transform(dataframe[['06.AGE']])
        dataframe[['20.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION']] = scaler.fit_transform(dataframe[['20.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION']])
    
    else:
        dataframe['06.AGE'] = le.fit_transform(dataframe['06.AGE'])
        dataframe['20.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION'] = le.fit_transform(dataframe['20.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION'])
    
    
    dataframe = pd.get_dummies(dataframe, columns=['07.PREGNANCY'])
    
    #dataframe = dataframe.drop(columns=['01.REGISTRATION_ID','03.OUTPATIENT'])
    
    col_list = ['02.SEX','04.INTUBATED','05.PNEUMONIA','08.DIABETIC','09.COPD','10.ASTHMA',
                   '11.IMMUNOSUPPRESSED','12.HYPERTENSION','13.OTHER_CHRONIC_DISEASE','14.CARDIOVASCULAR',
                   '15.OBESITY','16.CHRONIC_KIDNEY_FAILURE','17.SMOKER','18.CONTACT_WITH_COVID-19_CASE',
                   '19.ICU','21.SURVIVED']
    
    for col in col_list:
        
        dataframe[col] = le.fit_transform(dataframe[col])
    
    dataframe.to_csv(rel_path + '01.Covid19MPD.(Pos_valid_bin_21_scaled['+scaler_name+']_labeled).csv',index = False)
    
    return dataframe


def prepare_split_dataset(df_csv_file_name,rel_path,x_column_list,y_column_list,rand_state,train_sz,scaler_name,dataset_name):
    
    df = preprocessing_dataframe(df_csv_file_name,rel_path,scaler_name)
    
    df = df.sample(frac=1)
    
    #df.to_csv(rel_path + '01.Covid19MPD.(Pos_valid_bin_21_scaled['+scaler_name+']_labeled).csv',index = False)
    
    dataset = df.values
    
    X = dataset[:,x_column_list]
    #X = X.astype('int')
    
    y = dataset[:,y_column_list]
    y = y.astype('int')
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=rand_state,train_size=train_sz)
    
    print('\n'+'================[' + dataset_name + ']================' + '\n' +
          'train size X :',X_train.shape,'\n' + 
          'test size X :',X_test.shape,'\n' +
          'train size y :',y_train.shape,'\n' + 
          'test size y :',y_test.shape,'\n' +
          '==============================================' + '\n')
    
    return X_train,X_test,y_train,y_test,dataset,df



from sklearn.linear_model import LogisticRegression



rel_path = 'csv/'

#X_train,X_test,y_train,y_test,dataset_all,dataframe_all = prepare_split_dataset(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv',
#                                                                                            [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22],
#                                                                                             19,42,0.7,'std','All_features')

#X_train_mm,X_test_mm,y_train_mm,y_test_mm,dataset_all,dataframe_all = prepare_split_dataset(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv',
#                                                                                             [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22],
#                                                                                             19,42,0.7,'mm','All_features')

X_train_mm,X_test_mm,y_train_mm,y_test_mm,dataset_all,dataframe_all = prepare_split_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv','csv/',
                                                                                             [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22],
                                                                                             19,42,0.7,'mm','All_features')

X_train_std,X_test_std,y_train_std,y_test_std,dataset_all,dataframe_all = prepare_split_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv','csv/',
                                                                                             [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22],
                                                                                             19,42,0.7,'std','All_features')

X_train,X_test,y_train,y_test,dataset_all,dataframe_all = prepare_split_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv','csv/',
                                                                                             [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22],
                                                                                             19,42,0.7,'none','All_features')

#X_train_mm,X_test_mm,y_train_mm,y_test_mm,dataset_all,dataframe_all = prepare_split_dataset(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv',
#                                                                                             [1],
#                                                                                             19,42,0.7,'mm','All_features')

'''Random Forest'''


'''All Features'''
model_create_train_pred_analysis(X_train_mm,X_test_mm,y_train_mm,y_test_mm,'rf',{},'Default_All mm')
model_create_train_pred_analysis(X_train_mm,X_test_mm,y_train_mm,y_test_mm,'rf',
                                 {'bootstrap':True,'ccp_alpha':0.0,'class_weight':'balanced','criterion':'entropy',
                                  'max_depth':None, 'max_features':'sqrt','max_leaf_nodes':35,'max_samples':None,
                                  'min_impurity_decrease':0.0,'min_samples_leaf':1,'min_samples_split':2,
                                  'min_weight_fraction_leaf':0.0,'n_estimators':150,
                                  'oob_score':False,'random_state':None,'verbose':0,'warm_start':True},
                                 'Optimal_All_01 mm')

model_create_train_pred_analysis(X_train_std,X_test_std,y_train_std,y_test_std,'rf',{},'Default_All std')
model_create_train_pred_analysis(X_train_std,X_test_std,y_train_std,y_test_std,'rf',
                                 {'bootstrap':True,'ccp_alpha':0.0,'class_weight':'balanced','criterion':'entropy',
                                  'max_depth':None, 'max_features':'sqrt','max_leaf_nodes':35,'max_samples':None,
                                  'min_impurity_decrease':0.0,'min_samples_leaf':1,'min_samples_split':2,
                                  'min_weight_fraction_leaf':0.0,'n_estimators':150,
                                  'oob_score':False,'random_state':None,'verbose':0,'warm_start':True},
                                 'Optimal_All_01 std')

model_create_train_pred_analysis(X_train,X_test,y_train,y_test,'rf',{},'Default_All none')
model_create_train_pred_analysis(X_train,X_test,y_train,y_test,'rf',
                                 {'bootstrap':True,'ccp_alpha':0.0,'class_weight':'balanced','criterion':'entropy',
                                  'max_depth':None, 'max_features':'sqrt','max_leaf_nodes':35,'max_samples':None,
                                  'min_impurity_decrease':0.0,'min_samples_leaf':1,'min_samples_split':2,
                                  'min_weight_fraction_leaf':0.0,'n_estimators':150,
                                  'oob_score':False,'random_state':None,'verbose':0,'warm_start':True},
                                 'Optimal_All_01 none')

'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,precision_recall_fscore_support
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

model = RandomForestRegressor()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print('--------------------[Classification Report]--------------------' + '\n' + '\n' ,
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
'''
