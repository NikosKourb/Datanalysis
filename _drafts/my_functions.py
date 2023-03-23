import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp

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

import time
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

from classes.Column import Column



'''Anaconda commands in conda promt''''''
   
conda create -n py37 python=3.7  #for version 3.7
conda activate py37
conda install pip
conda install wheel
conda install pandas
conda install -c conda-forge imbalanced-learn
conda install matplotlib
pip install matplotlib

Visualizing Decision Trees:
pip install graphviz
pip install pydotplus

Installing XGBoost Regression:
pip install xgboost

'''



'''Functions'''
'''
#DataFrame(40) in English & Numbered with All 40 columns
def df_40_en_conversion(rel_path,df_file_name_common,file_extension):
    
    file_path_name = rel_path + df_file_name_common + file_extension
    
    df =  pd.read_csv(file_path_name,header=0)
    
    df_40_en = df.rename(columns={'FECHA_ACTUALIZACION':'DATE_OF_DATA_UPDATE',
                                  'ID_REGISTRO':'REGISTRATION_ID',
                                  'ORIGEN':'ORIGIN',
                                  'SECTOR':'HEALTHCARE_SECTOR',
                                  'ENTIDAD_UM':'ENTITY_OF_HEALTHCARE_UNIT',
                                  'SEXO':'SEX',
                                  'ENTIDAD_NAC':'P_BIRTHPLACE_ENTITY_HEALTHCARE_UNIT',
                                  'ENTIDAD_RES':'P_RESIDENCE_HEALTHCARE_ENTITY',
                                  'MUNICIPIO_RES':'P_RESIDENCE_MUNICIPALITY',
                                  'TIPO_PACIENTE':'TYPE_OF_PATIENT',
                                  'FECHA_INGRESO':'ADMISSION_DATE',
                                  'FECHA_SINTOMAS':'SYMPTOM_ONSET_DATE',
                                  'FECHA_DEF':'DATE_OF_DEATH',
                                  'INTUBADO':'INTUBATED',
                                  'NEUMONIA':'PNEUMONIA',
                                  'EDAD':'AGE',
                                  'NACIONALIDAD':'NATIONALITY',
                                  'EMBARAZO':'PREGNANCY',
                                  'HABLA_LENGUA_INDIG':'NATIVE_LANGUAGE_SPEAKER',
                                  'INDIGENA':'INDIGENOUS',
                                  'DIABETES':'DIABETIC',
                                  'EPOC':'COPD',
                                  'ASMA':'ASTHMA',
                                  'INMUSUPR':'IMMUNOSUPPRESSED',
                                  'HIPERTENSION':'HYPERTENSION',
                                  'OTRA_COM':'OTHER_CHRONIC_DISEASE',
                                  'CARDIOVASCULAR':'CARDIOVASCULAR',
                                  'OBESIDAD':'OBESITY',
                                  'RENAL_CRONICA':'CHRONIC_KIDNEY_FAILURE',
                                  'TABAQUISMO':'SMOKER',
                                  'OTRO_CASO':'CONTACT_WITH_COVID-19_CASE',
                                  'TOMA_MUESTRA_LAB':'LAB_SAMPLE_TAKEN', 
                                  'RESULTADO_LAB':'LAB_RESULT',
                                  'TOMA_MUESTRA_ANTIGENO':'ANTIGEN_SAMPLE_TAKEN',
                                  'RESULTADO_ANTIGENO':'ANTIGEN_RESULT',
                                  'CLASIFICACION_FINAL':'FINAL_CLASSIFICATION',
                                  'MIGRANTE':'IMMIGRANT',
                                  'PAIS_NACIONALIDAD':'NATIONALITY_COUNTRY',
                                  'PAIS_ORIGEN':'COUNTRY_OF_ORIGIN',
                                  'UCI':'ICU'})

    df_40_en.name = 'df_40_en'
    
    new_file_path_name = rel_path + df_file_name_common + '_1_40_en' + file_extension
    
    df_40_en.to_csv(new_file_path_name, index = False)
    
    return df_40_en,new_file_path_name


#DataFrame(27=40-13), without 13 columns that contain data about region & nationality
def df_27_en_conversion(rel_path,df_file_name_common,df_prev_file_path_name,file_extension):
    
    df_40_en =  pd.read_csv(df_prev_file_path_name,header=0)
    
    df_27_en = df_40_en.drop(columns=['DATE_OF_DATA_UPDATE',
                                      'ORIGIN',
                                      'HEALTHCARE_SECTOR',
                                      'ENTITY_OF_HEALTHCARE_UNIT',
                                      'P_BIRTHPLACE_ENTITY_HEALTHCARE_UNIT',
                                      'P_RESIDENCE_HEALTHCARE_ENTITY',
                                      'P_RESIDENCE_MUNICIPALITY',
                                      'NATIONALITY',
                                      'NATIVE_LANGUAGE_SPEAKER',
                                      'INDIGENOUS',
                                      'IMMIGRANT',
                                      'NATIONALITY_COUNTRY',
                                      'COUNTRY_OF_ORIGIN'])
    
    df_27_en.name = 'df_27_en'
    
    new_file_path_name = rel_path + df_file_name_common + '_2_27_en' + file_extension
    
    df_27_en.to_csv(new_file_path_name, index = False)
    
    return df_27_en,new_file_path_name

    
#Dataframe(29=27+2) with 2 extra columns DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION & SURVIVED
def df_29_en_conversion(rel_path,df_file_name_common,df_prev_file_path_name,file_extension):
    
    df_29_en =  pd.read_csv(df_prev_file_path_name,header=0)
    
    #Add column DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION
    df_29_en['ADMISSION_DATE'] = pd.to_datetime(df_29_en['ADMISSION_DATE'], format='%Y-%m-%d')
    df_29_en['SYMPTOM_ONSET_DATE'] = pd.to_datetime(df_29_en['SYMPTOM_ONSET_DATE'], format='%Y-%m-%d')
    df_29_en['DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION'] = (df_29_en['ADMISSION_DATE'] - df_29_en['SYMPTOM_ONSET_DATE']).dt.days

    #Add column SURVIVED from DATE_OF_DEATH
    df_29_en['SURVIVED'] = np.where(df_29_en['DATE_OF_DEATH']=='9999-99-99', 1, 2)
 
    df_29_en.name = 'df_29_en'
    
    new_file_path_name = rel_path + df_file_name_common + '_3_29_en' + file_extension
    
    df_29_en.to_csv(new_file_path_name, index = False)
    
    return df_29_en,new_file_path_name



#DataFrame(24=29-5), without the ADMISSION_DATE,SYMPTOM_ONSET_DATE,DATE_OF_DEATH,LAB_SAMPLE_TAKEN,ANTIGEN_SAMPLE_TAKEN columns
def df_24_en_conversion(rel_path,df_file_name_common,df_prev_file_path_name,file_extension):
    
    df_29_en =  pd.read_csv(df_prev_file_path_name,header=0)
    
    df_24_en = df_29_en.drop(columns=['ADMISSION_DATE',
                                      'SYMPTOM_ONSET_DATE',
                                      'DATE_OF_DEATH',
                                      'LAB_SAMPLE_TAKEN',
                                      'ANTIGEN_SAMPLE_TAKEN'])
    
    df_24_en.name = 'df_24_en'
    
    new_file_path_name = rel_path + df_file_name_common + '_4_24_en' + file_extension
    
    df_24_en.to_csv(new_file_path_name, index = False)
    
    return df_24_en,new_file_path_name


#Dataframe(24) that contains all the COVID-19 Positive samples according to LAB_RESULT,ANTIGEN_RESULT and FINAL_CLASSIFICATION columns
def df_24_en_pos_conversion(rel_path,df_file_name_common,df_prev_file_path_name,file_extension):
    
    df_24_en =  pd.read_csv(df_prev_file_path_name,header=0)
    
    df_24_en_pos = df_24_en[(df_24_en['LAB_RESULT'] == 1)|(df_24_en['ANTIGEN_RESULT'] == 1)|(df_24_en['FINAL_CLASSIFICATION'].isin([1,2,3]))]
    
    df_24_en_pos.name = 'df_24_en_pos'
    
    new_file_path_name = rel_path + df_file_name_common + '_5_24_en_pos' + file_extension
    
    df_24_en_pos.to_csv(new_file_path_name, index = False)
    
    return df_24_en_pos,new_file_path_name


#Dataframe(23=24-1) that contains all the COVID-19 Positive samples according to FINAL_CLASSIFICATION or LAB_RESULT column, without the ANTIGEN_RESULT column
#df_23_en_pos_fc contains all the values of df_23_en_pos_lr
def df_23_en_pos_fc_lr_conversion(rel_path,df_file_name_common,df_prev_file_path_name,file_extension,df_type):
    
    df_24_en_pos =  pd.read_csv(df_prev_file_path_name,header=0)
    
    df_23_en_pos_fc_lr = df_24_en_pos.drop(columns=['ANTIGEN_RESULT'])
    
    if df_type == 'fc':
        
        df_23_en_pos_fc_lr = df_23_en_pos_fc_lr[df_23_en_pos_fc_lr['FINAL_CLASSIFICATION'].isin([1,2,3])]
    
    elif df_type == 'lr':
        
        df_23_en_pos_fc_lr = df_23_en_pos_fc_lr[df_23_en_pos_fc_lr['LAB_RESULT'] == 1]
    
    
    df_23_en_pos_fc_lr.name = 'df_23_en_pos_' + df_type
    
    new_file_path_name = rel_path + df_file_name_common + '_6_23_en_pos_' + df_type + file_extension
        
    df_23_en_pos_fc_lr.to_csv(new_file_path_name, index = False)
    
    return df_23_en_pos_fc_lr,new_file_path_name


#Dataframe(23) that contains all the COVID-19 Positive samples according to FINAL_CLASSIFICATION or LAB_RESULT column, without the rows that contain certain invalid values
def df_23_en_pos_fc_lr_valid_conversion(rel_path,df_file_name_common,df_prev_file_path_name,file_extension,df_type,invalid_list):
    
    df_23_en_pos_fc_lr =  pd.read_csv(df_prev_file_path_name,header=0)
    
    df_23_col_list = ['SEX','TYPE_OF_PATIENT','INTUBATED','PNEUMONIA','AGE','PREGNANCY','DIABETIC','COPD','ASTHMA',
                      'IMMUNOSUPPRESSED','HYPERTENSION','OTHER_CHRONIC_DISEASE','CARDIOVASCULAR','OBESITY',
                      'CHRONIC_KIDNEY_FAILURE','SMOKER','CONTACT_WITH_COVID-19_CASE','LAB_RESULT',
                      'FINAL_CLASSIFICATION','ICU','DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION','SURVIVED']

    df_23_en_pos_fc_lr_valid = filter_values_of_multi_columns(df_23_en_pos_fc_lr,df_23_col_list,invalid_list)
    
    df_23_en_pos_fc_lr_valid.name = 'df_23_en_pos_' + df_type + '_valid'
    new_file_path_name = rel_path + df_file_name_common + '_7_23_en_pos_' + df_type + '_valid' + file_extension
    
    df_23_en_pos_fc_lr_valid.to_csv(new_file_path_name, index = False)

    return df_23_en_pos_fc_lr_valid,new_file_path_name
'''
'''
#Show the number of NaN and Non Unique values of a column
def show_column_info(df,column_name):
    
    row_number = len(df.index)
    null_count = df[column_name].isnull().sum()
    unique_count = df[column_name].nunique()
    non_unique_count = row_number - (null_count + unique_count)
    data_type = df[column_name].dtypes
        
    print('\n' + '-----[' + df.name + ']-----' +
          '\n' + 'Column ' + column_name + ' [' + str(data_type) + ']' +
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


#Analysis of a df column with categorical values
def categorical_column_analysis(df,column):
    
    result_list = []
    data_type = df[column.name].dtypes
    row_number = len(df.index)
    null_count = df[column.name].isnull().sum()
    
    for char_key in column.characteristics_dict:
        
        char_name = char_key
        char_value=column.characteristics_dict[char_key]
        char_quantity = (df[column.name]== char_value).sum()
        result_list.append([char_name,char_value,char_quantity])
        
    print('\n' + '-----[' + df.name + ']-----' +
          '\n' + 'Column ' + column.name + ' [' + str(data_type) + ']' +
          '\n' + 'has ' + str(row_number) + ' rows, containing:')

    for j in range(len(result_list)):

        var_name_r = result_list[j][0]
        var_value_r =result_list[j][1]
        var_name_count_r =result_list[j][2]

        print(str(var_name_count_r) + ' '+ str(var_name_r) + '(' + str(var_value_r) + ')')

    print(str(null_count) + ' NaN values' +
          '\n' + '--------------------' + '\n')

''''''
#Analysis of a dataframe column with categorical values
def categorical_column_analysis(df,column_name,var_name_value_list,boolean_extra):
    
    data_type = df[column_name].dtypes
    row_number = len(df.index)
    null_count = df[column_name].isnull().sum()
    result_list = []
    extra_list = [['Do not apply',97],['Ignored',98],['Not Specified',99]]
    
    if boolean_extra == 1:
        
        for var in range(len(var_name_value_list)):
        
            var_name_value_list.append(extra_list[var])#; var_name_value_list.append(extra_list[1]); var_name_value_list.append(extra_list[2])
                                                              
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
''''''


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
def create_categorical_column_bar_chart(df,column,chart_title):
    
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
def create_multi_categorical_column_list_chart(df,col_dict,chart_title):
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


#Numerical Column Analysis
def numerical_column_analysis(df,col_name):
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
def ages_column_analysis_and_barchart(df1,col_ages_name1,col_sexes_name1,chart_title1,df2,col_ages_name2,col_sexes_name2,chart_title2):
    numerical_column_analysis(df1,col_ages_name1)
    ages_column_barchart(df1,col_ages_name1,col_sexes_name1,chart_title1)
    numerical_column_analysis(df2,col_ages_name2)
    ages_column_barchart(df2,col_ages_name2,col_sexes_name2,chart_title2)
    

#Analyze a categorical values of a df column and plot the corresponding barchart 
def categorical_column_analysis_and_bar_chart(df,col,chart_title):
    categorical_column_analysis(df,col)
    create_categorical_column_bar_chart(df,col,chart_title)

''''''
#Analyze a categorical values df column and create the corresponding barchart 
def categorical_column_analysis_and_bar_chart(df_first,df_last,col_first,col_last,column_values_keys_list,chart_title):
    categorical_column_analysis(df_first,col_first.name,column_values_keys_list,1)
    categorical_column_analysis(df_last,col_last.name,column_values_keys_list,1)
    create_categorical_column_bar_chart(df_first,col_first,chart_title+' 2M(29_transf)')
    create_categorical_column_bar_chart(df_last,col_last,chart_title+' 400K(21_valid)')
''''''


#Analyze the categorical values of a certain column and create the corresponding barchart for 2 dfs
def df_compare_categorical_column_analysis_and_bar_chart(col,df1,chart1,df2,chart2):
    
    categorical_column_analysis_and_bar_chart(df1,col,chart1)
    categorical_column_analysis_and_bar_chart(df2,col,chart2)
'''
'''    
#Return a Dataframe with all the rows than contain all the valid and Not NaN values for a single column
def filter_values_of_a_column(df,df_col_name,invalid_val_list):
    
    df_not_null = df
    
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
'''
'''
#Create the Train & Test sets fron a dataframe
def prepare_dataset(df_csv_file,x_column_list,y_column_list,rand_state,train_sz,dataset_name):
    
    dataframe = pd.read_csv(df_csv_file)
    dataset = dataframe.values
    
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
    
    return X_train,X_test,y_train,y_test
'''
'''
#Create a Classifier Model with default or specific Hyperparameters
def create_model(method_name,param_dict):
    
    method='none'
    
    method_dict = {'lr':('Logistic Regression',LogisticRegression()),
                   'dt':('Decision Tree',DecisionTreeClassifier()),
                   'rf':('Random Forest',RandomForestClassifier()),
                   'kn':('KNeighbors',KNeighborsClassifier()),
                   'kms':('KMeans',KMeans()),
                   'mlp':('Multi-Layer Perceprtrons',MLPClassifier()),
                   'svm':('Support Vector Machine',SVC())}
    
    method = method_dict[method_name][0]
    model = method_dict[method_name][1]
    model.set_params(**param_dict)
    
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

'''
'''
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
    
    file_path = 'files/txt/' + method_name + '.txt'
    
    with open(file_path, 'a+') as file_object:
        
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
