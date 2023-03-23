import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder #Για unordered categorical features
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

from Column import Column#[created class]



'''Anaconda commands in conda promt''''''
   
conda create -n py37 python=3.7  #for version 3.6
conda activate py37
conda install wheel
conda install pandas
conda install -c conda-forge imbalanced-learn
conda install matplotlib
pip install matplotlib

'''



'''Dataset Description''''''

GENERAL VALUES:
97 ------> NO APLICA (DOES NOT APPLY)
98 ------> SE IGNORA (IT IS IGNORED)
99 ------> NO ESPECIFICADO (NOT SPECIFIED)

COLUMN TITLES:
01.FECHA_ACTUALIZACION-----> DATE OF DATA UPDATE (AAAA-MM-DD)
02.ID_REGISTRO-------------> REGISTRATION ID (Text ex. zz7202)
03.ORIGEN------------------> ORIGIN (Whether or not the patient diagnosed and or hospitalized to a USER medical Unit or not [1:USMER, 2:NOT USMER])
04.SECTOR------------------> HEALTHCARE SECTOR (The type of institution of the National Health System that provided the care [1:RED CROSS, 2:DIF, 3:STATE, 4:IMSS, 5:IMSS-BIENESTAR, 6:ISSSTE, 7:MUNICIPAL, 8:PEMEX, 9:PRIVATE, 10:SEDENA, 11:SEMAR, 12:SSA , 13:UNIVERSITY])
05.ENTIDAD_UM--------------> ENTITY OF MEDICAL UNIT (The entity where the medical unit that provided the care is located [CATALOG of the diffenet Federal Health Entities 1:AGUASCALIENTES, 2:BAJA CALIFORNIA, 3:BAJA CALIFORNIA SOUTH, 4:CAMPECHE, 5:COAHUILA DE ZARAGOZA e.t.c.])
06.SEXO--------------------> SEX (Patient's sex [1:Woman, 2: Man])
07.ENTIDAD_NAC-------------> PATIENT'S BIRTH HEALTHCARE ENTITY (The patient's health care entity of birth [CATALOG of the diffenet Federal Health Entities 1:AGUASCALIENTES, 2:BAJA CALIFORNIA, 3:BAJA CALIFORNIA SOUTH, 4:CAMPECHE, 5:COAHUILA DE ZARAGOZA e.t.c.])
08.ENTIDAD_RES-------------> PATIENT RESIDENCE'S HEALTHCARE ENTITY (The health care entity that the patient's residence belongs [CATALOG of the diffenet Federal Health Entities 1:AGUASCALIENTES, 2:BAJA CALIFORNIA, 3:BAJA CALIFORNIA SOUTH, 4:CAMPECHE, 5:COAHUILA DE ZARAGOZA e.t.c.])
09.MUNICIPIO_RES-----------> PATIENT RESIDENCE'S MUNICIPALITY (The municipality of residence of the patient [CATALOG of the diffenet Municipalities 1:AGUASCALIENTES, 2:ASIENTOS, 3:CALVILLO, 4:COSÍO, 5:JESÚS MARÍA e.t.c.])
10.TIPO_PACIENTE-----------> TYPE OF PATIENT (The type of care the patient received in the unit. It is called an outpatient if you returned home or it is called an inpatient if you were admitted to the hospital [1:OUTPATIENT-AMBULATORY, 2:INPATIENT-HOSPITALIZED]) 
11.FECHA_INGRESO-----------> HOSPITAL ADMISSION DATE (AAAA-MM-DD)
12.FECHA_SINTOMAS----------> SYMPTOM ONSET DATE (The date on which the patient's symptoms began [AAAA-MM-DD])
13.FECHA_DEF---------------> DATE OF DEATH (The date the patient died [AAAA-MM-DD, exception 9999-99-99: Survived])
14.INTUBADO----------------> INTUBATED (Whether or not the patient required intubation [1:YES, 2:NO])
15.NEUMONIA----------------> PNEUMONIA (Whether or not the patient was diagnosed with pneumonia [1:YES, 2:NO])
16.EDAD--------------------> AGE (Patient's Age)
17.NACIONALIDAD------------> NATIONALITY (Patient's Nationality [1:MEXICAN, 2:FOREIGN])
18.EMBARAZO----------------> PREGNANT (Whether or not the patient is pregnant [1:YES, 2:NO])
19.HABLA_LENGUA_INDIG------> NATIVE LANGUAGE SPEAKER (Whether or not the patient speaks an indigenous language [1:YES, 2:NO])
20.INDIGENA----------------> INDIGENOUS (Whether or not the patient identifies himself as an indigenous person [1:YES, 2:NO])
21.DIABETES----------------> DIABETIC (Whether or not the patient has been diagnosed with diabetes [1:YES, 2:NO])  
22.EPOC--------------------> COPD (Whether or not the patient has been diagnosed with COPD[ΧΑΠ] [1:YES, 2:NO])
23.ASMA--------------------> ASTHMA (Whether or not the patient has been diagnosed with ASTHMA [1:YES, 2:NO])
24.INMUSUPR----------------> IMMUNOSUPPRESSED (Whether or not the patient is immunosuppressed [1:YES, 2:NO])
25.HIPERTENSION------------> HYPERTENSION (Whether or not the patient has been diagnosed with hypertension [1:YES, 2:NO])
26.OTRA_COM----------------> OTHER CHRONIC DISEASE (Whether or not the patient has been diagnosed with any other disease [1:YES, 2:NO])
27.CARDIOVASCULAR----------> CARDIOVASCULAR (Whether or not the patient has been diagnosed with cardiovascular disease [1:YES, 2:NO])
28.OBESIDAD----------------> OBESITY (Whether or not the patient has been diagnosed with obesity [1:YES, 2:NO])
29.RENAL_CRONICA-----------> CHRONIC KIDNEY FAILURE(Whether or not the patient has been diagnosed with chronic kidney failure [1:YES, 2:NO])
30.TABAQUISMO--------------> SMOKER (Whether or not the patient has a smoking habit [1:YES, 2:NO])
31.OTRO_CASO---------------> CONTACT WITH COVID-19 CASE (Whether or not the patient had contact with any other case diagnosed with SARS CoV-2 [1:YES, 2:NO])
32.TOMA_MUESTRA_LAB--------> LAB SAMPLE TAKEN (Whether or not the patient had a laboratory sample taken [1:YES, 2:NO])
33.RESULTADO_LAB-----------> LAB RESULT (The result of the analysis of the sample reported by the laboratory [1:POSITIVE TO SARS-CoV-2, 2:NOT POSITIVE TO SARS-CoV-2, 3:RESULT PENDING, 4:RESULT NOT ADEQUATE]
34.TOMA_MUESTRA_ANTIGENO---> ANTIGEN SAMPLE TAKEN (Whether or not the patient had an antigen sample for SARS-CoV-2 taken [1:YES, 2:NO])
35.RESULTADO_ANTIGENO------> ANTIGEN RESULT (The result of the analysis of the antigen sample taken from the patient [1:POSITIVE TO SARS-COV-2, 2:NEGATIVE TO SARS-COV-2])
36.CLASIFICACION_FINAL-----> FINAL CLASSIFICATION (If the patient is a case of COVID-19 according to the catalog [1:CASE OF COVID-19 CONFIRMED BY CLINICAL EPIDEMIOLOGICAL ASSOCIATION, 2:CASE OF COVID-19 CONFIRMED BY DICTAMINATION COMMITTEE, 3:CASE OF SARS-COV-2 CONFIRMED, 4:INVALID BY LABORATORY, 5:NOT PERFORMED BY LABORATORY 6:SUSPECT CASE, 7:NEGATIVE TO SARS-COV-2]) 
37.MIGRANTE----------------> IMMIGRANT (Whether or not the patient is a immigrant [1:YES, 2:NO])
38.PAIS_NACIONALIDAD-------> NATIONALITY (The nationality of the patient [TEXT]) 
39.PAIS_ORIGEN-------------> COUNTRY OF ORIGIN (The country from which the patient left for Mexico [TEXT]) 
40.UCI---------------------> ICU (Whether or not the patient required admission to an Intensive Care Unit [1:YES, 2:NO])

'''



'''Functions'''

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


''''''



#Start Time
start = time.time()



'''Dataframes'''

#DataFrame_01(40)-from the Original Dataset in Spanish with All(40) columns
df_40_sp = pd.read_csv('01.Covid19MPD.(10K).csv',header=0)
df_40_sp = pd.read_csv('01.Covid19MPD.(Original).csv',header=0)###[12.425.181 samples in total]
df_40_sp.name = 'df_40_sp'


#DataFrame_02(40)-DataFrame in English & Numbered with All(40) columns
#---40 Columns--------------------------------------------------------
df_40_en_num = df_40_sp.rename(columns={'FECHA_ACTUALIZACION':'01.DATE_OF_DATA_UPDATE',
                                        'ID_REGISTRO':'02.REGISTRATION_ID',
                                        'ORIGEN':'03.ORIGIN',
                                        'SECTOR':'04.HEALTHCARE_SECTOR',
                                        'ENTIDAD_UM':'05.ENTITY_OF_HEALTHCARE_UNIT',
                                        'SEXO':'06.SEX',
                                        'ENTIDAD_NAC':'07.P_BIRTHPLACE_ENTITY_HEALTHCARE_UNIT',
                                        'ENTIDAD_RES':'08.P_RESIDENCE_HEALTHCARE_ENTITY',
                                        'MUNICIPIO_RES':'09.P_RESIDENCE_MUNICIPALITY',
                                        'TIPO_PACIENTE':'10.TYPE_OF_PATIENT',
                                        'FECHA_INGRESO':'11.ADMISSION DATE',
                                        'FECHA_SINTOMAS':'12.SYMPTOM ONSET DATE',
                                        'FECHA_DEF':'13.DATE_OF_DEATH',
                                        'INTUBADO':'14.INTUBATED',
                                        'NEUMONIA':'15.PNEUMONIA',
                                        'EDAD':'16.AGE',
                                        'NACIONALIDAD':'17.NATIONALITY',
                                        'EMBARAZO':'18.PREGNANCY',
                                        'HABLA_LENGUA_INDIG':'19.NATIVE_LANGUAGE_SPEAKER',
                                        'INDIGENA':'20.INDIGENOUS',
                                        'DIABETES':'21.DIABETIC',
                                        'EPOC':'22.COPD',
                                        'ASMA':'23.ASTHMA',
                                        'INMUSUPR':'24.IMMUNOSUPPRESSED',
                                        'HIPERTENSION':'25.HYPERTENSION',
                                        'OTRA_COM':'26.OTHER_CHRONIC_DISEASE',
                                        'CARDIOVASCULAR':'27.CARDIOVASCULAR',
                                        'OBESIDAD':'28.OBESITY',
                                        'RENAL_CRONICA':'29.CHRONIC_KIDNEY_FAILURE',
                                        'TABAQUISMO':'30.SMOKER',
                                        'OTRO_CASO':'31.CONTACT_WITH_COVID-19_CASE',
                                        'TOMA_MUESTRA_LAB':'32.LAB_SAMPLE_TAKEN', 
                                        'RESULTADO_LAB':'33.LAB_RESULT',
                                        'TOMA_MUESTRA_ANTIGENO':'34.ANTIGEN_SAMPLE_TAKEN',
                                        'RESULTADO_ANTIGENO':'35.ANTIGEN_RESULT',
                                        'CLASIFICACION_FINAL':'36.FINAL CLASSIFICATION',
                                        'MIGRANTE':'37.IMMIGRANT',
                                        'PAIS_NACIONALIDAD':'38.NATIONALITY',
                                        'PAIS_ORIGEN':'39.COUNTRY_OF_ORIGIN',
                                        'UCI':'40.ICU'})

df_40_en_num.name = 'df_40_en_num'


#DataFrame_03(26+1)-DataFrame in English & Numbered, without 14 columns that contains data about region & nationality
#---27 Columns-------------------------------------------------------------------------------------------------------
df_27_en_num = df_40_sp.drop(columns=['FECHA_ACTUALIZACION',
                                      'ORIGEN',
                                      'SECTOR',
                                      'ENTIDAD_UM',
                                      'ENTIDAD_NAC',
                                      'ENTIDAD_RES',
                                      'MUNICIPIO_RES',
                                      'NACIONALIDAD',
                                      'HABLA_LENGUA_INDIG',
                                      'INDIGENA',
                                      'MIGRANTE',
                                      'PAIS_NACIONALIDAD',
                                      'PAIS_ORIGEN'])

df_27_en_num = df_27_en_num.rename(columns={'ID_REGISTRO':'01.REGISTRATION_ID',
                                            'SEXO':'02.SEX',
                                            'TIPO_PACIENTE':'03.TYPE_OF_PATIENT',
                                            'FECHA_INGRESO':'04.ADMISSION_DATE',
                                            'FECHA_SINTOMAS':'05.SYMPTOM_ONSET_DATE',
                                            'FECHA_DEF':'06.DATE_OF_DEATH',
                                            'INTUBADO':'07.INTUBATED',
                                            'NEUMONIA':'08.PNEUMONIA',
                                            'EDAD':'09.AGE',
                                            'EMBARAZO':'10.PREGNANCY',
                                            'DIABETES':'11.DIABETIC',
                                            'EPOC':'12.COPD',
                                            'ASMA':'13.ASTHMA',
                                            'INMUSUPR':'14.IMMUNOSUPPRESSED',
                                            'HIPERTENSION':'15.HYPERTENSION',
                                            'OTRA_COM':'16.OTHER_CHRONIC_DISEASE',
                                            'CARDIOVASCULAR':'17.CARDIOVASCULAR',
                                            'OBESIDAD':'18.OBESITY',
                                            'RENAL_CRONICA':'19.CHRONIC_KIDNEY_FAILURE',
                                            'TABAQUISMO':'20.SMOKER',
                                            'OTRO_CASO':'21.CONTACT_WITH_COVID-19_CASE',
                                            'TOMA_MUESTRA_LAB':'22.LAB_SAMPLE_TAKEN', 
                                            'RESULTADO_LAB':'23.LAB_RESULT',
                                            'TOMA_MUESTRA_ANTIGENO':'24.ANTIGEN_SAMPLE_TAKEN',
                                            'RESULTADO_ANTIGENO':'25.ANTIGEN_RESULT',
                                            'CLASIFICACION_FINAL':'26.FINAL_CLASSIFICATION',
                                            'UCI':'27.ICU'})


#Create a csv with the Dataframe that contains all the COVID-19 Positive samples
#---27-Columns-ALL-(2.062.829)--------------------------------------------------
df_sars_cov_2_p_all_27_en_num = df_27_en_num.loc[df_27_en_num['23.LAB_RESULT'] == 1] #[2.062.829 COVID-19 Positive samples from 12.425.181 total samples]
df_sars_cov_2_p_all_27_en_num.name = 'df_sars_cov_2_p_all_27_en_num'
df_sars_cov_2_p_all_27_en_num.to_csv('01.Covid19MPD.(Pos_27).csv', index = False)###
df_sars_cov_2_p_all_27_en_num = pd.read_csv('01.Covid19MPD.(Pos_27).csv',header=0)


#Create a csv with the Dataframe that contains all the COVID-19 Positive samples(29 Columns)
#with the extra 2 columns (28.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION,29.SURVIVED)
#Added the column 28.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION from 04.ADMISSION_DATE & 05.SYMPTOM_ONSET_DATE
#---29.TRANFORMED.ALL.(2.062.829)-----------------------------------------------------------------------
df_sars_cov_2_p_all_transf_29_en_num = pd.read_csv('01.Covid19MPD.(Pos_27).csv',header=0)#[2.062.829 COVID-19 Positive samples from 12.425.181 total samples]
df_sars_cov_2_p_all_transf_29_en_num['04.ADMISSION_DATE'] = pd.to_datetime(df_sars_cov_2_p_all_transf_29_en_num['04.ADMISSION_DATE'], format='%Y-%m-%d')
df_sars_cov_2_p_all_transf_29_en_num['05.SYMPTOM_ONSET_DATE'] = pd.to_datetime(df_sars_cov_2_p_all_transf_29_en_num['05.SYMPTOM_ONSET_DATE'], format='%Y-%m-%d')
df_sars_cov_2_p_all_transf_29_en_num['28.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION'] = (df_sars_cov_2_p_all_transf_29_en_num['04.ADMISSION_DATE'] - df_sars_cov_2_p_all_transf_29_en_num['05.SYMPTOM_ONSET_DATE']).dt.days

#Added the column 29.SURVIVED from 06.DATE_OF_DEATH
df_sars_cov_2_p_all_transf_29_en_num['29.SURVIVED'] = np.where(df_sars_cov_2_p_all_transf_29_en_num['06.DATE_OF_DEATH']=='9999-99-99', 1, 2)
 
df_sars_cov_2_p_all_transf_29_en_num.name = 'df_sars_cov_2_p_all_transf_29_en_num'
df_sars_cov_2_p_all_transf_29_en_num.to_csv('01.Covid19MPD.(Pos_transf_29).csv', index = False)
df_sars_cov_2_p_all_transf_29_en_num = pd.read_csv('01.Covid19MPD.(Pos_transf_29).csv',header=0)


#Clear the Dataframe from rows that contain invalid values (NaN,97,98,99)
#except for the Pregnancy Column because the value 97(not apply) applies to men
#---29.VALID.(410.362)---------------------------------------------------------
df_sars_cov_2_p_all_transf_valid_29_en_num = pd.read_csv('01.Covid19MPD.(Pos_transf_29).csv',header=0)
my_custom_29_col_list=['07.INTUBATED','08.PNEUMONIA','11.DIABETIC','12.COPD','13.ASTHMA','14.IMMUNOSUPPRESSED',
                    '15.HYPERTENSION','16.OTHER_CHRONIC_DISEASE','17.CARDIOVASCULAR','18.OBESITY',
                    '19.CHRONIC_KIDNEY_FAILURE','20.SMOKER','21.CONTACT_WITH_COVID-19_CASE','27.ICU','29.SURVIVED']

df_sars_cov_2_p_all_transf_valid_29_en_num = filter_values_of_multi_columns(df_sars_cov_2_p_all_transf_valid_29_en_num,my_custom_29_col_list,[97,98,99])
df_sars_cov_2_p_all_transf_valid_29_en_num = filter_values_of_multi_columns(df_sars_cov_2_p_all_transf_valid_29_en_num,['10.PREGNANCY'],[98,99])
df_sars_cov_2_p_all_transf_valid_29_en_num.name = 'df_sars_cov_2_p_all_transf_valid_29_en_num'
df_sars_cov_2_p_all_transf_valid_29_en_num.to_csv('01.Covid19MPD.(Pos_valid_29).csv', index = False)#[410.362 valid from 2.062.829 COVID-19 Positive samples]
df_sars_cov_2_p_all_transf_valid_29_en_num = pd.read_csv('01.Covid19MPD.(Pos_valid_29).csv',header=0)


#Create a csv with the Dataframe tha contains all the values in binary format
#(except 01.REGISTRATION_ID,09.AGE,10.PREGNANCY,23.LAB_RESULT,26.FINAL_CLASSIFICATION)
#and without the columns 19.LAB_RESULT,20.FINAL_CLASSIFICATION
#---29.BINARY.(410.362)---------------------------------------
df_sars_cov_2_p_all_transf_valid_bin_29_en_num = pd.read_csv('01.Covid19MPD.(Pos_valid_29).csv',header=0)
column_29_list = ['02.SEX','03.TYPE_OF_PATIENT','07.INTUBATED','08.PNEUMONIA','10.PREGNANCY','11.DIABETIC',
                  '12.COPD','13.ASTHMA','14.IMMUNOSUPPRESSED','15.HYPERTENSION','16.OTHER_CHRONIC_DISEASE',
                  '17.CARDIOVASCULAR','18.OBESITY','19.CHRONIC_KIDNEY_FAILURE','20.SMOKER',
                  '21.CONTACT_WITH_COVID-19_CASE','27.ICU','29.SURVIVED']

df_sars_cov_2_p_all_transf_valid_bin_29_en_num[column_29_list]=np.where(df_sars_cov_2_p_all_transf_valid_bin_29_en_num[column_29_list]==2, int(0),
                                                              (np.where(df_sars_cov_2_p_all_transf_valid_bin_29_en_num[column_29_list]==1, int(1), int(97))))

df_sars_cov_2_p_all_transf_valid_bin_29_en_num.name = 'df_sars_cov_2_p_all_transf_valid_bin_29_en_num'
df_sars_cov_2_p_all_transf_valid_bin_29_en_num.to_csv('01.Covid19MPD.(Pos_valid_bin_29).csv',index = False)#[410.362 valid from 2.062.829 COVID-19 Positive samples]
df_sars_cov_2_p_all_transf_valid_bin_29_en_num = pd.read_csv('01.Covid19MPD.(Pos_valid_bin_29).csv',header=0)


#Create a csv with the Dataframe that contains all the COVID-19 Positive samples(23 Columns) 
#with the extra 2 columns (28.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION,29.SURVIVED)
#and minus 6 columns (04.ADMISSION_DATE,05.SYMPTOM_ONSET_DATE,06.DATE_OF_DEATH,22.LAB_SAMPLE_TAKEN,24.ANTIGEN_SAMPLE_TAKEN,25.ANTIGEN_RESULT)
#---23.TRANFORMED.ALL.(2.062.829)------------------------------------------------------------------------------------------------------------
df_sars_cov_2_p_all_transf_23_en_num = pd.read_csv('01.Covid19MPD.(Pos_transf_29).csv',header=0)
df_sars_cov_2_p_all_transf_23_en_num = df_sars_cov_2_p_all_transf_23_en_num.drop(columns=['04.ADMISSION_DATE',
                                                                                          '05.SYMPTOM_ONSET_DATE',
                                                                                          '06.DATE_OF_DEATH',
                                                                                          '22.LAB_SAMPLE_TAKEN',
                                                                                          '24.ANTIGEN_SAMPLE_TAKEN',
                                                                                          '25.ANTIGEN_RESULT'])

df_sars_cov_2_p_all_transf_23_en_num = df_sars_cov_2_p_all_transf_23_en_num.rename(columns={'03.TYPE_OF_PATIENT':'03.OUTPATIENT',
                                                                                            '07.INTUBATED':'04.INTUBATED',
                                                                                            '08.PNEUMONIA':'05.PNEUMONIA',
                                                                                            '09.AGE':'06.AGE',
                                                                                            '10.PREGNANCY':'07.PREGNANCY',
                                                                                            '11.DIABETIC':'08.DIABETIC',
                                                                                            '12.COPD':'09.COPD',
                                                                                            '13.ASTHMA':'10.ASTHMA',
                                                                                            '14.IMMUNOSUPPRESSED':'11.IMMUNOSUPPRESSED',
                                                                                            '15.HYPERTENSION':'12.HYPERTENSION',
                                                                                            '16.OTHER_CHRONIC_DISEASE':'13.OTHER_CHRONIC_DISEASE',
                                                                                            '17.CARDIOVASCULAR':'14.CARDIOVASCULAR',
                                                                                            '18.OBESITY':'15.OBESITY',
                                                                                            '19.CHRONIC_KIDNEY_FAILURE':'16.CHRONIC_KIDNEY_FAILURE',
                                                                                            '20.SMOKER':'17.SMOKER',
                                                                                            '21.CONTACT_WITH_COVID-19_CASE':'18.CONTACT_WITH_COVID-19_CASE',
                                                                                            '23.LAB_RESULT':'19.LAB_RESULT',
                                                                                            '26.FINAL_CLASSIFICATION':'20.FINAL_CLASSIFICATION',
                                                                                            '27.ICU':'21.ICU',
                                                                                            '28.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION':'22.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION',
                                                                                            '29.SURVIVED':'23.SURVIVED'})

df_sars_cov_2_p_all_transf_23_en_num.name = 'df_sars_cov_2_p_all_transf_23_en_num'
df_sars_cov_2_p_all_transf_23_en_num.to_csv('01.Covid19MPD.(Pos_transf_23).csv', index = False)
df_sars_cov_2_p_all_transf_23_en_num = pd.read_csv('01.Covid19MPD.(Pos_transf_23).csv',header=0)


#Clear the Dataframe from rows that contains invalid values (NaN,97,98,99)
#except for the Pregnancy Column because the value 97(not apply) applies to men
#---23.VALID.(410.362)---------------------------------------------------------
df_sars_cov_2_p_all_transf_valid_23_en_num = pd.read_csv('01.Covid19MPD.(Pos_transf_23).csv',header=0)

my_custom_23_col_list=['04.INTUBATED','05.PNEUMONIA','08.DIABETIC','09.COPD','10.ASTHMA','11.IMMUNOSUPPRESSED',
                      '12.HYPERTENSION','13.OTHER_CHRONIC_DISEASE','14.CARDIOVASCULAR','15.OBESITY',
                      '16.CHRONIC_KIDNEY_FAILURE','17.SMOKER','18.CONTACT_WITH_COVID-19_CASE','21.ICU','23.SURVIVED']

df_sars_cov_2_p_all_transf_valid_23_en_num = filter_values_of_multi_columns(df_sars_cov_2_p_all_transf_valid_23_en_num,my_custom_23_col_list,[97,98,99])
df_sars_cov_2_p_all_transf_valid_23_en_num = filter_values_of_multi_columns(df_sars_cov_2_p_all_transf_valid_23_en_num,['07.PREGNANCY'],[98,99])
df_sars_cov_2_p_all_transf_valid_23_en_num.name = 'df_sars_cov_2_p_all_transf_valid_23_en_num'
df_sars_cov_2_p_all_transf_valid_23_en_num.to_csv('01.Covid19MPD.(Pos_valid_23).csv', index = False)#[410.362 valid from 2.062.829 COVID-19 Positive samples]
df_sars_cov_2_p_all_transf_valid_23_en_num = pd.read_csv('01.Covid19MPD.(Pos_valid_23).csv',header=0)#[410.362 valid from 2.062.829 COVID-19 Positive samples]


#Create a csv with the Dataframe that contains all the valid values in Binary format
#(except 01.REGISTRATION_ID,06.AGE,07.PREGNANCY,19.LAB_RESULT,20.FINAL_CLASSIFICATION)
#---23.BINARY.(410.362)---------------------------------------------------------------
df_sars_cov_2_p_all_transf_valid_bin_23_en_num = pd.read_csv('01.Covid19MPD.(Pos_valid_23).csv',header=0)

column_23_list = ['02.SEX','03.OUTPATIENT','04.INTUBATED','05.PNEUMONIA','07.PREGNANCY','08.DIABETIC',
                  '09.COPD','10.ASTHMA','11.IMMUNOSUPPRESSED','12.HYPERTENSION','13.OTHER_CHRONIC_DISEASE',
                  '14.CARDIOVASCULAR','15.OBESITY','16.CHRONIC_KIDNEY_FAILURE','17.SMOKER',
                  '18.CONTACT_WITH_COVID-19_CASE','21.ICU','23.SURVIVED']

df_sars_cov_2_p_all_transf_valid_bin_23_en_num[column_23_list] = np.where(df_sars_cov_2_p_all_transf_valid_bin_23_en_num[column_23_list]==2, int(0),
                                                                (np.where(df_sars_cov_2_p_all_transf_valid_bin_23_en_num[column_23_list]==1, int(1), int(97))))

df_sars_cov_2_p_all_transf_valid_bin_23_en_num.name = 'df_sars_cov_2_p_all_transf_valid_bin_23_en_num'
df_sars_cov_2_p_all_transf_valid_bin_23_en_num.to_csv('01.Covid19MPD.(Pos_valid_bin_23).csv', index = False)#[410.362 valid from 2.062.829 COVID-19 Positive samples]
df_sars_cov_2_p_all_transf_valid_bin_23_en_num = pd.read_csv('01.Covid19MPD.(Pos_valid_bin_23).csv',header=0)#[410.362 valid from 2.062.829 COVID-19 Positive samples]


#Create a csv with the Dataframe that contains all the COVID-19 Positive samples
#and without the columns 19.LAB_RESULT,20.FINAL_CLASSIFICATION
#---21.TRANFORMED.ALL.(2.062.829)-----------------------------
df_sars_cov_2_p_all_transf_21_en_num = pd.read_csv('01.Covid19MPD.(Pos_transf_23).csv',header=0)
df_sars_cov_2_p_all_transf_21_en_num = df_sars_cov_2_p_all_transf_21_en_num.drop(columns=['19.LAB_RESULT',
                                                                                          '20.FINAL_CLASSIFICATION'])

df_sars_cov_2_p_all_transf_21_en_num = df_sars_cov_2_p_all_transf_21_en_num.rename(columns={'21.ICU':'19.ICU',
                                                                                            '22.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION':'20.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION',
                                                                                            '23.SURVIVED':'21.SURVIVED'})

df_sars_cov_2_p_all_transf_21_en_num.name = 'df_sars_cov_2_p_all_transf_21_en_num'
df_sars_cov_2_p_all_transf_21_en_num.to_csv('01.Covid19MPD.(Pos_transf_21).csv', index = False)
df_sars_cov_2_p_all_transf_21_en_num = pd.read_csv('01.Covid19MPD.(Pos_transf_21).csv',header=0)#[2.062.829 COVID-19 Positive samples from 12.425.181 total samples]


#Clear the Dataframe from rows that contains invalid values (NaN,97,98,99)
#except for the Pregnancy Column because the value 97(not apply) applies to men
#---21.VALID.(410.362)---------------------------------------------------------
df_sars_cov_2_p_all_transf_valid_21_en_num = pd.read_csv('01.Covid19MPD.(Pos_transf_21).csv',header=0)

my_custom_21_col_list=['04.INTUBATED','05.PNEUMONIA','08.DIABETIC','09.COPD','10.ASTHMA','11.IMMUNOSUPPRESSED',
                      '12.HYPERTENSION','13.OTHER_CHRONIC_DISEASE','14.CARDIOVASCULAR','15.OBESITY','16.CHRONIC_KIDNEY_FAILURE','17.SMOKER',
                      '18.CONTACT_WITH_COVID-19_CASE','19.ICU','21.SURVIVED']

df_sars_cov_2_p_all_transf_valid_21_en_num = filter_values_of_multi_columns(df_sars_cov_2_p_all_transf_valid_21_en_num,my_custom_21_col_list,[97,98,99])
df_sars_cov_2_p_all_transf_valid_21_en_num = filter_values_of_multi_columns(df_sars_cov_2_p_all_transf_valid_21_en_num,['07.PREGNANCY'],[98,99])
df_sars_cov_2_p_all_transf_valid_21_en_num.name = 'df_sars_cov_2_p_all_transf_valid_21_en_num'
df_sars_cov_2_p_all_transf_valid_21_en_num.to_csv('01.Covid19MPD.(Pos_valid_21).csv', index = False)#[410.362 valid from 2.062.829 COVID-19 Positive samples]
df_sars_cov_2_p_all_transf_valid_21_en_num = pd.read_csv('01.Covid19MPD.(Pos_valid_21).csv',header=0)

#Create a csv with the Dataframe tha contains all the values in binary format
#(except 01.REGISTRATION_ID,06.AGE,07.PREGNANCY,19.LAB_RESULT,20.FINAL_CLASSIFICATION)
#and without the columns 19.LAB_RESULT,20.FINAL_CLASSIFICATION
#---21.BINARY.(410.362)---------------------------------------
df_sars_cov_2_p_all_transf_valid_bin_21_en_num = pd.read_csv('01.Covid19MPD.(Pos_valid_21).csv',header=0)

column_21_list = ['02.SEX','03.OUTPATIENT','04.INTUBATED','05.PNEUMONIA','07.PREGNANCY','08.DIABETIC',
                  '09.COPD','10.ASTHMA','11.IMMUNOSUPPRESSED','12.HYPERTENSION','13.OTHER_CHRONIC_DISEASE',
                  '14.CARDIOVASCULAR','15.OBESITY','16.CHRONIC_KIDNEY_FAILURE','17.SMOKER',
                  '18.CONTACT_WITH_COVID-19_CASE','19.ICU','21.SURVIVED']

df_sars_cov_2_p_all_transf_valid_bin_21_en_num[column_21_list]=np.where(df_sars_cov_2_p_all_transf_valid_bin_21_en_num[column_21_list]==2, int(0),
                                                              (np.where(df_sars_cov_2_p_all_transf_valid_bin_21_en_num[column_21_list]==1, int(1), int(97))))

df_sars_cov_2_p_all_transf_valid_bin_21_en_num.name = 'df_sars_cov_2_p_all_transf_valid_bin_21_en_num'
df_sars_cov_2_p_all_transf_valid_bin_21_en_num.to_csv('01.Covid19MPD.(Pos_valid_bin_21).csv',index = False)#[410.362 valid from 2.062.829 COVID-19 Positive samples]
df_sars_cov_2_p_all_transf_valid_bin_21_en_num = pd.read_csv('01.Covid19MPD.(Pos_valid_bin_21).csv',header=0)

''''''



'''Column Analyis & Chart Ploting'''

'''01.REGISTRATION_ID'''
show_column_info(df_sars_cov_2_p_all_27_en_num,'01.REGISTRATION_ID')
show_column_info(df_sars_cov_2_p_all_transf_valid_21_en_num,'01.REGISTRATION_ID')

'''02.SEX'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('02.SEX','Sexes', {'Women':1,'Men':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('02.SEX','Sexes', {'Women':1,'Men':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Women',1],['Men',2]],'02.Gender Distribution')

'''03.TYPE_OF_PATIENT(Renamed 03.OUTPATIENT)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('03.TYPE_OF_PATIENT','Type of Patient', {'Outpatient':1,'Inpatient':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('03.OUTPATIENT','Type of Patient', {'Outpatient':1,'Inpatient':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Outpatient-Ambulatory',1],['Inpatient-Hospitalized',2]],'03.Types of Patients')

'''07.INTUBATED(Renamed 04.INTUBATED)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('07.INTUBATED','Intubations', {'Intubated':1,'Not Intubated':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('04.INTUBATED','Intubations', {'Intubated':1,'Not Intubated':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Intubated',1],['Not Intubated',2]],'04.Intubated Distribution')

'''08.PNEUMONIA(Renamed 05.PNEUMONIA)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                         Column('08.PNEUMONIA','Pneumonia', {'Have Pneumonia':1,'Do not have Pneumonia':2,'not apply':97,'ignored':98,'not specified':99}),
                                         Column('05.PNEUMONIA','Pneumonia', {'Have Pneumonia':1,'Do not have Pneumonia':2,'not apply':97,'ignored':98,'not specified':99}),
                                         [['Have Pneumonia',1],['Do not have Pneumonia',2]],'05.Pneumonia Distribution')

'''09.AGE(Renamed 06.AGE)'''
ages_column_analysis_and_barchart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num
                                  ,'09.AGE','06.AGE','02.SEX','02.SEX','09.Age Distribution')

'''10.PREGNANCY(Renamed 07.PREGNANCY)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('10.PREGNANCY','Pregnancies',{'Pregnant':1,'Not Pregnant':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('07.PREGNANCY','Pregnancies',{'Pregnant':1,'Not Pregnant':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Pregnant',1],['Not Pregnant',2]],'07.Pregnancy Distribution')

'''11.DIABETIC(Renamed 08.DIABETIC)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('11.DIABETIC','Diabetics', {'Diabetic':1,'Non Diabetic':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('08.DIABETIC','Diabetics', {'Diabetic':1,'Non Diabetic':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Diabetic',1],['Non Diabetic',2]],'08.Diabetes Distribution')

'''12.COPD(Renamed 09.COPD)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('12.COPD','COPD', {'Have COPD':1,'Do not have COPD':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('09.COPD','COPD', {'Have COPD':1,'Do not have COPD':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Have COPD',1],['Do not have COPD',2]],'09.COPD Distribution')

'''13.ASTHMA(Renamed 10.ASTHMA)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('13.ASTHMA','Asthma', {'Have Asthma':1,'Do not have Asthma':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('10.ASTHMA','Asthma', {'Have Asthma':1,'Do not have Asthma':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Have Asthma',1],['Do not have Asthma',2]],'10.Asthma Distribution')

'''14.IMMUNOSUPPRESSED(Renamed 11.IMMUNOSUPPRESSED)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('14.IMMUNOSUPPRESSED','Immunosuppressed', {'Immunosuppressed':1,'Not Immunosuppressed':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('11.IMMUNOSUPPRESSED','Immunosuppressed', {'Immunosuppressed':1,'Not Immunosuppressed':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Immunosuppressed',1],['Not Immunosuppressed',2]],'11.Immunosuppressed Distribution')

'''15.HYPERTENSION(Renamed 12.HYPERTENSION)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('15.HYPERTENSION','Hypertension', {'Have Hypertension':1,'Do not have Hypertension':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('12.HYPERTENSION','Hypertension', {'Have Hypertension':1,'Do not have Hypertension':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Have Hypertension',1],['Do not have Hypertension',2]],'12.Hypertension Distribution')

'''16.OTHER_CHRONIC_DISEASE(Renamed 13.OTHER_CHRONIC_DISEASE)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('16.OTHER_CHRONIC_DISEASE','Other Chronic Disease', {'Have Other Chronic Disease':1,'Do not have Other Chronic Disease':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('13.OTHER_CHRONIC_DISEASE','Other Chronic Disease', {'Have Other Chronic Disease':1,'Do not have Other Chronic Disease':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Have Other Chronic Disease',1],['Do not have Other Chronic Disease',2]],'13.Other Chronic Disease Distribution')

'''17.CARDIOVASCULAR(Renamed 14.CARDIOVASCULAR)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('17.CARDIOVASCULAR','Cardiovascular Disease', {'Have Cardiovascular Disease':1,'Do not have Cardiovascular Disease':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('14.CARDIOVASCULAR','Cardiovascular Disease', {'Have Cardiovascular Disease':1,'Do not have Cardiovascular Disease':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Have Cardiovascular Disease',1],['Do not have Cardiovascular Disease',2]],'14.Cardiovascular Diseases Distribution')
                                          
'''18.OBESITY(Renamed 15.OBESITY)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('18.OBESITY','Obese', {'Obese':1,'Not Obese':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('15.OBESITY','Obese', {'Obese':1,'Not Obese':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Obese',1],['Not Obese',2]],'15.Obesity Distribution')

'''19.CHRONIC_KIDNEY_FAILURE(Renamed 16.CHRONIC_KIDNEY_FAILURE)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('19.CHRONIC_KIDNEY_FAILURE','Chronic Kidney Failure', {'Have Chronic Kidney Failure':1,'Do not have Chronic Kidney Failure':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('16.CHRONIC_KIDNEY_FAILURE','Chronic Kidney Failure', {'Have Chronic Kidney Failure':1,'Do not have Chronic Kidney Failure':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Have Chronic Kidney Failure',1],['Do not have Chronic Kidney Failure',2]],'16.Chronic Kidney Failure Distribution')

'''20.SMOKER(Renamed 17.SMOKER)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('20.SMOKER','Smokers', {'Smokers':1,'Non Smokers':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('17.SMOKER','Smokers', {'Smokers':1,'Non Smokers':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Smokers',1],['Non Smokers',2]],'17.Smokers Distribution')

'''21.CONTACT_WITH_COVID-19_CASE(Renamed 18.CONTACT_WITH_COVID-19_CASE)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('21.CONTACT_WITH_COVID-19_CASE','Contact with COVID-19 Case', {'Had Contact with COVID-19 Case':1,'Did not have Contact with COVID-19 Case':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('18.CONTACT_WITH_COVID-19_CASE','Contact with COVID-19 Case', {'Had Contact with COVID-19 Case':1,'Did not have Contact with COVID-19 Case':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Had Contact with COVID-19 Case',1],['Did not have Contact with COVID-19 Case',2]],
                                          '18.Contact with COVID-19 Case Distribution')
                                          
'''27.ICU(Renamed 19.ICU)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('27.ICU','ICU', {'ICU':1,'Not ICU':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('19.ICU','ICU', {'ICU':1,'Not ICU':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['ICU',1],['Not ICU',2]],'19.ICU Distribution')

'''04.ADMISSION_DATE & 05.SYMPTOM_ONSET_DATE (Dropped and Added the column 20[28].DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION)'''
#df_sars_cov_2_p_all_27_en_num['06.DATE_OF_DEATH'] = np.where(df_sars_cov_2_p_all_27_en_num['06.DATE_OF_DEATH']=='9999-99-99','2019-09-01',df_sars_cov_2_p_all_27_en_num['06.DATE_OF_DEATH'])
#df_sars_cov_2_p_all_27_en_num['06.DATE_OF_DEATH'] = pd.to_datetime(df_sars_cov_2_p_all_27_en_num['06.DATE_OF_DEATH'], format='%Y-%m-%d')
#df_sars_cov_2_p_all_27_en_num['Hospitalization to Death'] = (df_sars_cov_2_p_all_27_en_num['06.DATE_OF_DEATH'] - df_sars_cov_2_p_all_27_en_num['04.ADMISSION_DATE']).dt.days
#df_sars_cov_2_p_all_27_en_num['Symptom to Death'] = (df_sars_cov_2_p_all_27_en_num['06.DATE_OF_DEATH'] - df_sars_cov_2_p_all_27_en_num['05.SYMPTOM_ONSET_DATE']).dt.days

'''06.DATE_OF_DEATH (Dropped and Added the column 21[29].SURVIVED)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_21_en_num,
                                          Column('29.SURVIVED','Survived', {'Survived':1,'Died':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('21.SURVIVED','Survived', {'Survived':1,'Died':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Survived',1],['Died',2]],'21.Survivors Distribution')

'''22.LAB_SAMPLE_TAKEN(Dropped)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_29_en_num,
                                          Column('22.LAB_SAMPLE_TAKEN','Lab Sample Taken', {'Had Taken Lab Sample':1,'Did not take Lab Sample':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('22.LAB_SAMPLE_TAKEN','Lab Sample Taken', {'Had Taken Lab Sample':1,'Did not take Lab Sample':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Had Taken Lab Sample',1],['Did not take Lab Sample',2]],'22.Lab Sample Taken Distribution')

'''23.LAB_RESULT S.O.S.(Dropped)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_29_en_num,
                                          Column('23.LAB_RESULT','Lab Results', {'Positive to SARS-CoV-2':1,'Not Positive to SARS-CoV-2':2,'Results Pending':3,'Results not Adequate':4,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('23.LAB_RESULT','Lab Results', {'Positive to SARS-CoV-2':1,'Not Positive to SARS-CoV-2':2,'Results Pending':3,'Results not Adequate':4,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Positive to SARS-CoV-2',1],['Not Positive to SARS-CoV-2',2],['Results Pending',3],['Results not Adequate',4]],
                                          '23.Lab Results Distribution')
                                          
'''24.ANTIGEN_SAMPLE_TAKEN(Dropped)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_29_en_num,
                                          Column('24.ANTIGEN_SAMPLE_TAKEN','Antigen Sample Taken', {'Had Taken Antigen Sample':1,'Did not take Antigen Sample':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('24.ANTIGEN_SAMPLE_TAKEN','Antigen Sample Taken', {'Had Taken Antigen Sample':1,'Did not take Antigen Sample':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Had Taken Antigen Sample',1],['Did not take Antigen Sample',2]],'24.Antigen Sample Taken Distribution')
                                          
'''25.ANTIGEN_RESULT(Dropped)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_29_en_num,
                                          Column('25.ANTIGEN_RESULT','Antigen Result', {'Positive to SARS-CoV-2':1,'Negative to SARS-CoV-2':2,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('25.ANTIGEN_RESULT','Antigen Result', {'Positive to SARS-CoV-2':1,'Negative to SARS-CoV-2':2,'not apply':97,'ignored':98,'not specified':99}),
                                          [['Positive to SARS-CoV-2',1],['Negative to SARS-CoV-2',2]],'25.Antigen Result Distribution')
                                          
'''26.FINAL_CLASSIFICATION(Dropped)'''
categorical_column_analysis_and_bar_chart(df_sars_cov_2_p_all_transf_29_en_num,df_sars_cov_2_p_all_transf_valid_29_en_num,
                                          Column('26.FINAL_CLASSIFICATION','Final Classification', {'COVID-19 Cases confirmed by C.E.A.':1,'COVID-19 Cases confirmed by D.C.':2,'Confirmed COVID-19 Cases':3,'Ivalid by Laboratory':4,'Not Performed by Laboratory':5,'Suspected Cases':6,'Negative to SARS-CoV-2':7,'not apply':97,'ignored':98,'not specified':99}),
                                          Column('26.FINAL_CLASSIFICATION','Final Classification', {'COVID-19 Cases confirmed by C.E.A.':1,'COVID-19 Cases confirmed by D.C.':2,'Confirmed COVID-19 Cases':3,'Ivalid by Laboratory':4,'Not Performed by Laboratory':5,'Suspected Cases':6,'Negative to SARS-CoV-2':7,'not apply':97,'ignored':98,'not specified':99}),
                                          [['COVID-19 Cases confirmed by C.E.A.',1],['COVID-19 Cases confirmed by D.C.',2],
                                           ['Confirmed COVID-19 Cases',3],['Ivalid by Laboratory',4],
                                           ['Not Performed by Laboratory',5],['Suspected Cases',6],['Negative to SARS-CoV-2',7]],
                                          '26.Final Classification Distribution')


'''Create a barchart for all binary columns for All samples[2.062.829 COVID-19 Positive samples]'''
col_29_dict={'03.TYPE_OF_PATIENT':'outpatient','07.INTUBATED':'intubated','08.PNEUMONIA':'pneumonia','10.PREGNANCY':'pregnant',
 '11.DIABETIC':'diabetic','12.COPD':'copd','13.ASTHMA':'asthma','14.IMMUNOSUPPRESSED':'immunosuppressed',
 '15.HYPERTENSION':'hypertension','16.OTHER_CHRONIC_DISEASE':'other chronic disease','17.CARDIOVASCULAR':'cardiovascular',
 '18.OBESITY':'obesity','19.CHRONIC_KIDNEY_FAILURE':'chronic kidney failure','20.SMOKER':'smoker',
 '21.CONTACT_WITH_COVID-19_CASE':'contact with COVID-19 case','27.ICU':'icu','29.SURVIVED':'survived'}

create_multi_df_categorical_column_list_chart(df_sars_cov_2_p_all_transf_29_en_num,col_29_dict,'All Distributions 2M (All)')


'''Create a barchart for all binary columns for Valid samples[410.362 Valid COVID-19 Positive samples]'''
col_21_dict={'03.OUTPATIENT':'outpatient','04.INTUBATED':'intubated','05.PNEUMONIA':'pneumonia','07.PREGNANCY':'pregnant',
 '08.DIABETIC':'diabetic','09.COPD':'copd','10.ASTHMA':'asthma','11.IMMUNOSUPPRESSED':'immunosuppressed',
 '12.HYPERTENSION':'hypertension','13.OTHER_CHRONIC_DISEASE':'other chronic disease','14.CARDIOVASCULAR':'cardiovascular',
 '15.OBESITY':'obesity','16.CHRONIC_KIDNEY_FAILURE':'chronic kidney failure','17.SMOKER':'smoker',
 '18.CONTACT_WITH_COVID-19_CASE':'contact with COVID-19 case','19.ICU':'icu','21.SURVIVED':'survived'}

create_multi_df_categorical_column_list_chart(df_sars_cov_2_p_all_transf_valid_21_en_num,col_21_dict,'All Distributions 400K (Valid)')

''''''



#End Time
end = time.time()
print('\n',end-start,'\n')
print(f"Runtime of the program is {end - start}")
#Total Time taken[257.42474913597107 seconds]



