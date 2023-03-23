import pandas as pd
import numpy as np

#from classes.Column import Column

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



'''Cleaning Dataset Functions'''

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


#All  dataframe cleaning methods in one
def df_cleaning(rel_path,df_file_name_common,file_extension,invalid_list):
    
    df_40_en,df_40_en_path = df_40_en_conversion(rel_path,df_file_name_common,file_extension)
    
    df_27_en,df_27_en_path = df_27_en_conversion(rel_path,df_file_name_common,df_40_en_path,file_extension)
    
    df_29_en,df_29_en_path = df_29_en_conversion(rel_path,df_file_name_common,df_27_en_path,file_extension)
    
    df_24_en,df_24_en_path = df_24_en_conversion(rel_path,df_file_name_common,df_29_en_path,file_extension)
    
    df_24_en_pos,df_24_en_pos_path = df_24_en_pos_conversion(rel_path,df_file_name_common,df_24_en_path,file_extension)
    
    df_23_en_pos_fc,df_23_en_pos_fc_path = df_23_en_pos_fc_lr_conversion(rel_path,df_file_name_common,df_24_en_path,file_extension,'fc')
    
    df_23_en_pos_lr,df_23_en_pos_lr_path = df_23_en_pos_fc_lr_conversion(rel_path,df_file_name_common,df_24_en_path,file_extension,'lr')
    
    df_23_en_pos_fc_valid,df_23_en_pos_fc_valid_path = df_23_en_pos_fc_lr_valid_conversion(rel_path,df_file_name_common,df_23_en_pos_fc_path,file_extension,'fc',invalid_list)
    
    df_23_en_pos_lr_valid,df_23_en_pos_lr_valid_path = df_23_en_pos_fc_lr_valid_conversion(rel_path,df_file_name_common,df_23_en_pos_lr_path,file_extension,'lr',invalid_list)
    
    return df_40_en,df_27_en,df_29_en,df_24_en,df_24_en_pos,df_23_en_pos_fc,df_23_en_pos_lr,df_23_en_pos_fc_valid,df_23_en_pos_lr_valid


''''''
