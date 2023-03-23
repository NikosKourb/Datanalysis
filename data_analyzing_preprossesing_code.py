import pandas as pd
#import numpy as np
import time

from classes.Column import Column

from functions.preprocess.cleaning import df_40_en_conversion
from functions.preprocess.cleaning import df_27_en_conversion
from functions.preprocess.cleaning import df_29_en_conversion
from functions.preprocess.cleaning import df_24_en_conversion
from functions.preprocess.cleaning import df_24_en_pos_conversion
from functions.preprocess.cleaning import df_23_en_pos_fc_lr_conversion
from functions.preprocess.cleaning import df_23_en_pos_fc_lr_valid_conversion
#from functions.preprocess.cleaning import filter_values_of_a_column
#from functions.preprocess.cleaning import filter_values_of_multi_columns
from functions.preprocess.cleaning import df_cleaning

from functions.preprocess.preparing import df_preprocessing

from functions.analyzing import show_column_info
#from functions.analyzing import show_all_column_info
#from functions.analyzing import categorical_column_analysis
#from functions.analyzing import create_categorical_column_bar_chart
#from functions.analyzing import create_multi_df_categorical_column_bar_chart
from functions.analyzing import create_multi_categorical_column_list_chart
#from functions.analyzing import create_multi_df_numerical_column_bar_chart
from functions.analyzing import numerical_column_analysis
#from functions.analyzing import ages_column_barchart
from functions.analyzing import ages_column_analysis_and_barchart
#from functions.analyzing import categorical_column_analysis_and_bar_chart
from functions.analyzing import df_compare_categorical_column_analysis_and_bar_chart



'''Dataset Description''''''

GENERAL VALUES:
97 ------> NO APLICA (DOES NOT APPLY)
98 ------> SE IGNORA (IT IS IGNORED)
99 ------> NO ESPECIFICADO (NOT SPECIFIED)



COLUMN TITLES:
    
-ALREADY EXISTED-
01.FECHA_ACTUALIZACION-----> DATE_OF_DATA_UPDATE (AAAA-MM-DD)
02.ID_REGISTRO--------(00)-> REGISTRATION_ID (Text ex. zz7202)
03.ORIGEN------------------> ORIGIN (Whether or not the patient diagnosed and or hospitalized to a USER medical Unit or not [1:USMER, 2:NOT USMER])
04.SECTOR------------------> HEALTHCARE_SECTOR (The type of institution of the National Health System that provided the care [1:RED CROSS, 2:DIF, 3:STATE, 4:IMSS, 5:IMSS-BIENESTAR, 6:ISSSTE, 7:MUNICIPAL, 8:PEMEX, 9:PRIVATE, 10:SEDENA, 11:SEMAR, 12:SSA , 13:UNIVERSITY])
05.ENTIDAD_UM--------------> ENTITY_OF_MEDICAL_UNIT (The entity where the medical unit that provided the care is located [CATALOG of the diffenet Federal Health Entities 1:AGUASCALIENTES, 2:BAJA CALIFORNIA, 3:BAJA CALIFORNIA SOUTH, 4:CAMPECHE, 5:COAHUILA DE ZARAGOZA e.t.c.])
06.SEXO---------------(01)-> SEX (Patient's sex [1:Woman, 2: Man])
07.ENTIDAD_NAC-------------> PATIENT'S_BIRTH_HEALTHCARE_ENTITY (The patient's health care entity of birth [CATALOG of the diffenet Federal Health Entities 1:AGUASCALIENTES, 2:BAJA CALIFORNIA, 3:BAJA CALIFORNIA SOUTH, 4:CAMPECHE, 5:COAHUILA DE ZARAGOZA e.t.c.])
08.ENTIDAD_RES-------------> PATIENT_RESIDENCE'S_HEALTHCARE_ENTITY (The health care entity that the patient's residence belongs [CATALOG of the diffenet Federal Health Entities 1:AGUASCALIENTES, 2:BAJA CALIFORNIA, 3:BAJA CALIFORNIA SOUTH, 4:CAMPECHE, 5:COAHUILA DE ZARAGOZA e.t.c.])
09.MUNICIPIO_RES-----------> PATIENT_RESIDENCE'S_MUNICIPALITY (The municipality of residence of the patient [CATALOG of the diffenet Municipalities 1:AGUASCALIENTES, 2:ASIENTOS, 3:CALVILLO, 4:COSÍO, 5:JESÚS MARÍA e.t.c.])
10.TIPO_PACIENTE------(02)-> TYPE_OF_PATIENT (The type of care the patient received in the unit. It is called an outpatient if you returned home or it is called an inpatient if you were admitted to the hospital [1:OUTPATIENT-AMBULATORY, 2:INPATIENT-HOSPITALIZED]) 
11.FECHA_INGRESO------(21)-> HOSPITAL_ADMISSION_DATE (AAAA-MM-DD)
12.FECHA_SINTOMAS-----(21)-> SYMPTOM_ONSET_DATE (The date on which the patient's symptoms began [AAAA-MM-DD])
13.FECHA_DEF----------(22)-> DATE_OF_DEATH (The date the patient died [AAAA-MM-DD, exception 9999-99-99: Survived])
14.INTUBADO-----------(03)-> INTUBATED (Whether or not the patient required intubation [1:YES, 2:NO])
15.NEUMONIA-----------(04)-> PNEUMONIA (Whether or not the patient was diagnosed with pneumonia [1:YES, 2:NO])
16.EDAD---------------(05)-> AGE (Patient's Age)
17.NACIONALIDAD------------> NATIONALITY (Patient's Nationality [1:MEXICAN, 2:FOREIGN])
18.EMBARAZO-----------(06)-> PREGNANCY (Whether or not the patient is pregnant [1:YES, 2:NO])
19.HABLA_LENGUA_INDIG------> NATIVE_LANGUAGE_SPEAKER (Whether or not the patient speaks an indigenous language [1:YES, 2:NO])
20.INDIGENA----------------> INDIGENOUS (Whether or not the patient identifies himself as an indigenous person [1:YES, 2:NO])
21.DIABETES-----------(07)-> DIABETIC (Whether or not the patient has been diagnosed with diabetes [1:YES, 2:NO])  
22.EPOC---------------(08)-> COPD (Whether or not the patient has been diagnosed with COPD[ΧΑΠ] [1:YES, 2:NO])
23.ASMA---------------(09)-> ASTHMA (Whether or not the patient has been diagnosed with ASTHMA [1:YES, 2:NO])
24.INMUSUPR-----------(10)-> IMMUNOSUPPRESSED (Whether or not the patient is immunosuppressed [1:YES, 2:NO])
25.HIPERTENSION-------(11)-> HYPERTENSION (Whether or not the patient has been diagnosed with hypertension [1:YES, 2:NO])
26.OTRA_COM-----------(12)-> OTHER_CHRONIC_DISEASE (Whether or not the patient has been diagnosed with any other disease [1:YES, 2:NO])
27.CARDIOVASCULAR-----(13)-> CARDIOVASCULAR (Whether or not the patient has been diagnosed with cardiovascular disease [1:YES, 2:NO])
28.OBESIDAD-----------(14)-> OBESITY (Whether or not the patient has been diagnosed with obesity [1:YES, 2:NO])
29.RENAL_CRONICA------(15)-> CHRONIC_KIDNEY_FAILURE(Whether or not the patient has been diagnosed with chronic kidney failure [1:YES, 2:NO])
30.TABAQUISMO---------(16)-> SMOKER (Whether or not the patient has a smoking habit [1:YES, 2:NO])
31.OTRO_CASO----------(17)-> CONTACT_WITH_COVID-19_CASE (Whether or not the patient had contact with any other case diagnosed with SARS CoV-2 [1:YES, 2:NO])
32.TOMA_MUESTRA_LAB--------> LAB_SAMPLE_TAKEN (Whether or not the patient had a laboratory sample taken [1:YES, 2:NO])
33.RESULTADO_LAB------(18)-> LAB_RESULT (The result of the analysis of the sample reported by the laboratory [1:POSITIVE TO SARS-CoV-2, 2:NOT POSITIVE TO SARS-CoV-2, 3:RESULT PENDING, 4:RESULT NOT ADEQUATE]
34.TOMA_MUESTRA_ANTIGENO---> ANTIGEN_SAMPLE_TAKEN (Whether or not the patient had an antigen sample for SARS-CoV-2 taken [1:YES, 2:NO])
35.RESULTADO_ANTIGENO------> ANTIGEN_RESULT (The result of the analysis of the antigen sample taken from the patient [1:POSITIVE TO SARS-COV-2, 2:NEGATIVE TO SARS-COV-2])
36.CLASIFICACION_FINAL(19)-> FINAL_CLASSIFICATION (If the patient is a case of COVID-19 according to the catalog [1:CASE OF COVID-19 CONFIRMED BY CLINICAL EPIDEMIOLOGICAL ASSOCIATION, 2:CASE OF COVID-19 CONFIRMED BY DICTAMINATION COMMITTEE, 3:CASE OF SARS-COV-2 CONFIRMED, 4:INVALID BY LABORATORY, 5:NOT PERFORMED BY LABORATORY 6:SUSPECT CASE, 7:NEGATIVE TO SARS-COV-2]) 
37.MIGRANTE----------------> IMMIGRANT (Whether or not the patient is a immigrant [1:YES, 2:NO])
38.PAIS_NACIONALIDAD-------> NATIONALITY (The nationality of the patient [TEXT]) 
39.PAIS_ORIGEN-------------> COUNTRY_OF_ORIGIN (The country from which the patient left for Mexico [TEXT]) 
40.UCI----------------(20)-> ICU (Whether or not the patient required admission to an Intensive Care Unit [1:YES, 2:NO])

-CREATED-
SYMPTOM_ONSET_DATE-HOSPITAL_ADMISSION_DATE----(21)-> DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION
DATE_OF_DEATH---------------------------------(22)-> SURVIVED [1:YES, 2:NO]

'''



#Start Time
start = time.time()

rel_path = 'files/csv/data/'


'''Dataframes'''


'''Cleaning Data'''

df_col_list_dict_sp={'TIPO_PACIENTE':'outpatient','INTUBADO':'intubated','NEUMONIA':'pneumonia',
                     'EMBARAZO':'pregnant','DIABETES':'diabetic','EPOC':'copd','ASMA':'asthma',
                     'INMUSUPR':'immunosuppressed','HIPERTENSION':'hypertension',
                     'OTRA_COM':'other chronic disease','CARDIOVASCULAR':'cardiovascular',
                     'OBESIDAD':'obesity','RENAL_CRONICA':'chronic kidney failure','TABAQUISMO':'smoker',
                     'OTRO_CASO':'contact with COVID-19 case','UCI':'icu'}

df_col_list_dict_en={'TYPE_OF_PATIENT':'outpatient','INTUBATED':'intubated','PNEUMONIA':'pneumonia',
                     'PREGNANCY':'pregnant','DIABETIC':'diabetic','COPD':'copd','ASTHMA':'asthma',
                     'IMMUNOSUPPRESSED':'immunosuppressed','HYPERTENSION':'hypertension',
                     'OTHER_CHRONIC_DISEASE':'other chronic disease','CARDIOVASCULAR':'cardiovascular',
                     'OBESITY':'obesity','CHRONIC_KIDNEY_FAILURE':'chronic kidney failure','SMOKER':'smoker',
                     'CONTACT_WITH_COVID-19_CASE':'contact with COVID-19 case','ICU':'icu'}



df_col_list_dict_en_final = df_col_list_dict_en.copy()
df_col_list_dict_en_final['SURVIVED'] = 'survived'


df_40_sp = pd.read_csv(rel_path + 'Covid19MPD.csv',header=0)
df_40_sp.name='df_40_sp'
show_column_info(df_40_sp,'ID_REGISTRO') # 12.425.179 rows
create_multi_categorical_column_list_chart(df_40_sp,df_col_list_dict_sp,'All Distributions (12M All 40_sp)')

#df_40_en,df_40_en_path = df_40_en_conversion(rel_path,'Covid19MPD','.csv')
df_40_en = pd.read_csv(rel_path + 'Covid19MPD_1_40_en.csv',header=0)
df_40_en.name='df_40_en'
show_column_info(df_40_en,'REGISTRATION_ID') # 12.425.179 rows
create_multi_categorical_column_list_chart(df_40_en,df_col_list_dict_en,'All Distributions (12M All 40_en)')

#df_27_en,df_27_en_path = df_27_en_conversion(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_1_40_en.csv','.csv')
df_27_en = pd.read_csv(rel_path + 'Covid19MPD_2_27_en.csv',header=0)
df_27_en.name='df_27_en'
show_column_info(df_27_en,'REGISTRATION_ID') # 12.425.179 rows
create_multi_categorical_column_list_chart(df_27_en,df_col_list_dict_en,'All Distributions (12M All 27_en)')

#df_29_en,df_29_en_path = df_29_en_conversion(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_2_27_en.csv','.csv')
df_29_en = pd.read_csv(rel_path + 'Covid19MPD_3_29_en.csv',header=0)
df_29_en.name='df_29_en'
show_column_info(df_29_en,'REGISTRATION_ID') # 12.425.179 rows
create_multi_categorical_column_list_chart(df_29_en,df_col_list_dict_en_final,'All Distributions (12M All 29_en)')

#df_24_en,df_24_en_path = df_24_en_conversion(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_3_29_en.csv','.csv')
df_24_en = pd.read_csv(rel_path + 'Covid19MPD_4_24_en.csv',header=0)
df_24_en.name='df_24_en'
show_column_info(df_24_en,'REGISTRATION_ID') # 12.425.179 rows
create_multi_categorical_column_list_chart(df_24_en,df_col_list_dict_en_final,'All Distributions (12M All 24_en)')

#df_24_en_pos,df_24_en_pos_path = df_24_en_pos_conversion(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_4_24_en.csv','.csv')
df_24_en_pos = pd.read_csv(rel_path + 'Covid19MPD_5_24_en_pos.csv',header=0)
df_24_en_pos.name='df_24_en_pos'
show_column_info(df_24_en_pos,'REGISTRATION_ID') # 3.997.697 rows
create_multi_categorical_column_list_chart(df_24_en_pos,df_col_list_dict_en_final,'All Distributions (4M All 24_en_pos)')

#df_23_en_pos_fc,df_23_en_pos_fc_path = df_23_en_pos_fc_lr_conversion(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_5_24_en_pos.csv','.csv','fc')
df_23_en_pos_fc = pd.read_csv(rel_path + 'Covid19MPD_6_23_en_pos_fc.csv',header=0)
df_23_en_pos_fc.name='df_23_en_pos_fc'
show_column_info(df_23_en_pos_fc,'REGISTRATION_ID') # 3.993.464 rows
create_multi_categorical_column_list_chart(df_23_en_pos_fc,df_col_list_dict_en_final,'All Distributions (4M All 23_en_pos_fc)')

#df_23_en_pos_lr,df_23_en_pos_lr_path = df_23_en_pos_fc_lr_conversion(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_5_24_en_pos.csv','.csv','lr')
df_23_en_pos_lr = pd.read_csv(rel_path + 'Covid19MPD_6_23_en_pos_lr.csv',header=0)
df_23_en_pos_lr.name='df_23_en_pos_lr'
show_column_info(df_23_en_pos_lr,'REGISTRATION_ID') # 2.062.829 rows
create_multi_categorical_column_list_chart(df_23_en_pos_lr,df_col_list_dict_en_final,'All Distributions (2M All 23_en_pos_lr)')

#df_23_en_pos_fc_valid,df_23_en_pos_fc_valid_path = df_23_en_pos_fc_lr_valid_conversion(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_6_23_en_pos_fc.csv','.csv','fc',[98,99])
df_23_en_pos_fc_valid = pd.read_csv(rel_path + 'Covid19MPD_7_23_en_pos_fc_valid.csv',header=0)
df_23_en_pos_fc_valid.name='df_23_en_pos_fc_valid'
show_column_info(df_23_en_pos_fc_valid,'REGISTRATION_ID') # 3.809.119 rows
create_multi_categorical_column_list_chart(df_23_en_pos_fc_valid,df_col_list_dict_en_final,'All Distributions (4M All 23_en_pos_fc_valid)')

#df_23_en_pos_lr_valid,df_23_en_pos_lr_valid_path = df_23_en_pos_fc_lr_valid_conversion(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_6_23_en_pos_lr.csv','.csv','lr',[98,99])
df_23_en_pos_lr_valid = pd.read_csv(rel_path + 'Covid19MPD_7_23_en_pos_lr_valid.csv',header=0)
df_23_en_pos_lr_valid.name='df_23_en_pos_lr_valid'
show_column_info(df_23_en_pos_lr_valid,'REGISTRATION_ID') # 1.912.608 rows
create_multi_categorical_column_list_chart(df_23_en_pos_lr_valid,df_col_list_dict_en_final,'All Distributions (2M All 23_en_pos_lr_valid)')
''''''


'''Preparing Data'''

'''Final Classification Positives'''
#No Scaler
#df_23_en_pos_fc_valid_lb_none,df_23_en_pos_fc_valid_lb_none_path = df_preprocessing(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_7_23_en_pos_fc_valid.csv','.csv','fc','none',0)
df_23_en_pos_fc_valid_lb_none = pd.read_csv(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv',header=0)
df_23_en_pos_fc_valid_lb_none.name='df_23_en_pos_fc_valid_lb_none'
show_column_info(df_23_en_pos_fc_valid_lb_none,'REGISTRATION_ID') # 3.809.119 rows

#Standard Scaler
#df_23_en_pos_fc_valid_lb_std,df_23_en_pos_fc_valid_lb_std_path = df_preprocessing(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_7_23_en_pos_fc_valid.csv','.csv','fc','std',0)
df_23_en_pos_fc_valid_lb_std = pd.read_csv(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_std.csv',header=0)
df_23_en_pos_fc_valid_lb_std.name='df_23_en_pos_fc_valid_lb_std'
show_column_info(df_23_en_pos_fc_valid_lb_std,'REGISTRATION_ID') # 3.809.119 rows

#MinMax Scaler 0-1
#df_23_en_pos_fc_valid_lb_mm_1,df_23_en_pos_fc_valid_lb_mm_1_path = df_preprocessing(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_7_23_en_pos_fc_valid.csv','.csv','fc','mm',1)
df_23_en_pos_fc_valid_lb_mm_1 = pd.read_csv(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_mm_0-1.csv',header=0)
df_23_en_pos_fc_valid_lb_mm_1.name='df_23_en_pos_fc_valid_lb_mm_0-1'
show_column_info(df_23_en_pos_fc_valid_lb_mm_1,'REGISTRATION_ID') # 3.809.119 rows

#MinMax Scaler 0-10
#df_23_en_pos_fc_valid_lb_mm_10,df_23_en_pos_fc_valid_lb_mm_10_path = df_preprocessing(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_7_23_en_pos_fc_valid.csv','.csv','fc','mm',10)
df_23_en_pos_fc_valid_lb_mm_10 = pd.read_csv(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_mm_0-10.csv',header=0)
df_23_en_pos_fc_valid_lb_mm_10.name='df_23_en_pos_fc_valid_lb_mm_0-10'
show_column_info(df_23_en_pos_fc_valid_lb_mm_10,'REGISTRATION_ID') # 3.809.119 rows

#MinMax Scaler 0-100
#df_23_en_pos_fc_valid_lb_mm_100,df_23_en_pos_fc_valid_lb_mm_100_path = df_preprocessing(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_7_23_en_pos_fc_valid.csv','.csv','fc','mm',100)
df_23_en_pos_fc_valid_lb_mm_100 = pd.read_csv(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_mm_0-100.csv',header=0)
df_23_en_pos_fc_valid_lb_mm_100.name='df_23_en_pos_fc_valid_lb_mm_0-100'
show_column_info(df_23_en_pos_fc_valid_lb_mm_100,'REGISTRATION_ID') # 3.809.119 rows

#MinMax Scaler 0-1000
#df_23_en_pos_fc_valid_lb_mm_1000,df_23_en_pos_fc_valid_lb_mm_1000_path = df_preprocessing(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_7_23_en_pos_fc_valid.csv','.csv','fc','mm',1000)
df_23_en_pos_fc_valid_lb_mm_1000 = pd.read_csv(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_mm_0-1000.csv',header=0)
df_23_en_pos_fc_valid_lb_mm_1000.name='df_23_en_pos_fc_valid_lb_mm_0-1000'
show_column_info(df_23_en_pos_fc_valid_lb_mm_1000,'REGISTRATION_ID') # 3.809.119 rows


'''Lab Results Positives'''
#None
#df_23_en_pos_lr_valid_lb_none,df_23_en_pos_lr_valid_lb_none_path = df_preprocessing(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_7_23_en_pos_lr_valid.csv','.csv','lr','none',0)
df_23_en_pos_lr_valid_lb_none = pd.read_csv(rel_path + 'Covid19MPD_8_23_en_pos_lr_valid_lb_none.csv',header=0)
df_23_en_pos_lr_valid_lb_none.name='df_23_en_pos_lr_valid_lb_none'
show_column_info(df_23_en_pos_lr_valid_lb_none,'REGISTRATION_ID') # 1.912.608 rows

#Standard Scaler
#df_23_en_pos_lr_valid_lb_std,df_23_en_pos_lr_valid_lb_std_path = df_preprocessing(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_7_23_en_pos_lr_valid.csv','.csv','lr','std',0)
df_23_en_pos_lr_valid_lb_std = pd.read_csv(rel_path + 'Covid19MPD_8_23_en_pos_lr_valid_lb_std.csv',header=0)
df_23_en_pos_lr_valid_lb_std.name='df_23_en_pos_lr_valid_lb_std'
show_column_info(df_23_en_pos_lr_valid_lb_std,'REGISTRATION_ID') # 1.912.608 rows

#MinMax Scaler 0-1
#df_23_en_pos_lr_valid_lb_mm_1,df_23_en_pos_lr_valid_lb_mm_1_path = df_preprocessing(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_7_23_en_pos_lr_valid.csv','.csv','lr','mm',1)
df_23_en_pos_lr_valid_lb_mm_1 = pd.read_csv(rel_path + 'Covid19MPD_8_23_en_pos_lr_valid_lb_mm_0-1.csv',header=0)
df_23_en_pos_lr_valid_lb_mm_1.name='df_23_en_pos_lr_valid_lb_mm_0-1'
show_column_info(df_23_en_pos_lr_valid_lb_mm_1,'REGISTRATION_ID') # 1.912.608 rows

#MinMax Scaler 0-10
#df_23_en_pos_lr_valid_lb_mm_10,df_23_en_pos_lr_valid_lb_mm_10_path = df_preprocessing(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_7_23_en_pos_lr_valid.csv','.csv','lr','mm',10)
df_23_en_pos_lr_valid_lb_mm_10 = pd.read_csv(rel_path + 'Covid19MPD_8_23_en_pos_lr_valid_lb_mm_0-10.csv',header=0)
df_23_en_pos_lr_valid_lb_mm_10.name='df_23_en_pos_lr_valid_lb_mm_0-10'
show_column_info(df_23_en_pos_lr_valid_lb_mm_10,'REGISTRATION_ID') # 1.912.608 rows

#MinMax Scaler 0-100
#df_23_en_pos_lr_valid_lb_mm_100,df_23_en_pos_lr_valid_lb_mm_100_path = df_preprocessing(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_7_23_en_pos_lr_valid.csv','.csv','lr','mm',100)
df_23_en_pos_lr_valid_lb_mm_100 = pd.read_csv(rel_path + 'Covid19MPD_8_23_en_pos_lr_valid_lb_mm_0-100.csv',header=0)
df_23_en_pos_lr_valid_lb_mm_100.name='df_23_en_pos_lr_valid_lb_mm_0-100'
show_column_info(df_23_en_pos_lr_valid_lb_mm_100,'REGISTRATION_ID') # 1.912.608 rows

#MinMax Scaler 0-1000
#df_23_en_pos_lr_valid_lb_mm_1000,df_23_en_pos_lr_valid_lb_mm_1000_path = df_preprocessing(rel_path,'Covid19MPD',rel_path + 'Covid19MPD_7_23_en_pos_lr_valid.csv','.csv','lr','mm',1000)
df_23_en_pos_lr_valid_lb_mm_1000 = pd.read_csv(rel_path + 'Covid19MPD_8_23_en_pos_lr_valid_lb_mm_0-1000.csv',header=0)
df_23_en_pos_lr_valid_lb_mm_1000.name='df_23_en_pos_lr_valid_lb_mm_0-1000'
show_column_info(df_23_en_pos_lr_valid_lb_mm_1000,'REGISTRATION_ID') # 1.912.608 rows

''''''



'''Column Analyis & Chart Ploting'''

#00.REGISTRATION_ID
show_column_info(df_23_en_pos_fc_valid,'REGISTRATION_ID') # 3.809.119 rows
show_column_info(df_23_en_pos_lr_valid,'REGISTRATION_ID') # 1.912.608 rows

#01.SEX
df_compare_categorical_column_analysis_and_bar_chart(Column('SEX','Sexes', {'Women':1,'Men':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Gender Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Gender Distribution (2M All pos_lr_valid)')

#02.TYPE_OF_PATIENT
df_compare_categorical_column_analysis_and_bar_chart(Column('TYPE_OF_PATIENT','Type of Patient', {'Outpatient-Ambulatory':1,'Inpatient-Hospitalized':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Type of Patient Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Type of Patient Distribution (2M All pos_lr_valid)')

#03.INTUBATED
df_compare_categorical_column_analysis_and_bar_chart(Column('INTUBATED','Intubated', {'Intubated':1,'Not Intubated':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Intubated Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Intubated Distribution (2M All pos_lr_valid)')

#04.PNEUMONIA
df_compare_categorical_column_analysis_and_bar_chart(Column('PNEUMONIA','Pneumonia', {'Have Pneumonia':1,'Do not have Pneumonia':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Pneumonia Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Pneumonia Distribution (2M All pos_lr_valid)')

#05.AGE
ages_column_analysis_and_barchart(df_23_en_pos_fc_valid,'AGE','SEX','Age & Gender Distribution (4M All pos_fc_valid)',
                                  df_23_en_pos_lr_valid,'AGE','SEX','Age & Gender Distribution (2M All pos_lr_valid)')

#06.PREGNANCY
df_compare_categorical_column_analysis_and_bar_chart(Column('PREGNANCY','Pregnancy', {'Pregnant':1,'Not Pregnant':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Pregnancy Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Pregnancy Distribution (2M All pos_lr_valid)')

#07.DIABETIC
df_compare_categorical_column_analysis_and_bar_chart(Column('DIABETIC','Diabetic', {'Diabetic':1,'Non Diabetic':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Diabetics Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Diabetics Distribution (2M All pos_lr_valid)')

#08.COPD
df_compare_categorical_column_analysis_and_bar_chart(Column('COPD','COPD', {'Have COPD':1,'Do not have COPD':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'COPD Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'COPD Distribution (2M All pos_lr_valid)')

#09.ASTHMA
df_compare_categorical_column_analysis_and_bar_chart(Column('ASTHMA','Asthma', {'Have Asthma':1,'Do not have Asthma':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Asthma Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Asthma Distribution (2M All pos_lr_valid)')

#10.IMMUNOSUPPRESSED
df_compare_categorical_column_analysis_and_bar_chart(Column('IMMUNOSUPPRESSED','Immunosuppressed', {'Immunosuppressed':1,'Not Immunosuppressed':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,' Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,' Distribution (2M All pos_lr_valid)')

#11.HYPERTENSION
df_compare_categorical_column_analysis_and_bar_chart(Column('HYPERTENSION','Hypertension', {'Have Hypertension':1,'Do not have Hypertension':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Hypertension Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Hypertension Distribution (2M All pos_lr_valid)')

#12.OTHER_CHRONIC_DISEASE
df_compare_categorical_column_analysis_and_bar_chart(Column('OTHER_CHRONIC_DISEASE','Other Chronic Disease', {'Have Other Chronic Disease':1,'Do not have Other Chronic Disease':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Other Chronic Disease Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Other Chronic Disease Distribution (2M All pos_lr_valid)')

#13.CARDIOVASCULAR
df_compare_categorical_column_analysis_and_bar_chart(Column('CARDIOVASCULAR','Cardiovascular Disease', {'Have Cardiovascular Disease':1,'Do not have Cardiovascular Disease':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Cardiovascular Disease Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Cardiovascular Disease Distribution (2M All pos_lr_valid)')

#14.OBESITY
df_compare_categorical_column_analysis_and_bar_chart(Column('OBESITY','Obesity', {'Obese':1,'Not Obese':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Obesity Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Obesity Distribution (2M All pos_lr_valid)')

#15.CHRONIC_KIDNEY_FAILURE
df_compare_categorical_column_analysis_and_bar_chart(Column('CHRONIC_KIDNEY_FAILURE','Chronic Kidney Failure', {'Have Chronic Kidney Failure':1,'Do not have Chronic Kidney Failure':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Chronic Kidney Failure Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Chronic Kidney Failure Distribution (2M All pos_lr_valid)')

#16.SMOKER
df_compare_categorical_column_analysis_and_bar_chart(Column('SMOKER','Smokers', {'Smokers':1,'Non Smokers':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Smoker Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Smoker Distribution (2M All pos_lr_valid)')

#17.CONTACT_WITH_COVID-19_CASE
df_compare_categorical_column_analysis_and_bar_chart(Column('CONTACT_WITH_COVID-19_CASE','Contact with COVID-19 Case', {'Had Contact with COVID-19 Case':1,'Did not have Contact with COVID-19 Case':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Contact with COVID-19 Case Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Contact with COVID-19 Case Distribution (2M All pos_lr_valid)')

#18.LAB_RESULT
df_compare_categorical_column_analysis_and_bar_chart(Column('LAB_RESULT','Lab Result', {'Positive':1,'Not Positive':2,'Pending':3,'Not Adequate':4,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Lab Result Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Lab Result Distribution (2M All pos_lr_valid)')

#19.FINAL_CLASSIFICATION
df_compare_categorical_column_analysis_and_bar_chart(Column('FINAL_CLASSIFICATION','Final Classification', {'Case Confirmed by Epid.Assoc.':1,'Case Confirmed by Dict.Commitee':2,'Case Confirmed':3,'Invalid':4,'Not Performed by Lab':5,'Suspected Case':6,'Negative to SARS-CoV-2':7,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Final Classification Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Final Classification Distribution (2M All pos_lr_valid)')

#20.ICU
df_compare_categorical_column_analysis_and_bar_chart(Column('ICU','ICU', {'Needed ICU':1,'Did not need ICU':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'ICU Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'ICU Distribution (2M All pos_lr_valid)')

#21.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION
numerical_column_analysis(df_23_en_pos_fc_valid,'DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION')
numerical_column_analysis(df_23_en_pos_lr_valid,'DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION')

#22.SURVIVED
df_compare_categorical_column_analysis_and_bar_chart(Column('SURVIVED','Survived', {'Survived':1,'Died':2,'not apply':97,'ignored':98,'not specified':99}),
                                                     df_23_en_pos_fc_valid,'Survivors Distribution (4M All pos_fc_valid)',
                                                     df_23_en_pos_lr_valid,'Survivors Distribution (2M All pos_lr_valid)')

''''''

#End Time
end = time.time()
print(f'\nRuntime: {end-start} seconds\n')
#Total Time taken[000.00 seconds]