#import pandas as pd
#import numpy as np

#from classes.Column import Column

from csv import writer
from csv import DictWriter

import time
from datetime import datetime

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

Visualizing Decision Trees:
pip install graphviz
pip install pydotplus

Installing XGBoost Regression:
pip install xgboost

'''



'''Write to File Functions'''

#Append a list as a row to a CSV file
def append_list_as_row(file_name,row_list):
    
    with open(file_name, 'a+', newline='') as write_obj:
        
        csv_writer = writer(write_obj)
        
        csv_writer.writerow(row_list)


#Append a dict as a row to a CSV file
def append_dict_as_row(file_name,row_dict,field_names):
    
    with open(file_name,'a+',newline='') as write_obj:
        
        dict_writer = DictWriter(write_obj,fieldnames = field_names)
        
        dict_writer.writerow(row_dict) 


#Write the Optimal Hyperparameter Classification Report's String to a *.txt File(and return the duration{end-start})
def write_string_in_text_file(file_path,report_text,start):
    
    duration_text = ''
    
    full_file_path = file_path + '.txt'
    
    with open(full_file_path, 'a+') as file_object:
        
        file_object.seek(0)
        
        data = file_object.read(100)
        
        if len(data) > 0 :
            
            file_object.write('\n'+'\n'+'\n')
            
        file_object.write(report_text)
        
        timestamp = datetime.now()
        
        end = time.time()
        
        duration = end - start
        
        duration_text = f'Runtime: {duration} seconds\n\n[{timestamp}]\n==============================================================='
        
        file_object.write(duration_text)
        
        full_text = '\n\n\n'+ report_text + duration_text
        
    return full_text,duration

''''''
