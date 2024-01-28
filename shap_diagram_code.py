'''
'''
## LogisticRegression ##
# SHAP diagram #
import pandas as pd
#import numpy as np
import shap
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

start = time.time()

# Loading dataset from the CSV file
#df = pd.read_csv('files/csv/data/Covid19MPD_7_23_en_pos_fc_valid.csv')
df = pd.read_csv('files/csv/data/Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv')

# Randomly selecting a sample of the dataset
#fract_lrg= 0.50
#fract_lrg= 0.20
fract_lrg= 0.50
df = df.sample(frac= fract_lrg)

#Selecting the X variables collumns
x_column_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,20,21,22,23,24,25]
x_column_list = ['SEX','TYPE_OF_PATIENT','PNEUMONIA','AGE','DIABETIC','COPD','ASTHMA','IMMUNOSUPPRESSED','HYPERTENSION','OTHER_CHRONIC_DISEASE','CARDIOVASCULAR','OBESITY','CHRONIC_KIDNEY_FAILURE','SMOKER','DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION','INTUBATED_2','INTUBATED_97','PREGNANCY_2','PREGNANCY_97','ICU_2','ICU_97']
x_collumn_drop = ['REGISTRATION_ID', 'CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION','SURVIVED']

X= df.drop(x_collumn_drop, axis=1)
X = X.astype('int')

#Selecting the y variable-s column-s
y_column_list = [19]
y_column_list = ['SURVIVED']

y = df['SURVIVED']
y = y.astype('int')

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the LogisticRegression model
lrg = LogisticRegression(max_iter=1000, random_state=42)
#lrg = LogisticRegression(random_state=42)
lrg.fit(X_train, y_train)

# Creating a SHAP explainer
explainer = shap.LinearExplainer(lrg, X, n_jobs=-1)
#explainer = shap.Explainer(lrg, X_test)

# Replacing the Decision Tree Classifier classes' names from numerical to Strings
class_names = lrg.classes_
class_names_list = class_names.tolist()

for i in range(len(class_names_list)):
    if class_names_list[i] == 0: # 1
        class_names_list[i] = 'Class: Survived'
    if class_names_list[i] == 1: # 2
        class_names_list[i] = 'Class: Died'

# Calculating Shapley values for a set of samples
shap_values = explainer.shap_values(X_test)

# Summary plots

#Barchart
plt.title('LogisticRegression SHAP Barchart')
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X.columns, class_names=class_names_list, show=False)

timestamp_chart = datetime.now()
timestamp_chart_sec = timestamp_chart.strftime("%Y-%m-%d %H:%M:%S")
timestamp_chart_sec_final = timestamp_chart_sec.replace(':','.')

path = f'files/png/shap_diagrams/lrg/lrg_shap_bar_plot_{timestamp_chart_sec_final}_{fract_lrg*100}%.png'
plt.savefig(path)
plt.show()

#Beeswarm
plt.title('LogisticRegression SHAP Beeswarm')  # Add a custom title
shap.summary_plot(shap_values, X_test, feature_names=X.columns, class_names=class_names_list, show=False)

timestamp_chart = datetime.now()
timestamp_chart_sec = timestamp_chart.strftime("%Y-%m-%d %H:%M:%S")
timestamp_chart_sec_final = timestamp_chart_sec.replace(':','.')

path = f'files/png/shap_diagrams/lrg/lrg_shap_bsw_plot_{timestamp_chart_sec_final}_{fract_lrg*100}%.png'
plt.savefig(path)
plt.show()


end = time.time()
duration = end-start
timestamp = datetime.now()
print(f'\n[{timestamp}]\nThe LogisticRegression SHAP Diagram rendering duration was {duration} seconds\n')
#The LogisticRegression SHAP Diagram rendering duration was 241.52443385124207 seconds
'''
'''


'''
'''
## DecisionTreeClassifier ## 
# SHAP diagram # 
import pandas as pd
#import numpy as np
import shap
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



start = time.time()

# Loading dataset from the CSV file
#df = pd.read_csv('files/csv/data/Covid19MPD_7_23_en_pos_fc_valid.csv')
df = pd.read_csv('files/csv/data/Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv')

# Randomly selecting a sample of the dataset
#fract_dtc= 0.50
#fract_dtc= 0.20
fract_dtc= 0.50
df = df.sample(frac= fract_dtc)

#Selecting the X variables collumns
x_column_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,20,21,22,23,24,25]
x_column_list = ['SEX','TYPE_OF_PATIENT','PNEUMONIA','AGE','DIABETIC','COPD','ASTHMA','IMMUNOSUPPRESSED','HYPERTENSION','OTHER_CHRONIC_DISEASE','CARDIOVASCULAR','OBESITY','CHRONIC_KIDNEY_FAILURE','SMOKER','DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION','INTUBATED_2','INTUBATED_97','PREGNANCY_2','PREGNANCY_97','ICU_2','ICU_97']
x_collumn_drop = ['REGISTRATION_ID', 'CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION','SURVIVED']

X= df.drop(x_collumn_drop, axis=1)
X = X.astype('int')

#Selecting the y variable-s column-s
y_column_list = [19]
y_column_list = ['SURVIVED']

y = df['SURVIVED']
y = y.astype('int')

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the DecisionTreeClassifier
dtr = DecisionTreeClassifier(max_depth=17, random_state=42)
#bagging_model = BaggingClassifier(dtr, n_jobs=-1)
dtr.fit(X_train, y_train)
#bagging_model.fit(X_train, y_train)

# Creating a SHAP explainer
explainer = shap.TreeExplainer(dtr, X, n_jobs=-1)
#explainer = shap.TreeExplainer(bagging_model, n_jobs=-1)
#explainer = shap.Explainer(dtr, X_train)

# Replacing the DecisionTreeClassifier classes' names from numerical to Strings
class_names = dtr.classes_
class_names_list = class_names.tolist()

for i in range(len(class_names_list)):
    if class_names_list[i] == 0:#1
        class_names_list[i] = 'Class: Survived'
    if class_names_list[i] == 1:#2
        class_names_list[i] = 'Class: Died'

# Calculating Shapley values for a set of samples
shap_values = explainer.shap_values(X_test)

# Summary plots

#Barchart
plt.title('DecisionTreeClassifier SHAP Barchart')  # Add a custom title
shap.summary_plot(shap_values, X_test, feature_names=X.columns, class_names=class_names_list, show=False)

timestamp_chart = datetime.now()
timestamp_chart_sec = timestamp_chart.strftime("%Y-%m-%d %H:%M:%S")
timestamp_chart_sec_final = timestamp_chart_sec.replace(':','.')

path = f'files/png/shap_diagrams/dtc/dtc_shap_bar_plot_{timestamp_chart_sec_final}_{fract_dtc*100}%.png'
plt.savefig(path)
plt.show()

#Beeswarm
plt.title('DecisionTreeClassifier SHAP Beeswarm')  # Add a custom title
shap.summary_plot(shap_values[0], X_test, show=False)
#shap_values_b = explainer(X_test)
#shap.plots.beeswarm(shap_values_b[:,:,1], show=False)
timestamp_chart = datetime.now()
timestamp_chart_sec = timestamp_chart.strftime("%Y-%m-%d %H:%M:%S")
timestamp_chart_sec_final = timestamp_chart_sec.replace(':','.')

path = f'files/png/shap_diagrams/dtc/dtc_shap_bsw_plot_{timestamp_chart_sec_final}_{fract_dtc*100}%.png'
plt.savefig(path)
plt.show()

end = time.time()
duration = end-start
timestamp = datetime.now()
print(f'\n[{timestamp}]\nThe DecisionTreeClassifier SHAP Diagram rendering duration was {duration} seconds\n')
#The DecisionTreeClassifier SHAP Diagram rendering duration was 3370.9074263572693 seconds
'''
'''



'''
'''
## RandomForestClassifier ## 
# SHAP diagram # 
import pandas as pd
#import numpy as np
import shap
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


start = time.time()

# Loading dataset from the CSV file
#df = pd.read_csv('files/csv/data/Covid19MPD_7_23_en_pos_fc_valid.csv')
df = pd.read_csv('files/csv/data/Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv')

# Randomly selecting a sample of the dataset
#fract_rfc= 0.50
#fract_rfc= 0.20
fract_rfc= 0.50
df = df.sample(frac= fract_rfc)

#Selecting the X variables collumns
x_column_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,20,21,22,23,24,25]
x_column_list = ['SEX','TYPE_OF_PATIENT','PNEUMONIA','AGE','DIABETIC','COPD','ASTHMA','IMMUNOSUPPRESSED','HYPERTENSION','OTHER_CHRONIC_DISEASE','CARDIOVASCULAR','OBESITY','CHRONIC_KIDNEY_FAILURE','SMOKER','DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION','INTUBATED_2','INTUBATED_97','PREGNANCY_2','PREGNANCY_97','ICU_2','ICU_97']
x_collumn_drop = ['REGISTRATION_ID', 'CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION','SURVIVED']

X= df.drop(x_collumn_drop, axis=1)
X = X.astype('int')

#Selecting the y variable-s column-s
y_column_list = [19]
y_column_list = ['SURVIVED']

y = df['SURVIVED']
y = y.astype('int')

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, max_depth=13, n_jobs=-1, random_state=42)
rfc.fit(X_train, y_train)

# Creating a SHAP explainer
explainer = shap.TreeExplainer(rfc, X, n_jobs=-1)
#explainer = shap.Explainer(rfc, X_train) # 180 min

# Replacing the RandomForestClassifier classes' names from numerical to Strings
class_names = rfc.classes_
class_names_list = class_names.tolist()

for i in range(len(class_names_list)):
    if class_names_list[i] == 0:#1
        class_names_list[i] = 'Class: Survived'
    if class_names_list[i] == 1:#2
        class_names_list[i] = 'Class: Died'

# Calculating Shapley values for a set of samples
shap_values = explainer.shap_values(X_test)

# Summary plots

#Barchart
plt.title('RandomForestClassifier SHAP Barchart')  # Add a custom title
shap.summary_plot(shap_values, X_test, feature_names=X.columns, class_names=class_names_list, show=False)

timestamp_chart = datetime.now()
timestamp_chart_sec = timestamp_chart.strftime("%Y-%m-%d %H:%M:%S")
timestamp_chart_sec_final = timestamp_chart_sec.replace(':','.')

path = f'files/png/shap_diagrams/rfc/rfc_shap_bar_plot_{timestamp_chart_sec_final}_{fract_rfc*100}%.png'
plt.savefig(path)
plt.show()

#Beeswarm
plt.title('RandomForestClassifier SHAP Beeswarm')  # Add a custom title
shap.summary_plot(shap_values[0], X_test, show=False)
#shap_values_b = explainer(X_test)
#shap.plots.beeswarm(shap_values_b[:,:,1], show=False)

timestamp_chart = datetime.now()
timestamp_chart_sec = timestamp_chart.strftime("%Y-%m-%d %H:%M:%S")
timestamp_chart_sec_final = timestamp_chart_sec.replace(':','.')

path = f'files/png/shap_diagrams/rfc/rfc_shap_bsw_plot_{timestamp_chart_sec_final}_{fract_rfc*100}%.png'
plt.savefig(path)
plt.show()


end = time.time()
duration = end-start
timestamp = datetime.now()
print(f'\n[{timestamp}]\nThe RandomForestClassifier SHAP Diagram rendering duration was {duration} seconds\n')
'''
'''



'''
'''
## XGBClassifier ## 
# SHAP diagram # 
import pandas as pd
#import numpy as np
import shap
import matplotlib.pyplot as plt
import time
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


start = time.time()

# Loading dataset from the CSV file
#df = pd.read_csv('files/csv/data/Covid19MPD_7_23_en_pos_fc_valid.csv')
df = pd.read_csv('files/csv/data/Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv')

# Randomly selecting a sample of the dataset
#fract_xgbc= 0.50
#fract_xgbc= 0.20
fract_xgbc= 0.50
df = df.sample(frac= fract_xgbc)

#Selecting the X variables collumns
x_column_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,20,21,22,23,24,25]
x_column_list = ['SEX','TYPE_OF_PATIENT','PNEUMONIA','AGE','DIABETIC','COPD','ASTHMA','IMMUNOSUPPRESSED','HYPERTENSION','OTHER_CHRONIC_DISEASE','CARDIOVASCULAR','OBESITY','CHRONIC_KIDNEY_FAILURE','SMOKER','DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION','INTUBATED_2','INTUBATED_97','PREGNANCY_2','PREGNANCY_97','ICU_2','ICU_97']
x_collumn_drop = ['REGISTRATION_ID', 'CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION','SURVIVED']

X= df.drop(x_collumn_drop, axis=1)
X = X.astype('int')

#Selecting the y variable-s column-s
y_column_list = [19]
y_column_list = ['SURVIVED']

y = df['SURVIVED']
y = y.astype('int')

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the XGBClassifier
xgb = XGBClassifier(max_depth=15, n_estimators=100,random_state=42)
xgb.fit(X_train, y_train)

# Creating a SHAP explainer
explainer = shap.TreeExplainer(xgb, X, n_jobs=-1)
#explainer = shap.Explainer(xgb, X_train) #120 min


# Replacing the XGBClassifier classes' names from numerical to Strings
class_names = xgb.classes_
class_names_list = class_names.tolist()

for i in range(len(class_names_list)):
    if class_names_list[i] == 0:#1
        class_names_list[i] = 'Class: Survived'
    if class_names_list[i] == 1:#2
        class_names_list[i] = 'Class: Died'

# Calculating Shapley values for a set of samples
shap_values = explainer.shap_values(X_test)

# Summary plots

#Barchart
plt.title('XGBClassifier SHAP Barchart')  # Add a custom title
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X.columns, class_names=class_names_list, show=False)

timestamp_chart = datetime.now()
timestamp_chart_sec = timestamp_chart.strftime("%Y-%m-%d %H:%M:%S")
timestamp_chart_sec_final = timestamp_chart_sec.replace(':','.')

path = f'files/png/shap_diagrams/xgbc/xgbc_shap_bar_plot_{timestamp_chart_sec_final}_{fract_xgbc*100}%.png'
plt.savefig(path)
plt.show()

#Beeswarm
plt.title('XGBClassifier SHAP Beeswarm')  # Add a custom title
shap.summary_plot(shap_values, X_test, feature_names=X.columns, class_names=class_names_list, show=False)
#shap_values_b = explainer(X_test)
#shap.plots.beeswarm(shap_values_b[:,:,1], show=False)

timestamp_chart = datetime.now()
timestamp_chart_sec = timestamp_chart.strftime("%Y-%m-%d %H:%M:%S")
timestamp_chart_sec_final = timestamp_chart_sec.replace(':','.')

path = f'files/png/shap_diagrams/xgbc/xgbc_shap_bsw_plot_{timestamp_chart_sec_final}_{fract_xgbc*100}%.png'
plt.savefig(path)
plt.show()

end = time.time()
duration = end-start
timestamp = datetime.now()
print(f'\n[{timestamp}]\nThe XGBClassifier SHAP Diagram rendering duration was {duration} seconds\n')
'''
'''



'''
'''
## MLPClassifier ## 
# SHAP diagram # 
import pandas as pd
#import numpy as np
import shap
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


start = time.time()

# Loading dataset from the CSV file
#df = pd.read_csv('files/csv/data/Covid19MPD_7_23_en_pos_fc_valid.csv')
df = pd.read_csv('files/csv/data/Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv')

# Randomly selecting a sample of the dataset
#fract_mlpc= 0.50
#fract_mlpc= 0.20
fract_mlpc= 0.005
df = df.sample(frac= fract_mlpc)

#Selecting the X variables collumns
x_column_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,20,21,22,23,24,25]
x_column_list = ['SEX','TYPE_OF_PATIENT','PNEUMONIA','AGE','DIABETIC','COPD','ASTHMA','IMMUNOSUPPRESSED','HYPERTENSION','OTHER_CHRONIC_DISEASE','CARDIOVASCULAR','OBESITY','CHRONIC_KIDNEY_FAILURE','SMOKER','DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION','INTUBATED_2','INTUBATED_97','PREGNANCY_2','PREGNANCY_97','ICU_2','ICU_97']
x_collumn_drop = ['REGISTRATION_ID', 'CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION','SURVIVED']

X= df.drop(x_collumn_drop, axis=1)
X = X.astype('int')

#Selecting the y variable-s column-s
y_column_list = [19]
y_column_list = ['SURVIVED']

y = df['SURVIVED']
y = y.astype('int')

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the MLPClassifier
mlp = MLPClassifier()
mlp.fit(X_train, y_train)

# Creating a SHAP explainer
explainer = shap.KernelExplainer(mlp.predict_proba, X, n_jobs=-1)
#explainer = shap.Explainer(mlp, X_train)


# Replacing the MLPClassifier classes' names from numerical to Strings
class_names = mlp.classes_
class_names_list = class_names.tolist()

for i in range(len(class_names_list)):
    if class_names_list[i] == 0:#1
        class_names_list[i] = 'Class: Survived'
    if class_names_list[i] == 1:#2
        class_names_list[i] = 'Class: Died'

# Calculating Shapley values for a set of samples
shap_values = explainer.shap_values(X_test)

# Summary plots

#Barchart
plt.title('MLPClassifier SHAP Barchart')  # Add a custom title
shap.summary_plot(shap_values, X_test, feature_names=X.columns, class_names=class_names_list, show=False)
#shap.plots.beeswarm(shap_values)

timestamp_chart = datetime.now()
timestamp_chart_sec = timestamp_chart.strftime("%Y-%m-%d %H:%M:%S")
timestamp_chart_sec_final = timestamp_chart_sec.replace(':','.')

path = f'files/png/shap_diagrams/mlpc/mlpc_shap_bar_plot_{timestamp_chart_sec_final}_{fract_mlpc*100}%.png'
plt.savefig(path)
plt.show()

#Beeswarm
plt.title('MLPClassifier SHAP Beeswarm')  # Add a custom title
shap.summary_plot(shap_values[0], X_test, show=False)
#shap_values_b = explainer(X_test)
#shap.plots.beeswarm(shap_values_b[:,:,1], show=False)

timestamp_chart = datetime.now()
timestamp_chart_sec = timestamp_chart.strftime("%Y-%m-%d %H:%M:%S")
timestamp_chart_sec_final = timestamp_chart_sec.replace(':','.')

path = f'files/png/shap_diagrams/mlpc/mlpc_shap_bsw_plot_{timestamp_chart_sec_final}_{fract_mlpc*100}%.png'
plt.savefig(path)
plt.show()

end = time.time()
duration = end-start
timestamp = datetime.now()
print(f'\n[{timestamp}]\nThe MLPClassifier SHAP Diagram rendering duration was {duration} seconds\n')
'''
'''



'''

## KNeighborsClassifier ## 
# SHAP diagram # 
import pandas as pd
#import numpy as np
import shap
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


start = time.time()

# Loading dataset from the CSV file
#df = pd.read_csv('files/csv/data/Covid19MPD_7_23_en_pos_fc_valid.csv')
df = pd.read_csv('files/csv/data/Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv')

# Randomly selecting a sample of the dataset
#fract_knnc= 0.50
#fract_knnc= 0.20
fract_knnc= 0.0005
df = df.sample(frac= fract_knnc)

#Selecting the X variables collumns
x_column_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,20,21,22,23,24,25]
x_column_list = ['SEX','TYPE_OF_PATIENT','PNEUMONIA','AGE','DIABETIC','COPD','ASTHMA','IMMUNOSUPPRESSED','HYPERTENSION','OTHER_CHRONIC_DISEASE','CARDIOVASCULAR','OBESITY','CHRONIC_KIDNEY_FAILURE','SMOKER','DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION','INTUBATED_2','INTUBATED_97','PREGNANCY_2','PREGNANCY_97','ICU_2','ICU_97']
x_collumn_drop = ['REGISTRATION_ID', 'CONTACT_WITH_COVID-19_CASE','LAB_RESULT','FINAL_CLASSIFICATION','SURVIVED']

X= df.drop(x_collumn_drop, axis=1)
X = X.astype('int')

#Selecting the y variable-s column-s
y_column_list = [19]
y_column_list = ['SURVIVED']

y = df['SURVIVED']
y = y.astype('int')

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, leaf_size=30, p=1)
knn.fit(X_train, y_train)

# Creating a SHAP explainer
explainer = shap.KernelExplainer(knn.predict_proba, X, n_jobs=-1)
#explainer = shap.Explainer(knn, X_train)


# Replacing the KNeighborsClassifier classes' names from numerical to Strings
class_names = knn.classes_
class_names_list = class_names.tolist()

for i in range(len(class_names_list)):
    if class_names_list[i] == 0:#1
        class_names_list[i] = 'Class: Survived'
    if class_names_list[i] == 1:#2
        class_names_list[i] = 'Class: Died'

# Calculating Shapley values for a set of samples
shap_values = explainer.shap_values(X_test)

# Summary plots

#Barchart
plt.title('KNeighborsClassifier SHAP Barchart')  # Add a custom title
shap.summary_plot(shap_values, X_test, feature_names=X.columns, class_names=class_names_list, show=False)

timestamp_chart = datetime.now()
timestamp_chart_sec = timestamp_chart.strftime("%Y-%m-%d %H:%M:%S")
timestamp_chart_sec_final = timestamp_chart_sec.replace(':','.')

path = f'files/png/shap_diagrams/knnc/knnc_shap_bar_plot_{timestamp_chart_sec_final}_{fract_knnc*100}%.png'
plt.savefig(path)
plt.show()

#Beeswarm
plt.title('KNeighborsClassifier SHAP Beeswarm')  # Add a custom title
shap.summary_plot(shap_values[0], X_test, show=False)
#shap_values_b = explainer(X_test)
#shap.plots.beeswarm(shap_values_b[:,:,1], show=False)

timestamp_chart = datetime.now()
timestamp_chart_sec = timestamp_chart.strftime("%Y-%m-%d %H:%M:%S")
timestamp_chart_sec_final = timestamp_chart_sec.replace(':','.')

path = f'files/png/shap_diagrams/knnc/knnc_shap_bsw_plot_{timestamp_chart_sec_final}_{fract_knnc*100}%.png'
plt.savefig(path)
plt.show()

end = time.time()
duration = end-start
timestamp = datetime.now()
print(f'\n[{timestamp}]\nThe KNeighborsClassifier SHAP Diagram rendering duration was {duration} seconds\n')

'''