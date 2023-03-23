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
#from imblearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('02.datasets_494724_1479226_COVID19_line_list_data.csv')
# dataset2 = pd.read_csv('covid_jpn_total.csv')
# dataset3 = pd.read_csv('all-states-history.csv')

'''
Μία γρήγορη ματιά στο dataset με functions των pandas
'''
print(dataset.head())
print(dataset.info())
print(dataset.describe())

'''
1ο Βήμα:
Έλεγχος για απουσιάζουσες (nan) τιμές στις στήλες του dataframe
Πώς επιλέγουμε να τις διαχειριστούμε τις τιμές αυτές;
- Μία επιλογή αφαιρούμε τα δείγματα (τις γραμμές δηλαδή) με nan τιμές. Ωστόσο μπορεί να μην το επιθυμούμε διότι μπορεί 
  να χάσουμε δείγματα που θα μπορούσαν να εκπαιδεύσουν καλύτερα το μοντέλο μας.
- Τυπικά για τις συνεχείς μεταβλητές αντικαθιστούμε τις nan τιμές με το μέσο όρο των υπολοίπων (έχει έτοιμη συνάρτηση το
  pandas και το scikit-learn) και την πιο συχνά εμφανιζόμενη στις κατηγορικές μεταβλητές.
- Αφαιρούμε τη στήλη τελειώς.
- Ελέγχουμε και "άχρηστες" στήλες, δηλαδή αυτές που αν αφαιρεθούν δε θα επηρεάσουν το μοντέλο machine learning που θα αναπτύξουμε
Το επιλύουμε ΠΡΙΝ το διαχωρισμό του dataset σε train και test set
'''

# Εντολή για να σου δείξει πόσες γραμμές είναι nan από μία στήλη
print(dataset['Unnamed: 25'].isnull().all())
dataset = dataset.drop(columns=['Unnamed: 3','Unnamed: 21','Unnamed: 22','Unnamed: 23','Unnamed: 24','Unnamed: 25','Unnamed: 26','source','link'])


'''
Αρχικά, για απλοποίηση του προβλήματος θα αφαιρέσουμε αρκετές από τις στήλες για την εκπαίδευση του μοντέλου μας.
Γενικά αν κρίνουμε μία στήλη οτι δε θα βοηθήσει στην εκπαίδευση του μοντέλου μας την αφαιρούμε (αρχή garbage in garbage out).
Θα πρέπει να επιστρέψουμε αφού εκαπιδεύσουμε τα μοντέλα μας να τις ξαναδούμε και να αποφασίσουμε ποιες θα 
αφαιρέσουμε ή θα συνδυάσουμε.
Για παράδειγμα, η στήλη "case_in_country" περιέχει μάλλον την πληροφορία ποιο κρούσμα είναι στη χώρα του (π.χ. 60ό 
κρούσμα στη Γερμανία), οπότε ενδέχεται να το χρειαστούμε γιατί αν είναι μεγάλος αριθμός η ΜΕΘ σε αυτή τη χώρα να είναι επιβαρυμένη.
Πολλές φορές όμως μπορεί να μην είναι τόσο προφανές το ποια χαρακτηριστικά είναι αχρείαστα στο πρόβλημά μας (πώς βλέπουμε 
αν έχει σχέση ένα χαρακτηριστικό με την ετικέτα; --- Correlation μεταξύ κλάσης που θέλουμε να προβλέψουμε και χαρακτηριστικών)
Στη στήλη "age" έχουμε επίσης nan values, οπότε τα γεμίζουμε με το μέσο όρο των υπολοίπων τιμών
Επίσης στα κατηγορικά χαρακτηριστικά θα γεμίσουμε τα nan values με την κατηγορία που εμφανίζεται τις περισσότερες φορές
Αυτό γίνεται είτε:
- Μέσω των pandas απευθείας με τη μέθοδο που φαίνεται παρακάτω
- Μέσω του function Imputer του scikit learn με τις μεθόδους fit και transform --> Αυτή τη λύση μπορούμε να τη βάλουμε και 
  σε pipeline (θα δούμε παρακάτω τη χρήση του).
  
'''
# , 'from Wuhan'
dataset = dataset.drop(columns=['case_in_country', 'visiting Wuhan','summary','location','country','symptom','symptom_onset','hosp_visit_date','exposure_start','exposure_end','If_onset_approximated','reporting date'])
dataset['gender'] = dataset['gender'].fillna(dataset['gender'].value_counts().index[0])
imp = SimpleImputer()

dataset['age'] = pd.DataFrame(imp.fit_transform(dataset[['age']]))
mask_fem = dataset['gender'] != 'male'
mask_mal = dataset['gender'] == 'male'
dataset.loc[mask_fem, 'gender'] = 0
dataset.loc[mask_mal, 'gender'] = 1

imp = SimpleImputer(strategy='most_frequent')
dataset['from Wuhan'] = pd.DataFrame(imp.fit_transform(dataset[['from Wuhan']]))

'''
Οι ετικέτες των κλάσεων που θέλουμε να προβλέψουμε είναι σε 2 στήλες και όχι σε 1. Οι στήλες αυτές είναι οι "death" και
"recovered", οι οποίες επίσης σε μερικά rows δεν έχουν τιμή 0 ή 1, έχουν κάποια ημερομηνία.
Άρα χρειάζεται να κάνουμε κάποια προεπεξεργασία πάλι στην οποία να τα ενώσουμε σε μία στήλη και να αντικαταστήσουμε τις
τιμές με 0 ή 1.
'''

dataset['class'] = np.nan
for (index, death), (index, recovered) in zip(dataset['death'].items(), dataset['recovered'].items()):
    if death == '1':
        dataset.at[index, 'class'] = 1
    if recovered == '1':
        dataset.at[index, 'class'] = 0
    # Να αποφασίσω τι να κάνω αν και τα 2 είναι 0 --- είτε βάζω καινούρια κλάση, ότι είναι ακόμα νοσοκομείο ή κάνω θεώρηση ότι δεν πέθανε
    if death == '0' and recovered == '0':
        dataset.at[index, 'class'] = 0
    if len(death) > 1:
        dataset.at[index, 'class'] = 1
    if len(recovered) > 1:
        dataset.at[index, 'class'] = 0


'''
Ελέγχουμε Correlation μεταξύ της κλάσης και των χαρακτηριστικών
Παίρνει τιμές στο διάστημα [-1, 1].
- Όσο πιο κοντά είναι στο 1, σημαίνει ότι έχουν ισχυρή θετική συσχέτιση -- όσο αυξάνεται το χαρακτηριστικό, τόσο αυξάνεται και το label
- Όσο πιο κοντά είναι στο -1, σημαίνει ότι έχουν ισχυρή αρνητική συσχέτιση -- όσο μειώνεται το χαρακτηριστικό, τόσο αυξάνεται το label
- Όσο πιο κοντά είναι στο 0, δείχνει ότι δεν υπάρχει γραμμική συσχέτιση
Προσοχή: Αυτή η εντολή δείχνει γραμμικές συσχετίσεις μόνο και όχι μη γραμμικές

Για features με συνεχείς τιμές βάζουμε τη μέθοδο "pearson"
Για features με κατηγορικές τιμές βάζουμε τη μέθοδο "spearman" -- Στο συγκεκριμένο δεν το υπολογίζει για κάποιο λόγο
'''

corr_matrix_cont = dataset.corr(method='pearson') # Η pearson είναι και η default
corr_matrix_cat = dataset.corr(method='spearman')
print(corr_matrix_cont['class'].sort_values(ascending=False))

print(corr_matrix_cat['class'].sort_values(ascending=False))

'''
Αφού φτιάξαμε τις ετικέτες σε μία στήλη, τώρα αφαιρούμε τις στήλες που περισσέυουν για να εκαπιδεύσουμε το μοντέλο, επίσης
ελέγχουμε εάν έχουμε χαρακτηρισιτκά (features) τα οποία είναι διακριτά. Εάν είναι διακριτά κοιτάζουμε εάν είναι διατεταγμένα
ή όχι. Εάν είναι διατεταγμένα, μπορούμε να αντικαταστήσουμε κατευθείαν τις τιμές αυτές με 0, 1 κοκ. Αλλιώς πρέπει να τις
αντικαταστήσουμε κάνοντας ένα mapping και έχοντας έναν πίνακα με ένα ενεργό στοιχείο κάθε φορά. Δηλ. αν έχουμε τις κλάσεις
"red", "green", "blue", τότε τους μετατρέπουμε σε έναν διάνυσμα το καθένα όπου:
red = [1, 0, 0]
green = [0, 1, 0]
blue = [0, 0, 1]
Στο παράδειγμά μας έχουμε το χαρακτηριστικό (feature) "gender"
Καλές πηγές: 
https://www.datacamp.com/community/tutorials/categorical-data
https://towardsdatascience.com/guide-to-encoding-categorical-features-using-scikit-learn-for-machine-learning-5048997a5c79
https://towardsdatascience.com/get-into-shape-14637fe1cd32
https://www.youtube.com/watch?v=WXHLLO4FnZs&ab_channel=DataCamp
'''
dataset = dataset.drop(columns=['death', 'recovered', 'id'])
# enc = OneHotEncoder(handle_unknown='ignore')
# dataset['gender'] = pd.DataFrame(enc.fit_transform(X=dataset[['gender']]))

'''
Ελέγχουμε επίσης όταν έχουμε ένα supervised machine learning πρόβλημα, εάν είναι ισορροπημένο το dataset, δηλαδή
εάν οι κλάσεις μίας τιμής δεν έχει πολύ μεγαλύτερο πλήθος από το πλήθος των κλάσεων της άλλης τιμής.
Εάν είναι 1.5 φορές μεγαλύτερο, τότε έχουμε ανισορροπία στο dataset.
Οι περισσότεροι ταξινομητές βγάζουν καλύτερα αποτελέσματα όταν το dataset Που του δίνουμε να εκπαιδεύσει είναι ισορροπημένο.
Για να εξισορροπήσουμε ένα dataset Υπάρχουν 2 λύσεις:
Undersampling --> Αφαιρούμε δείγματα από την κλάση που εμφανίζεται τις περισσότερες φορές. Έχει το μειονέκτημα ότι μπορεί
να αφαιρέσει δείγματα που περιέχουν σημαντική πληροφορία για το μοντέλο που θέλουμε να εκπαιδεύσουμε. Επίσης αναλόγως το μέγεθος
του dataset, ενδέχεται να μη θέλουμε να χάσουμε δείγματα
Oversampling --> Επαναλαμβάνουμε (τυχαία) δείγματα της κλάσης που εμφανίζεται τις λιγότερες φορές.
'''
print(dataset['class'].value_counts())


'''
Χωρίζουμε το dataset σε features X και labels y
'''
# X = pd.DataFrame(data)
'''
Πρόκειται για Supervised Learning και πιο συγκεκριμένα για Classification problem, δηλαδή θα έχουμε μία στήλη με ετικέτες που
έχουν διακριτές τιμές (δηλαδή 0 και 1) και θέλουμε να φτιάξουμε ένα μοντέλο που να τις προβλέπει.
Σε κάθε μοντέλο που θα αναπτύσσουμε θα ελέγχουμε την απόδοση με διάφορες μετρικές (precision, recall, accuracy και f1-score).
'''

'''
2ο Βήμα:
Αφού τελειώσουμε με την προεπεξεργασία των δεδομένων μας, χωρίζουμε το dataset σε train set και test set. Εμείς 
εκπαιδεύουμε με το train set και ελέγχουμε στο test set αν γενικεύει ικανοποιητικά το μοντέλο (ή αλλιώς ο ταξινομητής) 
που επιλέξαμε. 
Συνάρτηση train_test_split
Θα το χωρίσουμε σε 80-20 (δηλ. train_set 80% του dataset)
'''

X_train, X_test, y_train, y_test = train_test_split(dataset.loc[:, dataset.columns != 'class'], dataset['class'], test_size=0.2, random_state=42)

'''
Κατάρα της διαστατικότητας.
Προσέχουμε πάντα το πλήθος των δειγμάτων  (δηλ. το πλήθος των γραμμών) με το πλήθος των χαρακτηριστικών (δηλ. το 
πλήθος των στηλών). Εάν το πλήθος αυτό είναι "κοντά", τότε πρέπει να μειώσουμε τις διαστάσεις των χαρακτηριστικών
μέσω του PCA
'''

'''
Μετρικές που χρησιμοποιούμε γενικά:
- Σε πρόβλημα Regression, όπου θέλουμε να προβλέψουμε μία συνεχή τιμή, μας αρκεί να μετρήσουμε την ακρίβεια (precision).
- Σε πρόβλημα Classification, η μετρική του precision δε μας αρκεί και έχουν οριστεί επιπλέον οι μετρικές "recall" και "f1_score"
  Σε Classification πρόβλημα το ιδανικό είναι να έχουμε precision, recall και f1_score = 1.0. Παρόλα αυτά, υπάρχει trade-off
  μεταξύ precision και recall, δηλαδή εάν αυξήσουμε το ένα μειώνεται το άλλο. Το ποιο θέλουμε να έχουμε υψηλότερα εξαρτάται από
  το πρόβλημα που θέλουμε να επιλύσουμε.
Διαφορά μεταξύ precision και recall:
Precision πρακτικά σημαίνει ότι αν πάμε να προβλέψουμε μία τιμή, με τι ποσοστό θα το προβλέψουμε σωστά.
Recall πρακτικά σημαίνει ότι από όλα τα δείγματα που είχαν θετική τιμή, πόσα από αυτά βρήκα σωστά.
F1_score προκύπτει από τον συνδυασμό precision και recall (Harmonic Mean)
'''


'''
Όλες τις μετατροπές που μπορούμε να κάνουμε στα χαρακτηριστικά για τη βελτίωση της απόδοσης του μοντέλου που θα φτιάξουμε,
μπορούμε να τις βάλουμε σε ένα pipeline. Το pipeline θα λειτουργήσει ως σωλήνας ο οποίος θα κάνει αυτόματα όλες τις ενέργειες
με τη σειρά που θα του ορίσουμε εμείς, αντί να κάνουμε συνεχώς fit και transform.
Προσοχή!!: Το pipeline και τις αντίστοιχες ενέργειες σχετικά με τα χαρακτηριστικά ενός dataset τα κάνουμε αφού έχουμε
φροντίσει για τη συμπλήρωση/ αφαίρεση δειγμάτων που έχουν απουσιάζουσες τιμές
'''

scaler = StandardScaler()
ros = RandomOverSampler()

'''
Εφαρμόζουμε τον 1ο ταξινομητή Kneighbors
'''

# knn_clf = KNeighborsClassifier(n_neighbors=1)
# knn_clf.fit(X_train, y_train)
# y_pred = knn_clf.predict(X_test)
# print('KNN Classifier')
# print(classification_report(y_test, y_pred))

'''
Για τον Kneighbors
Ψάχνουμε να κάνουμε βελτιστοποίηση των Hyperparameters του μοντέλου μέσω του GridSearch
'''


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

'''
Εφαρμόζουμε 2ο ταξινομητή, το DecisionTree Classifier
'''

# tree_clf = DecisionTreeClassifier()
# tree_clf.fit(X_train, y_train)
# tree_pred = tree_clf.predict(X_test)
# print(f'For Decision Tree classifier\n{classification_report(y_test, tree_pred)}')
#
#
# steps = [('sampler',ros),
#          ('DTree', tree_clf)]
# pipe = Pipeline(steps=steps, memory='tmp')
#
# criterion = ['gini', 'entropy']
# strategy = ['not majority']
# params = {'DTree__criterion':criterion,
#           'sampler__sampling_strategy':strategy}
# estimator = GridSearchCV(pipe, param_grid=params, cv=10, n_jobs=-1)
# estimator.fit(X_train, y_train),
# dtree_best_pred = estimator.predict(X_test)
# print(estimator.best_estimator_)
# print(estimator.best_params_)

'''
Εφαρμόζουμε 3ο ταξινομητή, το LogisticRegression
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
Βελτιστοποίηση υπερπαραμέτρων μέσω GridSearch
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

'''
Εφαρμόζουμε 4ο ταξινομητή Random Forest classifier
'''

# rand_forest_clf = RandomForestClassifier()
# rand_forest_clf.fit(X_train, y_train)
# rand_pred = rand_forest_clf.predict(X_test)
# print(f'For Random Forest\n{classification_report(y_test, rand_pred)}')


'''
Εφαρμόζουμε 5ο ταξινομητή
'''


'''
Βελτιστοποιούμε τον Random Forest classifier
'''

# pipe = Pipeline(steps=[('RF', rand_forest_clf)], memory='tmp')
# n_estimators = [1,2,3,4,5]
# min_samples_split = [9]
# estimator = GridSearchCV(pipe, dict(RF__n_estimators=n_estimators, RF__min_samples_split=min_samples_split), cv=5, scoring='f1_macro')
# estimator.fit(X_train, y_train)
# rf_best_pred = estimator.predict(X_test)
# print(f'Optimized Random Forest\n{classification_report(y_test, rf_best_pred)}')
# print(estimator.best_params_)

'''
Εφαρμόζουμε 6ο ταξινομητή -- SVM -- Support Vector Machine
'''

# svm_clf = SVC()
# svm_clf.fit(X_train, y_train)
# svm_pred = svm_clf.predict(X_test)
# print('For SVM')
# print(classification_report(y_test, svm_pred))
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


