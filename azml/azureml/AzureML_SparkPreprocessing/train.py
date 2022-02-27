from azureml.core import Run
import joblib
import os
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

algorithms = {0: 'RF',
              1: 'SVM',
              2: 'NB',
              3: 'XGBoost'}

run = Run.get_context()
parent_run_id = run.parent.id

path = os.environ["TrainDTM"]
files = [f for f in listdir(path) if isfile(join(path, f))]
for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(path, file)
        df_L4 = pd.read_csv(file_path)
    elif file.endswith('.parquet'):
        file_path = os.path.join(path, file)
        df_l4 = pd.read_parquet(file_path)
    else:
        pass

# Splitting Data to train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(df_L4.iloc[:, :-1], df_L4.iloc[:, -1], test_size=0.1,
                                                    random_state=42)

print("RF Fitting for L4 started")

# According to R script
# Random Forest Classifier
rf_l4 = RandomForestClassifier(n_estimators=20)
rf_l4 = rf_l4.fit(X_train, Y_train)

print("SVM Fitting for L4 started")

# Support Vector Machine
svm_l4 = SVC(kernel='linear', probability=True)
svm_l4 = svm_l4.fit(X_train, Y_train)

print("NB Fitting for L4 started")

# Naive Bayes
nb_l4 = MultinomialNB(alpha=0)
nb_l4 = nb_l4.fit(X_train, Y_train)

print("XGBoost Fitting for L4 started")

# XGBoost
xgb_l4 = XGBClassifier()
xgb_l4 = xgb_l4.fit(X_train, Y_train)


# Testing the Model
## Random Forest
rf_pred = rf_l4.predict(X_test)
rf_acc = accuracy_score(Y_test, rf_pred)
print("RF Accuracy: %.2f %%" % round(rf_acc*100, 2))

## Support Vector Machine
svm_pred = svm_l4.predict(X_test)
svm_acc = accuracy_score(Y_test, svm_pred)
print("SVM Accuracy: %.2f %%" % round(svm_acc*100, 2))

## Naive Bayes
nb_pred = nb_l4.predict(X_test)
nb_acc = accuracy_score(Y_test, nb_pred)
print("NB Accuracy: %.2f %%" % round(nb_acc*100, 2))

## XGBoost
xgb_pred = xgb_l4.predict(X_test)
xgb_acc = accuracy_score(Y_test, xgb_pred)
print("XGB Accuracy: %.2f %%" % round(xgb_acc*100, 2))

# Log Accuracy Scores
run.log("RF Accuracy", str(round(rf_acc*100, 2)) + '%')
run.log("SVM Accuracy", str(round(svm_acc*100, 2)) + '%')
run.log("NB Accuracy", str(round(nb_acc*100, 2)) + '%')
run.log("XGB Accuracy", str(round(xgb_acc*100, 2)) + '%')

# Find best model
clfs = [rf_l4, svm_l4, nb_l4, xgb_l4]
scores = [rf_acc, svm_acc, nb_acc, xgb_acc]
max_score = max(scores)
max_index = scores.index(max_score)
best_clf = clfs[max_index]
best_model = algorithms[max_index]

print('Best model is %s.' % best_model)

# Register model to workspace
os.makedirs("outputs", exist_ok=True)
path = os.path.join('outputs', f'{best_model}_L4.pkl')
joblib.dump(value=best_clf, filename=path)
run.upload_file(f"outputs/{best_model}_L4.pkl", f"outputs/{best_model}_L4.pkl")
model = run.register_model(model_path=f'outputs/{best_model}_L4.pkl',
                           model_name='Model',
                           tags={'RunID': parent_run_id,
                                 'Best Model': best_model,
                                 'Accuracy': str(round(max_score*100, 2)) + '%'}
                           )

print(model.name, model.id, model.version, sep='\t')

run.complete()




