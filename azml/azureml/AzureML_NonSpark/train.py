from argparse import ArgumentParser
from azureml.core import Run
import joblib
import nltk
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
nltk.download('punkt')
nltk.download('stopwords')


parser = ArgumentParser()
parser.add_argument('--datafolder', type=str, dest='datafolder')
parser.add_argument('--logsdatafolder', type=str, dest='logsdatafolder')
parser.add_argument("--outputdatafolder", type=str, dest='outputdatafolder')
parser.add_argument("--input_filename", type=str, dest="input_filename")
args = parser.parse_args()

run = Run.get_context()


path = os.path.join(args.datafolder, args.input_filename)
df_L4 = pd.read_csv(path)

print("RF Fitting for L4 started")

# Splitting Data to train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(df_L4.iloc[:, :-1], df_L4.iloc[:, -1], test_size=0.1,
                                                    random_state=42)

# According to R script's number of trees
rf_l4 = RandomForestClassifier(n_estimators=20)
rf_l4 = rf_l4.fit(X_train, Y_train)

# Testing the Model
predictions = rf_l4.predict(X_test)
acc = round(accuracy_score(Y_test, predictions)*100, 2)
print("Accuracy: %.2f %%" % acc)

# Log Accuracy Score
run.log("Accuracy", str(acc) + '%')

# Register model to workspace
os.makedirs("outputs", exist_ok=True)
path = os.path.join('outputs', 'rf_l4.pkl')
joblib.dump(value=rf_l4, filename=path)
run.upload_file("outputs/rf_l4.pkl", "outputs/rf_l4.pkl")
model = run.register_model(model_path='outputs/rf_l4.pkl',
                           model_name='RF_L4',
                           tags={'source': 'SDK Run', 'algorithm': 'RandomForest'})

print(model.name, model.id, model.version, sep='\t')

run.complete()
