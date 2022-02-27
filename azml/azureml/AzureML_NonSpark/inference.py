import argparse
from azureml.core import Run, Model
import joblib
import nltk
import pandas as pd
import os
nltk.download('punkt')
nltk.download('stopwords')

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, dest="model_name")
parser.add_argument("--datafolder", type=str, dest="datafolder")
parser.add_argument("--features", type=str, dest="features")
parser.add_argument("--input_filename", type=str, dest="input_filename")
args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace

# Load model and features pkl files
model_path = Model.get_model_path(model_name=args.model_name, _workspace=ws)
model = joblib.load(model_path)
features_path = Model.get_model_path(model_name=args.features, _workspace=ws)
features = joblib.load(features_path)

test_path = os.path.join(args.datafolder, args.input_filename)
test = pd.read_csv(test_path)

train_cols = features
test_cols = test.columns

not_in_test = list(set(train_cols) - set(test_cols))
extra_in_test = list(set(test_cols) - set(train_cols))

print("Not in test: ", len(not_in_test))
print("Extra in test: ", len(extra_in_test))

# Dropping columns not present in test DTM
df_test = test.drop(columns=extra_in_test, axis=1)

# Adding columns with zeros present in train and not in test
if len(not_in_test) != 0:
    df_temp = pd.DataFrame(0, index=range(len(df_test)), columns=not_in_test)
    final_df = pd.concat([df_test, df_temp], axis=1)

# Putting the correct order of features according to trained features
final_df = final_df[train_cols]

X = model.predict(final_df.values)
predictions = pd.DataFrame(list(X), columns=['Category L4'])

os.makedirs('outputs', exist_ok=True)
predictions_path = os.path.join('outputs', 'predictions.csv')
predictions.to_csv(predictions_path, index=False)

run.complete()
