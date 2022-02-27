import argparse
from azureml.core import Run, Model
from azureml.core.experiment import Experiment
import joblib
import numpy as np
import os
import pandas as pd
from os import listdir
from os.path import isfile, join


parser = argparse.ArgumentParser()

parser.add_argument("--model_name")
parser.add_argument("--outputfolder")

args = parser.parse_args()

run = Run.get_context()
parent_run_id = run.parent.id
ws = run.experiment.workspace
exp_name = run.experiment.name
exp = Experiment(workspace=ws, name=exp_name)

# Retrieve current run using parent run id
current_run = Run(experiment=exp, run_id=parent_run_id)

# Get date of Run
date = pd.to_datetime(current_run.get_details()['startTimeUtc']).date().strftime('%Y-%m-%d')

# Retrieve Model
model_path = Model.get_model_path(model_name=args.model_name, _workspace=ws)
model = joblib.load(model_path)

# Retrieve Inference DTM
inference_path = os.environ["InferenceDTM"]
files = [f for f in listdir(inference_path) if isfile(join(inference_path, f))]
for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(inference_path, file)
        inference_dtm = pd.read_csv(file_path)
    elif file.endswith('.parquet'):
        file_path = os.path.join(inference_path, file)
        inference_dtm = pd.read_parquet(file_path)
    else:
        pass

# Keep Transaction IDs for merging with predictions in the end
ids = inference_dtm['Transaction_ID'].values

# Drop them for prediction in the end
inference_dtm = inference_dtm.drop(['Transaction_ID'], axis=1)

X = model.predict(inference_dtm.values)
X_proba = model.predict_proba(inference_dtm.values)
probs = np.max(X_proba, axis=1)

# Write predictions.csv to Blob storage
predictions = pd.DataFrame({'Transaction ID': list(ids),
                            'Probability': list(probs),
                            'Category L4': list(X)}
                           )

predictions_path = os.path.join(args.outputfolder, 'Predictions.csv')
predictions.to_csv(predictions_path, index=False)

predictions_path_hist = os.path.join(args.outputfolder, f'Predictions_{parent_run_id}_{date}.csv')
predictions.to_csv(predictions_path_hist, index=False)


run.complete()