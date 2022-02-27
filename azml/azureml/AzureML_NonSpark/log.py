import argparse
from azureml.core import Run
from azureml.core.experiment import Experiment
import os
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument("--process", type=str, dest="process")
parser.add_argument("--outputfolder", type=str, dest="outputfolder")
args = parser.parse_args()

if args.process == "Training":
    STEPS = {1: 'Data Preprocessing',
             2: 'Training'}
else:
    STEPS = {1: 'Data Preprocessing',
             2: 'Predict'}


run = Run.get_context()
parent_run_id = run.parent.id
ws = run.experiment.workspace
exp_name = run.experiment.name
exp = Experiment(workspace=ws, name=exp_name)

# Retrieve current run using parent run id
current_run = Run(experiment=exp, run_id=parent_run_id)

# List of children runs in reversed order
# We iterate from the last as the StepRuns are recorded chronologically with the latest in the first place
children_runs = list(reversed(list(current_run.get_children())))
num_children = len(list(current_run.get_children()))

# Create empty DataFrame
columns = ['Parent Run ID', 'Run Name', 'Run ID', 'Step ID', 'Step Name', 'Start Time (UTC)', 'End Time (UTC)', 'Error', 'Status']
df = pd.DataFrame(columns=columns)

# Create logs.csv
counter = 1
if num_children == 3:
    for child_run in children_runs[:2]:
        run_details = child_run.get_details()
        if 'error' not in run_details.keys():
            values = [child_run.parent.id, current_run.display_name, run_details['runId'], counter, STEPS[counter],
                      run_details['startTimeUtc'], run_details['endTimeUtc'], 'No error', run_details['status']]
            temp_df = pd.DataFrame([values], columns=columns)
            df = df.append(temp_df, ignore_index=True)
        else:
            error = run_details['error']['error']['message']
            values = [child_run.parent.id, current_run.display_name, run_details['runId'], counter, STEPS[counter],
                      run_details['startTimeUtc'], run_details['endTimeUtc'], error, run_details['status']]
            temp_df = pd.DataFrame([values], columns=columns)
            df = df.append(temp_df, ignore_index=True)
        counter += 1
else:
    for child_run in children_runs[:1]:
        run_details = child_run.get_details()
        if 'error' not in run_details.keys():
            values = [child_run.parent.id, current_run.display_name, run_details['runId'], counter, STEPS[counter],
                      run_details['startTimeUtc'], run_details['endTimeUtc'], 'No error', run_details['status']]
            temp_df = pd.DataFrame([values], columns=columns)
            df = df.append(temp_df, ignore_index=True)
        else:
            error = run_details['error']['error']['message']
            values = [child_run.parent.id, current_run.display_name, run_details['runId'], counter, STEPS[counter],
                      run_details['startTimeUtc'], run_details['endTimeUtc'], error, run_details['status']]
            temp_df = pd.DataFrame([values], columns=columns)
            df = df.append(temp_df, ignore_index=True)
        counter += 1


# Write logs.csv to outputs or write to db when time comes
os.makedirs(args.outputfolder, exist_ok=True)
logs_path = os.path.join(args.outputfolder, f'{args.process}Logs.csv')
df.to_csv(logs_path, index=False)
