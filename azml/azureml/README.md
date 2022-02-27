# AzureML



## Getting started


- First, create a virtual environment either with conda or virtualenv. Tests have been carried out with Python 3.8.12.
- Then, pip install azureml-sdk==1.35.0 to install all required packages for AzureML notebooks.
- Read through utils.py to understand the functionality of the functions, especially setup_workspace and prepare_pipeline.

## Synapse - Spark functionality

In order to use SynapseSparkStep in an AzureML pipeline, you need to have a Synapse workspace. Then, create a LinkedService in AzureML workspace
with the synapse workspace. You have to be an "Owner" of Synapse workspace resource to perform linking. You can check your role in the Azure resource management portal, if you don't have an "Owner" role, you can contact an "Owner" to link the workspaces for you. 
After that, create an Apache Spark Pool in Synapse workspace and attach it in the AzureML workspace under 'Compute' -> 'Attach Compute'
Finally, make sure you grant "Synapse Apache Spark Administrator" role of the synapse workspace to the generated workspace linking MSI in Synapse studio portal before you submit a pipeline run with a SynapseSparkStep.
