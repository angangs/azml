from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace, Datastore, Dataset, Environment
from azureml.core.compute import ComputeInstance, AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration
from typing import Tuple

VALID_COMPUTE_TYPES = {'instance', 'cluster'}
VALID_DATASET_TYPES = {'Tabular', 'File'}
LENGTHS = [2, 16]


def register_datastore(account_key: str, blob_datastore_name: str, CONTAINER_NAME: str,
                       STORAGE_ACCOUNT_NAME: str, ws: Workspace) -> None:
    """
    Register Datastore to Workspace.
    :param account_key: Primary key. Retrieve from Storage Account -> Access Keys -> key1 (Key)
    :param blob_datastore_name: Datastore name
    :param CONTAINER_NAME: Blob container name in Storage account
    :param STORAGE_ACCOUNT_NAME: Storage account name
    :param ws: Workspace object
    :return: None
    """

    Datastore.register_azure_blob_container(workspace=ws,
                                            datastore_name=blob_datastore_name,
                                            container_name=CONTAINER_NAME,
                                            account_name=STORAGE_ACCOUNT_NAME,
                                            account_key=account_key
                                            )


def register_dataset(blob_datastore_name: str, BLOBNAME_train: str, BLOBNAME_test: str, dataset_type: str,
                     encoding: str, test_dataset_name: str, train_dataset_name: str, ws: Workspace) -> None:
    """
    Register Tabular/File Datasets from Datastore to Workspace's Datasets.

    :param blob_datastore_name: Datastore name
    :param BLOBNAME_train: Train dataset name (uploaded csv)
    :param BLOBNAME_test: Test dataset name (uploaded csv)
    :param dataset_type: Dataset type, either 'Tabular' or 'File'
    :param encoding: encoding of TabularDataset object (eg: utf-8, iso88591 etc.)
    :param test_dataset_name: Test Dataset name to register as Dataset
    :param train_dataset_name: Train Dataset name to register as Dataset
    :param ws: Workspace object
    :return: None
    """

    if dataset_type not in VALID_DATASET_TYPES:
        raise ValueError("Dataset type must be one of %r." % VALID_DATASET_TYPES)
    else:
        datastore = Datastore.get(ws, blob_datastore_name)

        # Get datastore path for train
        train_datastore_path = [(datastore, BLOBNAME_train)]

        # Get datastore path for test
        test_datastore_path = [(datastore, BLOBNAME_test)]

        if dataset_type == 'Tabular':
            train_ds = Dataset.Tabular.from_delimited_files(path=train_datastore_path, encoding=encoding)
            test_ds = Dataset.Tabular.from_delimited_files(path=test_datastore_path, encoding=encoding)
        else:
            train_ds = Dataset.File.from_files(path=train_datastore_path)
            test_ds = Dataset.File.from_files(path=test_datastore_path)

        train_ds.register(workspace=ws,
                          name=train_dataset_name,
                          create_new_version=True)

        test_ds.register(workspace=ws,
                         name=test_dataset_name,
                         create_new_version=True)

    print(f'{train_dataset_name} registered.')
    print(f'{test_dataset_name} registered.')


def setup_workspace(local_path: str, LOCATION: str, RESOURCE_GROUP_NAME: str, STORAGE_ACCOUNT_NAME: str,
                    subscription_id: str, ws_name: str, create_group: bool = False,
                    create_ws: bool = False, set_force: bool = False, use_my_storage: bool = False) -> Workspace:
    """
    Workspace Creation/Setup.
    :param local_path: The local path to store the config_file if create_ws is True
    :param LOCATION: the location of the Workspace
    :param RESOURCE_GROUP_NAME: the name of the Resource Group
    :param STORAGE_ACCOUNT_NAME: the name of the Storage Account
    :param subscription_id: Required subscription id
    :param ws_name: the name of the Workspace
    :param create_group: If set to False (default), do not create Resource Group, else create it.
    :param create_ws: If set to False (default), do not create Workspace, else create it.
    :param set_force: If set to True, a sign-in prompt to Azure will popup , else no (default)
    :param use_my_storage: Whether to use already created storage account or create a new default one.
    :return: ws -> Workspace object
    """

    # Create InteractiveLoginAuthentication object with set_force parameter
    interactive_auth = InteractiveLoginAuthentication(force=set_force)

    # When creating a workspace, if you don't provide a specific storage account,
    # a default new one will be created along with an App Insights and a Key vault.
    if use_my_storage:
        storage_account = f"subscriptions/{subscription_id}/resourcegroups/{RESOURCE_GROUP_NAME}/providers/microsoft.storage/storageaccounts/{STORAGE_ACCOUNT_NAME}"
    else:
        storage_account = None

    # If there is already a ML workspace created, just retrieve the Workspace object.
    if create_ws:
        ws = Workspace.create(name=ws_name,
                              subscription_id=subscription_id,
                              resource_group=RESOURCE_GROUP_NAME,
                              create_resource_group=create_group,
                              location=LOCATION,
                              storage_account=storage_account,
                              auth=interactive_auth
                              )

        # Write to JSON
        ws.write_config(path=local_path, file_name="ws_config.json")
    else:
        ws = Workspace.get(name=ws_name,
                           subscription_id=subscription_id,
                           resource_group=RESOURCE_GROUP_NAME,
                           location=LOCATION,
                           auth=interactive_auth,
                           cloud='AzureCloud')

    return ws


def retrieve_dataset(test_dataset_name: str, train_dataset_name: str, ws: Workspace) -> Tuple[Dataset, Dataset]:
    """
    Retrieve already registered Train and Test Datasets.
    :param test_dataset_name: Test Dataset
    :param train_dataset_name: Train Dataset
    :param ws: Workspace object
    :return: Train and Test Dataset objects
    """

    train_dataset = Dataset.get_by_name(ws, train_dataset_name)
    test_dataset = Dataset.get_by_name(ws, test_dataset_name)

    return train_dataset, test_dataset


def prepare_pipeline(compute_name: str, compute_type: str, env_name: str, vm_size: str, ws: Workspace,
                     conda_packages: list = None, min_nodes: int = 1, max_nodes: int = 1,
                     pip_packages: list = None, vm_priority: str = 'lowpriority') -> Tuple[ComputeTarget, RunConfiguration, Environment]:
    """
    Setup Compute Target, Environment and RunConfiguration for Pipeline run.
    :param compute_name: ComputeTarget name (must be between 2 and 16 characters/digits)
    :param compute_type: Type of compute target, either Compute Instance ('instance') or Compute Cluster ('cluster')
    :param env_name: Name of the Environment object
    :param vm_size: Size of virtual machine
    :param ws: Workspace object
    :param conda_packages: List of conda packages to be used in the Environment. Default None.
    :param min_nodes: Number of minimum nodes, default 1
    :param max_nodes: Number of maximum nodes, default 1
    :param pip_packages: List of pip packages to be used in the Environment. Default None.
    :param vm_priority: VM Priority. Either 'lowpriority' (default) or 'dedicated'
    :return: ComputeTarget and RunConfiguration objects
    """

    if len(compute_name) > LENGTHS[1] or len(compute_name) < LENGTHS[0]:
        raise ValueError('Length must be between 2 and 16 characters/digits.')
    if compute_type not in VALID_COMPUTE_TYPES:
        raise ValueError("Compute type must be one of %r." % VALID_COMPUTE_TYPES)
    else:
        if compute_type == 'instance':
            # Create compute target or select if already created
            try:
                compute_target = ComputeInstance(workspace=ws, name=compute_name)
                print(f'Using existing compute instance: {compute_name}')
                if compute_target.get_status().state == 'Running':
                    print(f'{compute_name} is already running.')
                else:
                    compute_target.start()
                    print(f'{compute_name} started running.')
            except ComputeTargetException:
                compute_config = ComputeInstance.provisioning_configuration(
                    vm_size=vm_size,
                    ssh_public_access=False
                )
                compute_target = ComputeInstance.create(ws, compute_name, compute_config)
                compute_target.wait_for_completion(show_output=True)
                print(f'{compute_name} provisioned.')
        else:
            # Create compute target or select if already created
            try:
                compute_target = ComputeTarget(workspace=ws, name=compute_name)
                print(f'Using existing compute cluster: {compute_name}')
            except ComputeTargetException:
                compute_config = AmlCompute.provisioning_configuration(
                    vm_size=vm_size,
                    vm_priority=vm_priority,
                    min_nodes=min_nodes,
                    max_nodes=max_nodes
                )
                compute_target = ComputeTarget.create(ws, compute_name, compute_config)
                compute_target.wait_for_completion(show_output=True)
                print(f'{compute_name} provisioned.')

    # Create Environment
    env = Environment(name=env_name)

    # Create the dependencies object
    env_dep = CondaDependencies.create(conda_packages=conda_packages,
                                       pip_packages=pip_packages)

    env.python.conda_dependencies = env_dep

    # Register the environment
    env.register(ws)

    # Define RunConfig for the compute target
    run_config = RunConfiguration()
    run_config.target = compute_target
    run_config.environment = env

    print("Run configuration created.")

    return compute_target, run_config, env
