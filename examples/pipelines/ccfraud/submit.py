"""Federated Learning Cross-Silo basic pipeline for Credit Card Fraud example.

This script:
1) reads a config file in yaml specifying the number of silos and their parameters,
2) reads the components from a given folder,
3) builds a flexible pipeline depending on the config,
4) configures each step of this pipeline to read/write from the right silo.
"""
import os
import argparse
import random
import string
import datetime
import webbrowser
import time
import sys

# Azure ML sdk v2 imports
import azure
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

# to handle yaml config easily
from omegaconf import OmegaConf

#to read information from json config file
import json
from azure.ai.ml.exceptions import ErrorCategory, ErrorTarget, ValidationException

############################
### CONFIGURE THE SCRIPT ###
############################

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "--config",
    type=str,
    required=False,
    default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    help="path to a config yaml file",
)
parser.add_argument(
    "--offline",
    default=False,
    action="store_true",
    help="Sets flag to not submit the experiment to AzureML",
)

parser.add_argument(
    "--subscription_id",
    type=str,
    required=False,
    help="Subscription ID",
)
parser.add_argument(
    "--resource_group",
    type=str,
    required=False,
    help="Resource group name",
)

parser.add_argument(
    "--workspace_name",
    type=str,
    required=False,
    help="Workspace name",
)

parser.add_argument(
    "--wait",
    default=False,
    action="store_true",
    help="Wait for the pipeline to complete",
)

parser.add_argument(
    "--register_components",
    default=False,
    action="store_true",
    help="Sets flag to register the pipeline components in the workspace.",
)

args = parser.parse_args()

# load the config from a local yaml file
YAML_CONFIG = OmegaConf.load(args.config)

# path to the components
COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", "components", "CCFRAUD"
)

# path to the shared components
SHARED_COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", "components", "utils"
)

# path to the config file
CONFIG_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", ".."
)

###########################
### CONNECT TO AZURE ML ###
###########################

def get_tenant_id(config_path: str="config.json") -> str:
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Checking the keys in the config.json file to check for required parameters.
    if not all([k in config.keys() for k in ("subscription_id", "resource_group", "workspace_name", "tenant_id")]):
        msg = (
            "The config file found in: {} does not seem to contain the required "
            "parameters. Please make sure it contains your subscription_id, "
            "resource_group, workspace_name, and tenant_id."
        )
        raise ValidationException(
            message=msg.format(config_path),
            no_personal_data_message=msg.format("[config_path]"),
            target=ErrorTarget.GENERAL,
            error_category=ErrorCategory.USER_ERROR,
        )
    # get information from config.json
    tenant_id_from_config = config["tenant_id"]
    return tenant_id_from_config

def connect_to_aml():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        tenant_id = get_tenant_id(
            config_path=os.path.join(CONFIG_FOLDER, "config.json")
            )
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential does not work
        credential = InteractiveBrowserCredential(tenant_id=tenant_id)

    # Get a handle to workspace
    try:
        # tries to connect using cli args if provided else using config.yaml
        ML_CLIENT = MLClient(
            subscription_id=args.subscription_id or YAML_CONFIG.aml.subscription_id,
            resource_group_name=args.resource_group
            or YAML_CONFIG.aml.resource_group_name,
            workspace_name=args.workspace_name or YAML_CONFIG.aml.workspace_name,
            credential=credential,
        )

    except Exception as ex:
        print("Could not find either cli args or config.yaml.")
        # tries to connect using local config.json
        ML_CLIENT = MLClient.from_config(credential=credential)

    return ML_CLIENT


#############################################
### GET ML_CLIENT AND COMPUTE INFORMATION ###
#############################################

if not args.offline:
    ML_CLIENT = connect_to_aml()
    COMPUTE_SIZES = ML_CLIENT.compute.list_sizes()


def get_gpus_count(compute_name):
    if not args.offline:
        ws_compute = ML_CLIENT.compute.get(compute_name)
        if hasattr(ws_compute, "size"):
            silo_compute_size_name = ws_compute.size
            silo_compute_info = next(
                (
                    x
                    for x in COMPUTE_SIZES
                    if x.name.lower() == silo_compute_size_name.lower()
                ),
                None,
            )
            if silo_compute_info is not None and silo_compute_info.gpus >= 1:
                return silo_compute_info.gpus
    return 1


####################################
### LOAD THE PIPELINE COMPONENTS ###
####################################

# Loading the component from their yaml specifications
preprocessing_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "preprocessing", "spec.yaml")
)

training_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "traininsilo", "spec.yaml")
)

aggregate_component = load_component(
    source=os.path.join(SHARED_COMPONENTS_FOLDER, "aggregatemodelweights", "spec.yaml")
)

register_model_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "registermodel", "spec.yaml")
)

if (
    hasattr(YAML_CONFIG.training_parameters, "run_data_analysis")
    and YAML_CONFIG.training_parameters.run_data_analysis
):
    data_analysis_component = load_component(
        source=os.path.join(SHARED_COMPONENTS_FOLDER, "data_analysis", "spec.yaml")
    )


########################
### BUILD A PIPELINE ###
########################


def custom_fl_data_path(
    datastore_name, output_name, unique_id="${{name}}", iteration_num=None
):
    """Produces a path to store the data during FL training.

    Args:
        datastore_name (str): name of the Azure ML datastore
        output_name (str): a name unique to this output
        unique_id (str): a unique id for the run (default: inject run id with ${{name}})
        iteration_num (str): an iteration number if relevant

    Returns:
        data_path (str): direct url to the data path to store the data
    """
    data_path = f"azureml://datastores/{datastore_name}/paths/federated_learning/{output_name}/{unique_id}/"
    if iteration_num:
        data_path += f"iteration_{iteration_num}/"

    return data_path


def getUniqueIdentifier(length=8):
    """Generates a random string and concatenates it with today's date

    Args:
        length (int): length of the random string (default: 8)

    """
    str = string.ascii_lowercase
    date = datetime.date.today().strftime("%Y_%m_%d_")
    return date + "".join(random.choice(str) for i in range(length))


pipeline_identifier = getUniqueIdentifier()


@pipeline(
    description=f'FL cross-silo basic pipeline and the unique identifier is "{pipeline_identifier}" that can help you to track files in the storage account.',
)
def fl_ccfraud_basic():

    #####################
    ### DATA-ANALYSIS ###
    #####################

    if (
        hasattr(YAML_CONFIG.training_parameters, "run_data_analysis")
        and YAML_CONFIG.training_parameters.run_data_analysis
    ):
        for silo_index, silo_config in enumerate(YAML_CONFIG.federated_learning.silos):
            # run the pre-processing component once
            silo_pre_processing_step = data_analysis_component(
                training_data=Input(
                    type=silo_config.training_data.type,
                    mode=silo_config.training_data.mode,
                    path=silo_config.training_data.path + "/train.csv",
                ),
                testing_data=Input(
                    type=silo_config.testing_data.type,
                    mode=silo_config.testing_data.mode,
                    path=silo_config.testing_data.path + f"/test.csv",
                ),
                metrics_prefix=silo_config.computes[0],
                silo_index=silo_index,
                **YAML_CONFIG.data_analysis_parameters,
            )

            # add a readable name to the step
            silo_pre_processing_step.name = f"silo_{silo_index}_data_analysis"

            # make sure the compute corresponds to the silo
            silo_pre_processing_step.compute = silo_config.computes[0]

    ######################
    ### PRE-PROCESSING ###
    ######################

    # once per silo, we're running a pre-processing step

    silo_preprocessed_train_data = []  # list of preprocessed train datasets for each silo
    silo_preprocessed_test_data = []  # list of preprocessed test datasets for each silo

    for silo_index, silo_config in enumerate(YAML_CONFIG.federated_learning.silos):
        # run the pre-processing component once for train data
        silo_pre_processing_step_train = preprocessing_component(
            raw_data=Input(
                type=silo_config.training_data.type,
                mode=silo_config.training_data.mode,
                path=silo_config.training_data.path,
            ),
            metrics_prefix=silo_config.name,
        )

        # add a readable name to the step
        silo_pre_processing_step_train.name = f"silo_{silo_index}_preprocessing_train"

        # make sure the compute corresponds to the silo
        silo_pre_processing_step_train.compute = silo_config.computes[0]

        # make sure the data is written in the right datastore
        silo_pre_processing_step_train.outputs.processed_data = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(silo_config.datastore, "train_data"),
        )
        
        if silo_config.name == YAML_CONFIG.federated_learning.orchestrator.data_transformers.selected_silo:
            # save the transformers in the orchestrator
            silo_pre_processing_step_train.outputs.data_transformers_out = Output(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.MOUNT,
                path=YAML_CONFIG.federated_learning.orchestrator.data_transformers.path,
            )
        else:
            # save the transformers in the respective silo
            silo_pre_processing_step_train.outputs.data_transformers_out = Output(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.MOUNT,
                path=silo_config.data_transformers.path,
            )

        # run the pre-processing component once for test data
        silo_pre_processing_step_test = preprocessing_component(
            raw_data=Input(
                type=silo_config.testing_data.type,
                mode=silo_config.testing_data.mode,
                path=silo_config.testing_data.path,
            ),
            metrics_prefix=silo_config.name,
            data_transformers=silo_pre_processing_step_train.outputs.data_transformers_out
        )

        # add a readable name to the step
        silo_pre_processing_step_test.name = f"silo_{silo_index}_preprocessing_test"

        # make sure the compute corresponds to the silo
        silo_pre_processing_step_test.compute = silo_config.computes[0]

        # make sure the data is written in the right datastore
        silo_pre_processing_step_test.outputs.processed_data = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(silo_config.datastore, "test_data"),
        )
        # make sure the data is written in the right datastore
        silo_pre_processing_step_test.outputs.data_transformers_out = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=silo_config.data_transformers.path,
        )

        # if confidentiality is enabled, add the keyvault and key name as environment variables
        if hasattr(YAML_CONFIG, "confidentiality"):
            silo_pre_processing_step_train.environment_variables = {
                "CONFIDENTIALITY_DISABLE": str(not YAML_CONFIG.confidentiality.enable),
                "CONFIDENTIALITY_KEYVAULT": YAML_CONFIG.confidentiality.keyvault,
                "CONFIDENTIALITY_KEY_NAME": YAML_CONFIG.confidentiality.key_name,
            }
            silo_pre_processing_step_test.environment_variables = {
                "CONFIDENTIALITY_DISABLE": str(not YAML_CONFIG.confidentiality.enable),
                "CONFIDENTIALITY_KEYVAULT": YAML_CONFIG.confidentiality.keyvault,
                "CONFIDENTIALITY_KEY_NAME": YAML_CONFIG.confidentiality.key_name,
            }
        else:
            silo_pre_processing_step_train.environment_variables = {
                "CONFIDENTIALITY_DISABLE": "True",
            }
            silo_pre_processing_step_test.environment_variables = {
                "CONFIDENTIALITY_DISABLE": "True",
            }
            
        # assign instance type for AKS, if available
        if hasattr(silo_config, "instance_type"):
            silo_pre_processing_step_train.resources = {
                "instance_type": silo_config.instance_type
            }
            silo_pre_processing_step_test.resources = {
                "instance_type": silo_config.instance_type
            }

        # store a handle to the train data for this silo
        silo_preprocessed_train_data.append(
            silo_pre_processing_step_train.outputs.processed_data
        )
        # store a handle to the test data for this silo
        silo_preprocessed_test_data.append(
            silo_pre_processing_step_test.outputs.processed_data
        )

    ################
    ### TRAINING ###
    ################

    running_checkpoint = None  # for iteration 1, we have no pre-existing checkpoint

    # now for each iteration, run training
    for iteration in range(1, YAML_CONFIG.training_parameters.num_of_iterations + 1):
        # collect all outputs in a dict to be used for aggregation
        silo_weights_outputs = {}

        # for each silo, run a distinct training with its own inputs and outputs
        for silo_index, silo_config in enumerate(YAML_CONFIG.federated_learning.silos):
            # Determine number of processes to deploy on a given compute cluster node
            silo_processes = get_gpus_count(silo_config.computes[0])

            # We need to reload component because otherwise all the instances will share same
            # value for process_count_per_instance
            training_component = load_component(
                source=os.path.join(COMPONENTS_FOLDER, "traininsilo", "spec.yaml")
            )

            # we're using training component here
            silo_training_step = training_component(
                # with the train_data from the pre_processing step
                train_data=silo_preprocessed_train_data[silo_index],
                # with the test_data from the pre_processing step
                test_data=silo_preprocessed_test_data[silo_index],
                # and the checkpoint from previous iteration (or None if iteration == 1)
                checkpoint=running_checkpoint,
                # Learning rate for local training
                lr=YAML_CONFIG.training_parameters.lr,
                # Number of epochs
                epochs=YAML_CONFIG.training_parameters.epochs,
                # Dataloader batch size
                batch_size=YAML_CONFIG.training_parameters.batch_size,
                # Differential Privacy
                dp=YAML_CONFIG.training_parameters.dp,
                # DP target epsilon
                dp_target_epsilon=YAML_CONFIG.training_parameters.dp_target_epsilon,
                # DP target delta
                dp_target_delta=YAML_CONFIG.training_parameters.dp_target_delta,
                # DP max gradient norm
                dp_max_grad_norm=YAML_CONFIG.training_parameters.dp_max_grad_norm,
                # Total num of iterations
                total_num_of_iterations=YAML_CONFIG.training_parameters.num_of_iterations,
                # Silo name/identifier
                metrics_prefix=silo_config.name,
                # Iteration name
                iteration_name=f"Iteration-{iteration}",
                # Model name
                model_name=YAML_CONFIG.training_parameters.model_name,
            )
            # if confidentiality is enabled, add the keyvault and key name as environment variables
            if hasattr(YAML_CONFIG, "confidentiality"):
                silo_training_step.environment_variables = {
                    "CONFIDENTIALITY_DISABLE": str(
                        not YAML_CONFIG.confidentiality.enable
                    ),
                    "CONFIDENTIALITY_KEYVAULT": YAML_CONFIG.confidentiality.keyvault,
                    "CONFIDENTIALITY_KEY_NAME": YAML_CONFIG.confidentiality.key_name,
                }

            # add a readable name to the step
            silo_training_step.name = f"silo_{silo_index}_training"

            # make sure the compute corresponds to the silo
            silo_training_step.compute = silo_config.computes[0]

            # set distribution according to the number of available GPUs (1 in case of only CPU available)
            silo_training_step.distribution.process_count_per_instance = silo_processes

            # set number of instances to distribute training across
            if hasattr(silo_config, "instance_count"):
                if silo_training_step.resources is None:
                    silo_training_step.resources = {}
                silo_training_step.resources[
                    "instance_count"
                ] = silo_config.instance_count

            # assign instance type for AKS, if available
            if hasattr(silo_config, "instance_type"):
                if silo_training_step.resources is None:
                    silo_training_step.resources = {}
                silo_training_step.resources[
                    "instance_type"
                ] = silo_config.instance_type

            # make sure the data is written in the right datastore
            silo_training_step.outputs.model = Output(
                type=AssetTypes.URI_FOLDER,
                mode="mount",
                path=custom_fl_data_path(
                    # IMPORTANT: writing the output of training into the orchestrator datastore
                    YAML_CONFIG.federated_learning.orchestrator.datastore,
                    f"model/silo{silo_index}",
                    iteration_num=iteration,
                ),
            )

            # each output is indexed to be fed into aggregate_component as a distinct input
            silo_weights_outputs[
                f"input_silo_{silo_index+1}"
            ] = silo_training_step.outputs.model

        # aggregate all silo models into one
        aggregate_weights_step = aggregate_component(**silo_weights_outputs)
        # this is done in the orchestrator compute
        aggregate_weights_step.compute = (
            YAML_CONFIG.federated_learning.orchestrator.compute
        )
        # assign instance type for AKS, if available
        if hasattr(silo_config, "instance_type"):
            aggregate_weights_step.resources = {
                "instance_type": silo_config.instance_type
            }
        # add a readable name to the step
        aggregate_weights_step.name = f"iteration_{iteration}_aggregation"

        # make sure the data is written in the right datastore
        aggregate_weights_step.outputs.aggregated_output = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(
                YAML_CONFIG.federated_learning.orchestrator.datastore,
                "aggregated_output",
                unique_id=pipeline_identifier,
                iteration_num=iteration,
            ),
        )

        # let's keep track of the checkpoint to be used as input for next iteration
        running_checkpoint = aggregate_weights_step.outputs.aggregated_output
    
    # register the final aggregated model
    register_model_step = register_model_component(
        # The final aggregated model
        aggregated_output=running_checkpoint,
        # The relative path where the registered models are located
        model_name=YAML_CONFIG.training_parameters.model_name
    )

    register_model_step.name = "final_model_register"
    # this is done in the orchestrator compute
    register_model_step.compute = (
        YAML_CONFIG.federated_learning.orchestrator.compute
    )
    register_model_step.outputs.output_model = Output(
            type=AssetTypes.MLFLOW_MODEL,
            mode="rw_mount",
            path=custom_fl_data_path(
                YAML_CONFIG.federated_learning.orchestrator.datastore,
                "final_model",
                unique_id=pipeline_identifier, # Note: this might be changed to a constant to get model versions in the future.
            ),
        )
    
    if not args.offline and args.register_components:
        # register the preprocessing component to the workspace
        silo_pre_processing_component_train = ML_CLIENT.create_or_update(silo_pre_processing_step_train.component)
        silo_pre_processing_component_test = ML_CLIENT.create_or_update(silo_pre_processing_step_test.component)
        print(f"Component {silo_pre_processing_component_train.name} with Version {silo_pre_processing_component_train.version} is registered")
        print(f"Component {silo_pre_processing_component_test.name} with Version {silo_pre_processing_component_test.version} is registered")

        # register the silo training component to the workspace
        silo_training_component = ML_CLIENT.create_or_update(silo_training_step.component)
        print(f"Component {silo_training_component.name} with Version {silo_training_component.version} is registered")
        
        # register the aggregate weights component to the workspace
        aggregate_weights_component = ML_CLIENT.create_or_update(aggregate_weights_step.component)
        print(f"Component {aggregate_weights_component.name} with Version {aggregate_weights_component.version} is registered")

        # register the register_model component to the workspace
        register_component = ML_CLIENT.create_or_update(register_model_step.component)
        print(f"Component {register_component.name} with Version {register_component.version} is registered")

    return {"registered_model": register_model_step.outputs.output_model}


pipeline_job_train = fl_ccfraud_basic()

# Inspect built pipeline
print(pipeline_job_train)

if not args.offline:
    print("Submitting the pipeline job to your AzureML workspace...")
    pipeline_job = ML_CLIENT.jobs.create_or_update(
        pipeline_job_train, experiment_name="fl_demo_ccfraud"
    )

    print("The url to see your live job running is returned by the sdk:")
    print(pipeline_job.services["Studio"].endpoint)

    webbrowser.open(pipeline_job.services["Studio"].endpoint)

    if args.wait:
        job_name = pipeline_job.name
        status = pipeline_job.status

        while status not in ["Failed", "Completed", "Canceled"]:
            print(f"Job current status is {status}")

            # check status after every 100 sec.
            time.sleep(100)
            try:
                pipeline_job = ML_CLIENT.jobs.get(name=job_name)
            except azure.identity._exceptions.CredentialUnavailableError as e:
                print(f"Token expired or Credentials unavailable: {e}")
                sys.exit(5)
            status = pipeline_job.status

        print(f"Job finished with status {status}")
        if status in ["Failed", "Canceled"]:
            sys.exit(1)
else:
    print("The pipeline was NOT submitted, omit --offline to send it to AzureML.")
