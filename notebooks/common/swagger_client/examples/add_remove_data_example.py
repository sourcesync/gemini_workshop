import numpy as np
import swagger_client
from swagger_client.models import *
from swagger_client.rest import ApiException

# Configuration
configuration = swagger_client.configuration.Configuration()
configuration.verify_ssl = False
configuration.host = "http://localhost:7761/v1.0"
api_config = swagger_client.ApiClient(configuration)

gsi_boards_apis = swagger_client.BoardsApi(api_config)
gsi_datasets_apis = swagger_client.DatasetsApi(api_config)

# Inputs
num_of_boards = 1
dataset_path = "/path/to/dataset/npy/file"
data_to_add = "/path/to/record/npy/file"
queries_path = "/path/to/queries/npy/file"

# Example
allocation_id = None
try:

    # Import Dataset
    response = gsi_datasets_apis.apis_import_dataset(body=ImportDatasetRequest(dataset_path, train_ind=True))
    dataset_id = response.dataset_id

    # Allocate Board/s
    response = gsi_boards_apis.apis_allocate(body=AllocateRequest(num_of_boards))
    allocation_id = response.allocation_id

    # Load Dataset
    gsi_datasets_apis.apis_load_dataset(body=LoadDatasetRequest(allocation_id, dataset_id, topk=25))

    # Add Record To Dataset
    data_to_add_records_list = np.load(data_to_add).tolist()
    gsi_datasets_apis.apis_add_data( body=AddDataRequest(allocation_id, dataset_id, data_to_add_records_list))

    # Remove Record From Dataset
    gsi_datasets_apis.apis_remove_data(body=RemoveDataRequest(allocation_id, dataset_id, 128000))

    # Commit Transaction(Apply changes to base dataset file)
    gsi_datasets_apis.apis_commit_transactions(body=CommitTransactionsRequest(allocation_id, dataset_id))

    # Unload Dataset
    gsi_datasets_apis.apis_unload_dataset(body=UnloadDatasetRequest(allocation_id, dataset_id))

except ApiException as e:
    print(e.status)
    print(e.body)
    print(e.reason)
    raise e

finally:
    # Deallocate Board/s
    if allocation_id is not None:
        gsi_boards_apis.apis_deallocate(body=DeallocateRequest(allocation_id))
