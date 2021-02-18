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
gsi_search_apis = swagger_client.SearchApi(api_config)

# Inputs
num_of_boards = 1
dataset_path = "/path/to/dataset/npy/file"
metadata_path = "/path/to/metadata/pkl/file"
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

    # Search
    search_api_response = gsi_search_apis.apis_search(body=SearchRequest(allocation_id, dataset_id, queries_path=queries_path))
    indices = search_api_response.indices
    distance = search_api_response.distance
    metadata = search_api_response.metadata

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





