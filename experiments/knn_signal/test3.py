import sys
import datetime

sys.path.insert(0, '00.09.00')
sys.path.append('00.09.00/gnlpy')
sys.path.append('00.09.00/gnlpy/lib')
sys.path.append('00.09.00/libs')

import numpy as np
import time
import gnl_bindings
import gnl_bindings_utils as gbu
import random
import os
import faiss

np.set_printoptions(threshold=sys.maxsize)


# Environment variable assert
assert os.environ['LD_LIBRARY_PATH'].find("00.09.00/libs") is not -1, "LD_LIBRARY_PATH is not set"

def setUpGNL():
    s = gnl_bindings.gdl_init()
    if s:
        raise Exception('gnl_bindings.gdl_init failed with {}'.format(s))
    s, gdl_ctx = gnl_bindings.gdl_context_find_and_alloc(apuc_count=4, mem_size=0x10000000)  # need to change num of apuc

    if s:
        raise Exception('gnl_bindings.gdl_context_find_and_alloc failed with {}'.format(s))
    s = gnl_bindings.init()

    if s:
        raise Exception('gnl_bindings.init failed with {}'.format(s))
    s, base_ctx = gnl_bindings.create_base_context(gdl_ctx)

    if s:
        raise Exception('gnl_bindings.create_base_context failed with {}'.format(s))
    s, gnl_ctxs = gnl_bindings.create_contexts(base_ctx, [4])  # need to change num of apuc

    if s:
        raise Exception('gnl_bindings.create_contexts failed with {}'.format(s))
    ctx = gnl_ctxs[0]
    s = gnl_bindings.pm_ctl(ctx, True)

    if s:
        raise Exception('gnl_bindings.pm_ctl failed with {}'.format(s))
    return gdl_ctx, base_ctx, ctx

def tearDownGNL(gdl_ctx, base_ctx, ctx):
    s = gnl_bindings.pm_ctl(ctx, False)
    if s:
        raise Exception('gnl_bindings.pm_ctl failed with {}'.format(s))
    s = gnl_bindings.destroy_contexts(base_ctx)
    if s:
        raise Exception('gnl_bindings.destroy_contexts failed with {}'.format(s))
    s = gnl_bindings.destroy_base_context(base_ctx)
    if s:
        raise Exception('gnl_bindings.destroy_base_context failed with {}'.format(s))
    gnl_bindings.exit(False)
    s = gnl_bindings.gdl_context_free(gdl_ctx)
    if s:
        raise Exception('gnl_bindings.gdl_context_free failed with {}'.format(s))
    s = gnl_bindings.gdl_exit()
    if s:
        raise Exception('gnl_bindings.gdl_exit failed with {}'.format(s))

if __name__ == '__main__':
    if len(sys.argv)!=1:
        print("Invalid arguments.")
        sys.exit()

    # Setup GNL...
    print("Set Up GNL...")
    gdl_ctx, base_ctx, ctx = setUpGNL()
    print("Done.") 
   
    # Load the search dataset
    print("Loading Search DB...")
    search_db = np.load("bit_vector_train50_padded256.npy")
    print("Shape of search db array: ", search_db.shape)

    # Load the query dataset
    print("Loading Queries...")
    queries = np.load("bit_vector_test50_padded256.npy")
    print("Shape of queries array:", queries.shape)

    # Create a group of tests: [ db_size, query_size, k ]
    tests = [ [ 10000, 1, 10 ], [ 10000, 5, 10 ], [ 10000, 50, 10 ], [ 10000, 500, 10 ], [ 10000, 5000, 10 ] ]

    # Iterate over tests
    for test in tests:
       
        # Retrieve test parameters
        search_db_size = test[0]
        query_size = test[1]
        k = test[2]

        # Set size of search db
        test_search_db = search_db[0:search_db_size,:]
         
        # Set size of queries
        test_queries = queries[0:query_size, :]

        #
        # Begin GNL Test
        #
        gnl_search_db = np.copy( test_search_db )
        gnl_search_db = np.transpose( test_search_db, (1, 0) )

        # Define the shape of the output arrays according to the num of queries and k
        out_shape = ( queries.shape[0], k)
        print("outshape", out_shape)

        # Define the output arrays
        np_out_vals = np.empty(out_shape, np.uint16)
        np_out_indices = np.empty(out_shape, np.uint32)

        # Convert the numpy arrays to gnl
        s, gnl_out_vals = gbu.create_gnl_array_from_numpy(base_ctx, ctx, np_out_vals, gbu.gnl_type.GNL_U16, False)
        assert (s == 0)
        s, gnl_out_indices = gbu.create_gnl_array_from_numpy(base_ctx, ctx, np_out_indices, gbu.gnl_type.GNL_U32, False)
        assert (s == 0)
        s, gnl_data = gbu.create_gnl_array_from_numpy(base_ctx, ctx, gnl_search_db, gbu.gnl_type.GNL_U16)
        assert (s == 0)
        s, gnl_queries = gbu.create_gnl_array_from_numpy(base_ctx, ctx, queries, gbu.gnl_type.GNL_U16)
        assert (s == 0)

        # Search
        print(datetime.datetime.now(),"searching knn_hamming !!!!!!!!!!!")
        start = time.time()
        s = gnl_bindings.knn_hamming(ctx, gnl_out_vals, gnl_out_indices, gnl_queries, gnl_data, k)
        assert (s == 0)
        end = time.time()
        SearchTime = end - start
        print(datetime.datetime.now()," search duration:", SearchTime)
    
        # Convert the output from gnl arrays to numpy
        s, np_out_vals = gbu.create_numpy_array_from_gnl(ctx, gnl_out_vals)
        assert (s == 0)
        s, np_out_indices = gbu.create_numpy_array_from_gnl(ctx, gnl_out_indices)
        assert (s == 0)

        # Clean up (delete gnl arrays)
        gnl_bindings.destroy_array(gnl_out_vals)
        gnl_bindings.destroy_array(gnl_out_indices)
        gnl_bindings.destroy_array(gnl_queries)
        gnl_bindings.destroy_array(gnl_data)

        #
        # Begin FAISS Test
        #
        print(datetime.datetime.now(),"searching faiss knn_hamming !!!!!!!!!!!")
        start = time.time()
        index = faiss.IndexBinaryFlat(256)
        faiss_search_db = np.ascontiguousarray(test_search_db)
        index.add(faiss_search_db)
        faiss_queries = np.ascontiguousarray(queries)
        D, I = index.search(faiss_queries, k)
        end = time.time()
        SearchTime = end - start
        print(datetime.datetime.now()," search duration:", SearchTime )

        break
