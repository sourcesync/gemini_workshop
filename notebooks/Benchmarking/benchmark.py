import sys
import datetime

import numpy as np
import time
import gnl_bindings
import gnl_bindings_utils as gbu
import random
import os

if False:
    import faiss

np.set_printoptions(threshold=sys.maxsize)


# Environment variable assert
assert os.environ['LD_LIBRARY_PATH'].find("00.17.00/libs") is not -1, "LD_LIBRARY_PATH is not set"

# Get the data path
dpath = sys.argv[1]
print("data path=",dpath)

def setUpGNL():
    print("GDL init...")
    s = gnl_bindings.gdl_init()
    if s:
        raise Exception('gnl_bindings.gdl_init failed with {}'.format(s))

    print("GDL find and alloc...")
    s, gdl_ctx = gnl_bindings.gdl_context_find(apuc_count=4, mem_size=0x10000000)  # need to change num of apuc
    if s:
        raise Exception('gnl_bindings.gdl_context_find failed with {}'.format(s))

    print("bindings init...")
    s = gnl_bindings.init()
    if s:
        raise Exception('gnl_bindings.init failed with {}'.format(s))

    print("bindings base context...")
    s, base_ctx = gnl_bindings.create_base_context(gdl_ctx)
    if s:
        raise Exception('gnl_bindings.create_base_context failed with {}'.format(s))
    
    print("bindings contexts...")
    s, gnl_ctxs = gnl_bindings.create_contexts(base_ctx, [4])  # need to change num of apuc
    if s:
        raise Exception('gnl_bindings.create_contexts failed with {}'.format(s))
    
    print("bindings pm ctl...")
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
    #s = gnl_bindings.gdl_context_free(gdl_ctx)
    #if s:
    #    raise Exception('gnl_bindings.gdl_context_free failed with {}'.format(s))
    s = gnl_bindings.gdl_exit()
    if s:
        raise Exception('gnl_bindings.gdl_exit failed with {}'.format(s))

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print("Invalid arguments.")
        sys.exit()


    print("Setting up GNL...")
    gdl_ctx, base_ctx, ctx = setUpGNL()
    print("Done.")

    print("Loading search db...")
    search_db = np.load("%s/bit_vector_train50_padded256.npy" % dpath)
    print("Loaded.")
    
    print("Loading queries db...")
    queries_db = np.load("%s/bit_vector_test50_padded256.npy" % dpath)
    print("Loaded.")

    # Initialize a bunch of tests.  A test is an array [ search_db_size, query_size, k ]
    tests = [ [ 10000, 5, 10 ], [ 10000, 5, 10 ], [ 10000, 5, 10 ],
                [ 10000, 50, 10 ], [ 10000, 50, 10 ], [ 10000, 50, 10 ],
                [ 10000, 500, 10 ], [ 10000, 500, 10 ], [ 10000, 500, 10 ],
                [ 10000, 1000, 10 ], [ 10000, 1000, 10 ], [ 10000, 1000, 10 ],
                [ 10000, 2500, 10 ], [ 10000, 2500, 10 ], [ 10000, 2500, 10 ],
                [ 10000, 5000, 10 ], [ 10000, 5000, 10 ], [ 10000, 5000, 10 ] ]

    # Capture results in this array
    benchmarks = []

    for test in tests:

        search_size = test[0]
        query_size = test[1]
        k = test[2]

        np_records = np.copy(search_db[0:search_size,:])
        print("Search DB Size:", np_records.shape)
        
        np_queries= np.copy(queries_db[0:query_size,:])
        print("Queries size:", np_queries.shape)

        faiss_np_records = np.copy( np_records )
        np_records = np.transpose(np_records, (1, 0))

        # define the shape of the output arrays according to the num of queries and k
        out_shape = (np_queries.shape[0], k)
        print("Outshape", out_shape)
        print("Records shape after transpose", np_records.shape)
        print("Queries shape after transpose", np_queries.shape)

        # define the output arrays
        np_out_vals = np.empty(out_shape, np.uint16)
        np_out_indices = np.empty(out_shape, np.uint32)
        # convert the numpy arrays to gnl

        s, gnl_out_vals = gbu.create_gnl_array_from_numpy(base_ctx, ctx, np_out_vals, gbu.gnl_type.GNL_U16, False)
        assert (s == 0)
        
        s, gnl_out_indices = gbu.create_gnl_array_from_numpy(base_ctx, ctx, np_out_indices, gbu.gnl_type.GNL_U32, False)
        assert (s == 0)

        s, gnl_data = gbu.create_gnl_array_from_numpy(base_ctx, ctx, np_records, gbu.gnl_type.GNL_U16)
        assert (s == 0)

        s, gnl_queries = gbu.create_gnl_array_from_numpy(base_ctx, ctx, np_queries, gbu.gnl_type.GNL_U16)
        assert (s == 0)

        start = time.time()
        print(datetime.datetime.now(),"Searching knn_hamming !!!!!!!!!!!")
        s = gnl_bindings.knn_hamming(ctx, gnl_out_vals, gnl_out_indices, gnl_queries, gnl_data, k)
        assert (s == 0)
        end = time.time()
        GNLSearchTime = end - start
        print(datetime.datetime.now(),"Search duration:", GNLSearchTime )

        # convert the output from gnl arrays to numpy
        s, np_out_vals = gbu.create_numpy_array_from_gnl(ctx, gnl_out_vals)
        assert (s == 0)
        s, np_out_indices = gbu.create_numpy_array_from_gnl(ctx, gnl_out_indices)
        assert (s == 0)
        
        # Clean up (delete gnl arrays)
        print("Bindings destroy 1")
        gnl_bindings.destroy_array(gnl_out_vals)
        print("Bindings destroy 2")
        gnl_bindings.destroy_array(gnl_out_indices)
        print("Bindings destroy 3")
        gnl_bindings.destroy_array(gnl_queries)
        print("Bindings destroy 4")
        gnl_bindings.destroy_array(gnl_data)

        #
        # FAISS Test
        #
        if False:
            start = time.time()
            print(datetime.datetime.now(),"Searching faiss knn_hamming !!!!!!!!!!!")
            index = faiss.IndexBinaryFlat(256)
            crec = np.ascontiguousarray(faiss_np_records)
            index.add(crec)
            qrec = np.ascontiguousarray(np_queries)
            D, I = index.search(qrec, k)
            end = time.time()
            FAISSSearchTime = end - start
        else:
            FAISSSearchTime = 0
        

        print(datetime.datetime.now(),"Search duration:", FAISSSearchTime )

        # capture results
        benchmarks.append( [ search_size, query_size, k, GNLSearchTime, FAISSSearchTime ] )


    f = open("benchmarks.csv","w")
    for b in benchmarks:
        f.write( "%d, %d, %d, %f, %f\n" % ( b[0], b[1], b[2], b[3], b[4] ) )
    f.flush()
    f.close()
    print("Wrote benchmarks results.")

    print("Before tearDown.")
    tearDownGNL( gdl_ctx, base_ctx, ctx )
    print("After tearDown.")

