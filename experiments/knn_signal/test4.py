import sys
import datetime

#sys.path.insert(0, '00.09.00')
#sys.path.append('00.09.00/gnlpy')
#sys.path.append('00.09.00/gnlpy/lib')
#sys.path.append('00.09.00/libs')

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
assert os.environ['LD_LIBRARY_PATH'].find("00.15.00/libs") is not -1, "LD_LIBRARY_PATH is not set"

def setUpGNL():
    print("GDL init...")
    s = gnl_bindings.gdl_init()
    if s:
        raise Exception('gnl_bindings.gdl_init failed with {}'.format(s))

    print("GDL find and alloc...")
    s, gdl_ctx = gnl_bindings.gdl_context_find_and_alloc(apuc_count=4, mem_size=0x10000000)  # need to change num of apuc
    if s:
        raise Exception('gnl_bindings.gdl_context_find_and_alloc failed with {}'.format(s))

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


    print("Setting up GNL...")
    gdl_ctx, base_ctx, ctx = setUpGNL()
    print("Done.")

    print("Loading search db...")
    search_db = np.load("bit_vector_train50_padded256.npy")
    print("Loaded.")
    
    print("Loading queries db...")
    queries_db = np.load("bit_vector_test50_padded256.npy")
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
        print("outshape", out_shape)
        print("records shape after transpose", np_records.shape)
        print("queries shape after transpose", np_queries.shape)

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
        print(datetime.datetime.now(),"searching knn_hamming !!!!!!!!!!!")
        s = gnl_bindings.knn_hamming(ctx, gnl_out_vals, gnl_out_indices, gnl_queries, gnl_data, k)
        assert (s == 0)
        end = time.time()
        GNLSearchTime = end - start
        print(datetime.datetime.now()," search duration:", GNLSearchTime )

        # convert the output from gnl arrays to numpy
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
        # FAISS Test
        #
        if False:
            start = time.time()
            print(datetime.datetime.now(),"searching faiss knn_hamming !!!!!!!!!!!")
            index = faiss.IndexBinaryFlat(256)
            crec = np.ascontiguousarray(faiss_np_records)
            index.add(crec)
            qrec = np.ascontiguousarray(np_queries)
            D, I = index.search(qrec, k)
            end = time.time()
            FAISSSearchTime = end - start
        else:
            FAISSSearchTime = 0
        

        print(datetime.datetime.now()," search duration:", FAISSSearchTime )

        # capture results
        benchmarks.append( [ search_size, query_size, k, GNLSearchTime, FAISSSearchTime ] )


    f = open("benchmarks.csv","w")
    for b in benchmarks:
        f.write( "%d, %d, %d, %f, %f\n" % ( b[0], b[1], b[2], b[3], b[4] ) )
    f.flush()
    f.close()

    tearDownGNL( gdl_ctx, base_ctx, ctx )

