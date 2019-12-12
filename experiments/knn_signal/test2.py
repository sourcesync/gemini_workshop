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
    print("init 1")
    s = gnl_bindings.gdl_init()
    if s:
        raise Exception('gnl_bindings.gdl_init failed with {}'.format(s))

    print("alloc 2")
    s, gdl_ctx = gnl_bindings.gdl_context_find_and_alloc(apuc_count=4, mem_size=0x10000000)  # need to change num of apuc
    if s:
        raise Exception('gnl_bindings.gdl_context_find_and_alloc failed with {}'.format(s))

    print("init 3")
    s = gnl_bindings.init()
    if s:
        raise Exception('gnl_bindings.init failed with {}'.format(s))

    print("4")
    s, base_ctx = gnl_bindings.create_base_context(gdl_ctx)
    if s:
        raise Exception('gnl_bindings.create_base_context failed with {}'.format(s))
    
    print("5")
    s, gnl_ctxs = gnl_bindings.create_contexts(base_ctx, [4])  # need to change num of apuc
    if s:
        raise Exception('gnl_bindings.create_contexts failed with {}'.format(s))
    
    print("6")
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

    gdl_ctx, base_ctx, ctx = setUpGNL()

    if os.path.exists("bit_vector_train50_padded256.npy"):

        print("found file, loading...")
        np_records = np.load("bit_vector_train50_padded256.npy")
        print("loaded.")
    
        np_records = np_records[0:10000,:]
        
        print("train")
        print(np_records.shape)
        print(type(np_records))
        print( np_records[0,:].shape )
        print( np_records[0,:].dtype )
        np_records = np_records[0:10000,:]

    else:
        print("Could not load file.")
        sys.exit(1)

    if os.path.exists("bit_vector_test50_padded256.npy"):
        print("found padded train file, loading...")
        np_queries = np.load("bit_vector_test50_padded256.npy")
        print("loaded.")

        np_queries=np_queries[0:5000,:]

        print("test")
        print(np_queries.shape)
        print(type(np_queries))
        print( np_queries[0,:].shape )
        print( np_queries[0,:].dtype )

    else:
        print("Could not load file.")
        sys.exit(1)


    # now load labels
    train_labels = np.load("data/npy_data/signal_dataset/train/labels10000.npy")
    test_labels = np.load("data/npy_data/signal_dataset/test/labels5000.npy")

    # now load snr
    train_snrs = np.load("data/npy_data/signal_dataset/train/snrs10000.npy")
    test_snrs = np.load("data/npy_data/signal_dataset/test/snrs5000.npy")

    k = 10
        
    querie_idx = [ 0,1 ]
    print("queryidx=",querie_idx)
    
    # np_queries = np_queries[querie_idx, :]

    label_queries = train_labels[querie_idx,:]
    snrs_queries = train_snrs[querie_idx]

    print("label queries=", label_queries)
    print("snrs queries=", snrs_queries)
    
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
    #print("ret", s)
    assert (s == 0)
    end = time.time()
    SearchTime = end - start
    print(datetime.datetime.now()," search duration:", str(end - start))
    # convert the output from gnl arrays to numpy
    s, np_out_vals = gbu.create_numpy_array_from_gnl(ctx, gnl_out_vals)
    assert (s == 0)
    #print("np_out_vals=",np_out_vals)
    s, np_out_indices = gbu.create_numpy_array_from_gnl(ctx, gnl_out_indices)
    assert (s == 0)
    #print("np_out_indices=",np_out_indices)

    gnl_labels = train_labels[ np_out_indices ]
    #print("gnl_labels", gnl_labels)
    gnl_cls = np.argwhere( gnl_labels==1 )
    #print("gnl_cls", gnl_cls)
    gnl_snrs = train_snrs[ np_out_indices ]
    #print("gnl_snrs", gnl_snrs)

    start = time.time()
    print(datetime.datetime.now(),"searching faiss knn_hamming !!!!!!!!!!!")
    index = faiss.IndexBinaryFlat(256)
    crec = np.ascontiguousarray(faiss_np_records)
    index.add(crec)
    qrec = np.ascontiguousarray(np_queries)
    D, I = index.search(qrec, k)
    #print(D)
    #print(I)
    end = time.time()
    SearchTime = end - start
    print(datetime.datetime.now()," search duration:", str(end - start))

    faiss_labels = train_labels[ I ]
    #print("faiss_labels", faiss_labels)
    faiss_cls = np.argwhere( faiss_labels==1 )
    #print("faiss_cls", faiss_cls)
    faiss_snrs = train_snrs[ I ]
    #print("faiss_snrs", faiss_snrs)

    # Clean up (delete gnl arrays)
    gnl_bindings.destroy_array(gnl_out_vals)
    gnl_bindings.destroy_array(gnl_out_indices)
    gnl_bindings.destroy_array(gnl_queries)
    gnl_bindings.destroy_array(gnl_data)
