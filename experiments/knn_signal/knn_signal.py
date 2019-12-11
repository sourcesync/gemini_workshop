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

# Environment variable assert
assert os.environ['LD_LIBRARY_PATH'].find("00.09.00/libs") is not -1, "LD_LIBRARY_PATH is not set"

def setUpSim():
    cfg = [gbu.sim_config(0xa0000000, 1, 4)]
    s = gnl_bindings.create_simulator(cfg)
    if s:
        raise Exception('gnl_bindings.create_simulator failed with {}'.format(s))

def tearDownSim():
    s = gnl_bindings.destroy_simulator()
    if s:
        raise Exception('gnl_bindings.destroy_simulator failed with {}'.format(s))

def setUpGNL():
    print(datetime.datetime.now(),"1")
    s = gnl_bindings.gdl_init()
    if s:
        raise Exception('gnl_bindings.gdl_init failed with {}'.format(s))
    s, gdl_ctx = gnl_bindings.gdl_context_find_and_alloc(apuc_count=4, mem_size=0x10000000)  # need to change num of apuc
    print(datetime.datetime.now(),"2")
    if s:
        raise Exception('gnl_bindings.gdl_context_find_and_alloc failed with {}'.format(s))
    s = gnl_bindings.init()
    print(datetime.datetime.now(),"3")
    if s:
        raise Exception('gnl_bindings.init failed with {}'.format(s))
    s, base_ctx = gnl_bindings.create_base_context(gdl_ctx)
    print(datetime.datetime.now(),"4")
    if s:
        raise Exception('gnl_bindings.create_base_context failed with {}'.format(s))
    s, gnl_ctxs = gnl_bindings.create_contexts(base_ctx, [4])  # need to change num of apuc
    print(datetime.datetime.now(),"5")
    if s:
        raise Exception('gnl_bindings.create_contexts failed with {}'.format(s))
    ctx = gnl_ctxs[0]
    s = gnl_bindings.pm_ctl(ctx, True)
    print(datetime.datetime.now(),"6")
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



def run_on_host(ctx):
    s, orig_flags = gnl_bindings.get_flags(ctx)
    assert (s == 0)
    inp_flags = orig_flags | gbu.gnl_global_flags.GNL_AVOID_DEV_USAGE
    s = gnl_bindings.set_flags(ctx, inp_flags)
    assert (s == 0)


if __name__ == '__main__':
    #if need to run on simulator uncomment the following line
    # setUpSim()
    if len(sys.argv)!=2:
        print("please provide iterations number or 0 for infinite execution.")
        sys.exit()
    else:
        num_iterations = int(sys.argv[1])
    print(datetime.datetime.now(),"before gnl setup")
    gdl_ctx, base_ctx, ctx = setUpGNL()
    print(datetime.datetime.now(),"after gnl setup")

    # define the knn we want to execute

    if False:
        path = 'places_365/'
        data = np.load(path + "Places_Vgg16_lsh_256_1803460_5x4.npz")
        np_records = np.array(data['arr_0'])
        data.close()
        print("orig")
        print(np_records.shape)
        print(type(np_records))
        print( np_records[0,:].shape )
        print( np_records[0,:].dtype )
        print(np_records)
        print("")

    if os.path.exists("bit_vector_train50_padded256.npy"):
        print("found padded train file, loading...")
        np_records = np.load("bit_vector_train50_padded256.npy")
        print("loaded.")
        
        print("train")
        print(np_records.shape)
        print(type(np_records))
        print( np_records[0,:].shape )
        print( np_records[0,:].dtype )
        print(np_records)

    else:
        # load the signal dataset
        path = "data/bit_vector/faiss_test/"
        np_records = np.load(path + "bit_vector_train50.npy")
        print("new")
        print(np_records.shape)
        print(type(np_records))
        print( np_records[0,:].shape )
        print( np_records[0,:].dtype )
        print(np_records)
        print("")
       
        # adjust the signal dataset to len 256
        new_records = []
        for i in range( np_records.shape[0] ):
            orig_rec = np_records[i,:]
            new_rec = np.append( np.zeros(32-7, np.uint8), orig_rec )
            new_records.append( new_rec )
            #print( new_rec.shape )
            #print( new_rec )
            #print( new_rec.dtype )
            #sys.exit(0)
        np_records = np.asarray( new_records )
        
        print("train")
        print(np_records.shape)
        print(type(np_records))
        print( np_records[0,:].shape )
        print( np_records[0,:].dtype )
        print(np_records)

        print("saving padded train file...")
        np.save("bit_vector_train50_padded256.npy", np_records)
        print("saved.")
    
    if os.path.exists("bit_vector_test50_padded256.npy"):
        print("found padded test file, loading...")
        np_queries_all = np.load("bit_vector_test50_padded256.npy")
        print("loaded.")
        
        print("test")
        print(np_queries_all.shape)
        print(type(np_queries_all))
        print( np_queries_all[0,:].shape )
        print( np_queries_all[0,:].dtype )
        print(np_queries_all)

    else:
        # load the signal dataset
        path = "data/bit_vector/faiss_test/"
        np_queries_all = np.load(path + "bit_vector_test50.npy")
       
        # adjust the signal dataset to len 256
        new_records = []
        for i in range( np_queries_all.shape[0] ):
            orig_rec = np_queries_all[i,:]
            new_rec = np.append( np.zeros(32-7, np.uint8), orig_rec )
            new_records.append( new_rec )
        np_queries_all = np.asarray( new_records )
        
        print("test")
        print(np_queries_all.shape)
        print(type(np_queries_all))
        print( np_queries_all[0,:].shape )
        print( np_queries_all[0,:].dtype )
        print(np_queries_all)

        print("saving padded test file...")
        np.save("bit_vector_test50_padded256.npy", np_queries_all)
        print("saved.")

    #define the k
    k = 200

    if False: # orig
        querie_idx = [1506]
        np_queries = np_records[querie_idx, :]
    elif False:  # separate arrays attempt
        np_queries = np_queries_all[querie_idx, :]
        #np_queries = np.transpose(np_queries, (1,0) )
    elif False: # same array attempt, choose last element in records array and readjust records array
        querie_idx = [ np_records.shape[0]-1 ]
        np_queries = np_records[querie_idx, :]
        np_records = np_records[0:(np_records.shape[0]-1),:]
    else: # same array attempt, choose second to last element
        querie_idx = [ 0 ]
        print("queryidx=",querie_idx)
        np_queries = np_records[querie_idx, :]
        np_records = np_records[1:,:]

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

    
    # execute search
    if num_iterations != 0:
        for i in range(num_iterations):
            # Act
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
            print("np_out_vals=",np_out_vals)
            s, np_out_indices = gbu.create_numpy_array_from_gnl(ctx, gnl_out_indices)
            assert (s == 0)
            print("np_out_indices=",np_out_indices)

            print("now faiss")
            index = faiss.IndexBinaryFlat(256)
            crec = np.ascontiguousarray(faiss_np_records)
            index.add(crec)
            qrec = np.ascontiguousarray(np_queries)
            D, I = index.search(qrec, k)
            print(D)
            print(I)

    else:
        while True:
            # Act
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
            print(s)
            assert (s == 0)
            s, np_out_indices = gbu.create_numpy_array_from_gnl(ctx, gnl_out_indices)
            assert (s == 0)

        
    # Clean up (delete gnl arrays)
    gnl_bindings.destroy_array(gnl_out_vals)
    gnl_bindings.destroy_array(gnl_out_indices)
    gnl_bindings.destroy_array(gnl_queries)
    gnl_bindings.destroy_array(gnl_data)
