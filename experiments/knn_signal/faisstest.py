import sys
import datetime

import numpy as np
import time
import random
import os
import faiss

np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
    if len(sys.argv)!=1:
        print("Invalid arguments.")
        sys.exit()

    if os.path.exists("data/bit_vector/faiss_test/bit_vector_train50.npy"):

        np_records = np.load("data/bit_vector/faiss_test/bit_vector_train50.npy")
        np_records = np_records[:10000,:]
        print(np_records.shape)
        print(type(np_records))
        print( np_records[0,:].shape )
        print( np_records[0,:].dtype )
        #print(np_records)

    else:
        print("Could not load file.")
        sys.exit(1)

    # now load labels
    labels = np.load("data/npy_data/signal_dataset/train/labels10000.npy")
    
    # now load snr
    snrs = np.load("data/npy_data/signal_dataset/train/snrs10000.npy")
    print("snrs", snrs.shape)

    k = 10
        
    querie_idx = [ 0, 1 ]
    print("queryidx=",querie_idx)

    np_queries = np_records[querie_idx, :]
    label_queries = labels[querie_idx,:]
    print("label queries", label_queries)

    snrs_queries = snrs[querie_idx]
    print("snrs queries", snrs_queries)

    np_records = np_records[1:,:]
    labels = labels[1:]
    snrs = snrs[1:]

    index = faiss.IndexBinaryFlat(56)
    crec = np.ascontiguousarray(np_records)
    index.add(crec)
    qrec = np.ascontiguousarray(np_queries)
    D, I = index.search(qrec, k)
    print(D)
    print(I)

    faiss_labels = labels[ I ]
    print("faiss_labels", faiss_labels)
    faiss_cls = np.argwhere( faiss_labels==1 )
    print("faiss_cls", faiss_cls)
    
    faiss_snrs = snrs[ I ]
    print("faiss_snrs", faiss_snrs)
