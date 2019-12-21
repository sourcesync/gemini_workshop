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

    print("Loading search db...")
    search_db = np.load("data/test/db_1000000_256bits.npy")
    print("Loaded.")
    
    print("Loading queries db...")
    queries_db = np.load("data/test/queries_100000_256bits.npy")
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

        #
        # FAISS Test
        #
        index = faiss.IndexBinaryFlat(256)
        crec = np.ascontiguousarray(faiss_np_records)
        index.add(crec)
        qrec = np.ascontiguousarray(np_queries)
        start = time.time()
        print(datetime.datetime.now(),"searching faiss knn_hamming !!!!!!!!!!!")
        D, I = index.search(qrec, k)
        end = time.time()
        FAISSSearchTime = end - start
        print(datetime.datetime.now()," search duration:", FAISSSearchTime )

        # capture results
        benchmarks.append( [ search_size, query_size, k, FAISSSearchTime ] )

    f = open("benchmarks.csv","w")
    for b in benchmarks:
        f.write( "%d, %d, %d, %f\n" % ( b[0], b[1], b[2], b[3] ) )
    f.flush()
    f.close()

