import os
import math
import faiss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# constants
bitsize = 50
bank_size = 10000
q_size = 5000
k = 5

# globals
db_bit_vectors = None
loaded = False

def init( bit_vectors ):
    global loaded, db_bit_vectors
    db_bit_vectors = bit_vectors
    loaded = True
    print("GNL Initialized.")
    
def get_device_info():
    return "Found A Gemini Device (Simulated.)"
    
def search(bit_vectors_test):

    global db_bit_vectors 
    bit_vectors = db_bit_vectors
    
    # set padding
    p = int((math.ceil(bitsize/8.0))*8 - bitsize)
    padding = []
    for i in range(p):
        padding.append(0)

    # similarity search
    #bank_size = 100000
    #q_size = 50000
    #bank_size = 10000
    #q_size = 5000
    d = bitsize + len(padding)
    #k = 5

    # initialize database
    db = bit_vectors[:bank_size]
    #print('Size of database: ', len(db))
    # initialize query set
    queries = bit_vectors_test[:q_size]
    #print('Size of query set: ', len(queries))
    #print('\n')
    # Initializing index.
    index = faiss.IndexBinaryFlat(d)
    # Adding the database vectors.
    index.add(db)
    #print('Searching for nearest neighbors ...')
    D, I = index.search(queries, k)
    print('Search Complete!')
    return I

#gnl.get_accuracy( rf_signals.snrs, rf_signals.labels, unknown_signals.snrs, unknown_signals.labels, results, plot=True)

def get_accuracy(snrs, labels, test_snrs, test_labels, I, plot=False):
    
    # vote function
    def vote(lst):
        return max(set(lst), key=lst.count)

    # get accuracy 
    total = dict(zip(range(-20,31), [0]*len(range(-20,31))))
    total_correct = dict(zip(range(-20,31), [0]*len(range(-20,31))))
    for i in range(len(I)):
        class_idx = []
        q_snr = test_snrs[i]
        total[q_snr] = total[q_snr] + 1
        for j in range(k):
            tr_idx = I[i][j]
            class_idx.append(str(labels[tr_idx]))
        if str(test_labels[i]) == str(vote(class_idx)):
            total_correct[q_snr] = total_correct[q_snr] + 1

    for i in total.keys():
        if total[i] != 0:
            accuracy = round((total_correct[i]/total[i])*100, 2)
            print("Accuracy at snr = %d and k = %d: "%(i,k), accuracy)
    
    # plot accuracy data
    if (plot):
        x = []
        y = []
        keys = list(total_correct.keys())
        keys = [k for k in keys if total[k] != 0]
        keys.sort()
        for i in keys:
            x.append(i)
            y.append((total_correct[i]/total[i])*100)

        #figure(figsize=(15,8))
        plt.plot(x, y)
        plt.xlabel('Signal to Noise Ratio')
        plt.ylabel('Accuracy')
        plt.show()