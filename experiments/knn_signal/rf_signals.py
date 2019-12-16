# import libraries
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt

# constants
bitsize = 50
sigcount=10000
classes = ['32PSK',
 '16APSK',
 '32QAM',
 'FM',
 'GMSK',
 '32APSK',
 'OQPSK',
 '8ASK',
 'BPSK',
 '8PSK',
 'AM-SSB-SC',
 '4ASK',
 '16PSK',
 '64APSK',
 '128QAM',
 '128APSK',
 'AM-DSB-SC',
 'AM-SSB-WC',
 '64QAM',
 'QPSK',
 '256QAM',
 'AM-DSB-WC',
 'OOK',
 '16QAM']

# globals
loaded = False
bit_vectors = None
raw_signals = None
labels = None
snrs = None

# load the "known" signals database using the training set data
def load():
    global loaded, bit_vectors, raw_signals, labels, snrs
    if not loaded:
        #path = 'data/bit_vector/faiss_test/bit_vector_train' + str(bitsize) + '.npy'
        path = 'bit_vector_train50_padded256.npy'
        bit_vectors = np.load(path)
        bit_vectors = bit_vectors[0:sigcount,:]
        path = 'data/npy_data/signal_dataset/train/signals' + str(sigcount) + '.npy'
        raw_signals = np.load(path)
        path = 'data/npy_data/signal_dataset/train/labels' + str(sigcount) + '.npy'
        labels = np.load(path)
        path = 'data/npy_data/signal_dataset/train/snrs' + str(sigcount) + '.npy'
        snrs = np.load(path)
    loaded = True
    sz = bit_vectors.shape
    print("The known RF signal database loaded successfully.  There are %d signals, of bitsize=%d." % ( sz[0], sz[1]*8 ))
   
# return the numpy array of bitvectors
def get_fingerprints():
    return bit_vectors

# return actual labels and signal-to-noise data
def get_labels():
    global snrs, labels
    return snrs, labels
   
# display 9 signals at random
def randisplay():
    global loaded, bit_vectors
    if not loaded:
        print("No data has been loaded.")
        return

    t = range(1024)

    subs = random.sample( range(len(raw_signals)), 9)

    fig, ax = plt.subplots(3,3, figsize=(10,10) )
    
    for i, j in enumerate(subs):
        first = raw_signals[j][:,0]
        second = raw_signals[j][:,1]
        r = int(i/3)
        c = i%3
        ax[r,c].plot(t, first)
        ax[r,c].plot(t, second)
        idx = np.where(labels[j] == 1)[0][0]
        wave_type = classes[ idx ]
        title = wave_type + ' Radio Wave'
        ax[r,c].set_title(title)
        xmax = 1024
        ymax = max([max(first),max(second)])
        ax[r,c].set_xticks([])
        ax[r,c].set_yticks([])
    plt.show()
