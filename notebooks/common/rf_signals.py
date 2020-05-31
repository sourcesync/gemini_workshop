# import libraries
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt

# define the class labels
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
data_dir = None
bit_vectors = None
raw_signals = None
labels = None
snrs = None

def load(deepsignal_data_dir=None, count=100000):

    '''Load the "known" signals database using the training set data.'''

    global loaded, data_dir, bit_vectors, raw_signals, labels, snrs
    if not loaded:

        # Get path to the dataset directory.
        if (deepsignal_data_dir):
            data_dir = deepsignal_data_dir
        else:
            data_dir = os.environ['DEEPSIGNAL_DATA_DIR']

        if not os.path.exists(data_dir):
            raise Exception("This is not a valid data directory->%s" % data_dir )

        print("Using radioml dataset directory at %s" % data_dir )

        # Load the bit vector fingerprints.
        print("Loading fingerprints...")
        bit_vectors = np.load( os.path.join( data_dir, "bit_vector_train50_padded256.npy" ) )
        bit_vectors = bit_vectors[0:count,:]

        # Load the raw signal data.
        print("Loading raw signals...")
        raw_signals = np.load( os.path.join( data_dir, "train", "signals.npy" ))
        raw_signals = raw_signals[0:count,:]

        # Load the raw signal labels.
        print("Loading signal labels...")
        labels = np.load( os.path.join( data_dir, "train", "labels.npy" ) )
        labels = labels[0:count,:]

        # Load the signal SNR values.
        print("Loading signal-to-noise values...")
        snrs = np.load( os.path.join( data_dir, "train", "snrs.npy" ) )
        snrs = snrs[0:count]

    loaded = True
    sz = bit_vectors.shape
    print("The RF signal database loaded successfully.  There are %d total signals." % sz[0]) #, of bitsize=%d." % ( sz[0], sz[1]*8 ))
   
def get_fingerprints():

    '''Return the numpy array of RF signal bitvector fingerprints.'''

    return bit_vectors

def get_labels():

    '''Return RF signal labels and signal-to-noise.'''

    global snrs, labels
    return snrs, labels
   
def randisplay():

    '''Display 9 signals at random.'''

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
