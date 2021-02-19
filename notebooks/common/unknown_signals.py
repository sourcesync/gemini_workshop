# import libraries
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt

# globals
loaded = False
data_dir = None
bit_vectors = None
raw_signals = None
snrs = None
#labels = None
path = "/efs/data/public/signal/bit_vector_test50_padded256_f32_50.npy" #TODO: hard-coded for the demo

def load(deepsignal_data_dir=None, count=50): #TODO: changed from 5000 for the demo

    '''Load the "known" signals database using the test set data.'''

    global loaded, data_dir, bit_vectors, raw_signals, labels, snrs
    if not loaded:

        # Get path to the dataset directory.
        if (deepsignal_data_dir):
            data_dir = deepsignal_data_dir
        else:
            data_dir = os.environ['DEEPSIGNAL_DATA_DIR']

        if not os.path.exists(data_dir):
            raise Exception("This is not a valid data directory->%s" % data_dir )

        print("Using rf signal dataset directory at %s" % data_dir )

        # Load the bit vector fingerprints.
        print("Loading fingerprints...")
        bit_vectors = np.load( os.path.join( data_dir, "bit_vector_test50_padded256.npy" ) )
        bit_vectors = bit_vectors[0:count,:]

        # Load the raw signal data.
        print("Loading raw signals...")
        raw_signals = np.load( os.path.join( data_dir, "test", "signals.npy" ))
        raw_signals = raw_signals[0:count,:]

        # Load the signal SNR values.
        print("Loading signal-to-noise values...")
        snrs = np.load( os.path.join( data_dir, "test", "snrs.npy" ) )
        snrs = snrs[0:count]

    loaded = True
    sz = bit_vectors.shape
    print("The RF signal database loaded successfully.  There are %d total signals." % sz[0]) #, of bitsize=%d." % ( sz[0], sz[1]*8 ))

def get_fingerprints():

    '''Retrieve the fingerprints array.'''

    return bit_vectors

def randisplay(color='white'):

    '''Choose 9 signals and randomly display.'''

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
        #idx = np.where(labels[j] == 1)[0][0]
        #wave_type = classes[ idx ]
        title = '? Unknown Radio Wave'
        ax[r,c].set_title(title, color=color, fontweight='bold')
        xmax = 1024
        ymax = max([max(first),max(second)])
        ax[r,c].set_xticks([])
        ax[r,c].set_yticks([])
    plt.show()
