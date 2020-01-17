import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import random

# constants
k = 5

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

def classify( db, query, I, plot=False):
    
    snrs = db.snrs
    labels = db.labels
    
    test_snrs = query.snrs
    test_labels = query.labels
    
    # vote function
    def vote(lst):
        return max(set(lst), key=lst.count)

    # get accuracy 
    total = dict(zip(range(-20,31), [0]*len(range(-20,31))))
    total_correct = dict(zip(range(-20,31), [0]*len(range(-20,31))))
    vote_labels = []
    for i in range(len(I)):
        class_idx = []
        q_snr = test_snrs[i]
        total[q_snr] = total[q_snr] + 1
        for j in range(k):
            tr_idx = I[i][j]
            class_idx.append(str(labels[tr_idx]))
        # get the class index from the vote label string
        vt = vote(class_idx)
        clsidx = eval(",".join(vt.split()) + ".index(1)" )
        info = {'v':clsidx,'i':I[i],'s':q_snr}
        vote_labels.append( info )
        if str(test_labels[i]) == str(vote(class_idx)):
            total_correct[q_snr] = total_correct[q_snr] + 1

    
    # plot accuracy data
    if (plot):
        
        for i in total.keys():
            if total[i] != 0:
                accuracy = round((total_correct[i]/total[i])*100, 2)
                print("Accuracy at snr = %d and k = %d: "%(i,k), accuracy)

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
        
    return vote_labels
        
    
def randisplay(predictions, db, query, threshold=0):
    
    ranset = random.sample( range(len(query.raw_signals) ), len(query.raw_signals) )
    
    t = range(1024)
    
    fig, ax = plt.subplots(k+1,3, figsize=(10,15) )
    #print(ax)
    
    qset = []
    for i in ranset:
        if query.snrs[i] > threshold:
            qset.append( i )
            if (len(qset)>2 ):
                break
    if len(qset)!=3:
        print("Could not find enough signals above threshold")
        return
    
    # iterate columns
    for c, qi in enumerate(qset):
        
        #print("c=",c)
        # Plot the query on top row
        first = query.raw_signals[qi][:,0]
        second = query.raw_signals[qi][:,1]
        ax[0,c].plot( t, first )
        ax[0,c].plot( t, second )
        
        # ground truth
        #idx = np.where(query.labels[qi] == 1)[0][0]
        
        pred = predictions[qi]
        idx = pred["v"]
        
        wave_type = classes[ idx ]
        title = "KNN predicts " + wave_type
        ax[0, c].set_title(title)
        xmax = 1024
        ymax = max([max(first),max(second)])
        ax[0, c].set_xticks([])
        ax[0, c].set_yticks([])
        
        #print(c,qi,pred)
        for nr in range(k):
        
            #print(pred)
            m = pred['i'][nr]
            first = db.raw_signals[m][:,0]
            second = db.raw_signals[m][:,1]
            ax[1+nr,c].plot( t, first )
            ax[1+nr,c].plot( t, second )
            idx = np.where(db.labels[m] == 1)[0][0]
            wave_type = classes[ idx ]
            title = "Top-%d is %s" % (nr+1, wave_type)
            ax[1+nr, c].set_title(title)
            xmax = 1024
            ymax = max([max(first),max(second)])
            ax[1+nr, c].set_xticks([])
            ax[1+nr, c].set_yticks([])
    
    