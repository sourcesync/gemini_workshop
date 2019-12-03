import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# constants
k = 5

def classify( db_labels, query_labels, I, plot=False):
    
    snrs, labels = db_labels
    
    test_snrs, test_labels = query_labels
    
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
        vote_labels.append( clsidx )
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
        
    return vote_labels
        