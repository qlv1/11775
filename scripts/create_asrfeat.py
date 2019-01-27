#!/bin/python
import numpy as np
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys

import re
def tokenizeDoc ( cur_doc ):
    return re. findall ( '\\w+', cur_doc )

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} vocab_file, file_list".format(sys.argv[0])
        print "vocab_file -- path to the vocabulary file"
        print "file_list -- the list of videos"
        exit(1)

    file_list = sys.argv[2]
    vocab_file = sys.argv[1]

    vocab = dict()
    fread = open(file_list,"r")

    i = 0
    for line in fread.readlines():
        asr_path = "asr/vocab/" + line.replace('\n','') + ".txt"
        if os.path.exists(asr_path) == False:
            continue

        f = open(asr_path, "r")
        text = tokenizeDoc(f.read())
        for word in text:
            if word not in vocab:
                vocab[word] = i
                i += 1
        f.close()

    vocab_size = i
    print (vocab_size)
    # np.savetxt(vocab_file, vocab)

    fread.close()
    fread = open(file_list,"r")

    i = 0
    for line in fread.readlines():
        asr_path = "asr/vocab/" + line.replace('\n','') + ".txt"
        if os.path.exists(asr_path) == False:
            continue

        X = np.zeros(vocab_size)
        f = open(asr_path, "r")
        text = tokenizeDoc(f.read())
        for word in text:
            X[vocab[word]] += 1
        f.close()

        if np.sum(X) != 0:
            X_norm = X.astype(float) / np.sum(X)
        if i == 0:
            X_all = X_norm
            i = 1
        else:
            X_all = np.vstack((X_all, X_norm))
        np.savetxt("asr_vector/" + line.replace('\n','') + ".asr.csv", X_norm.transpose())

    print "ASR features generated successfully!"
