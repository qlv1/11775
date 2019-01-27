#!/bin/python
import numpy as np
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model,"rb"))

    fread = open(file_list,"r")
    i = 0
    for line in fread.readlines():
        mfcc_path = "mfcc/" + line.replace('\n','') + ".mfcc.csv"
        if os.path.exists(mfcc_path) == False:
            continue
        array = np.genfromtxt(mfcc_path, delimiter=";")
        Y = kmeans.predict(array)
        Y_count = np.bincount(Y, minlength = cluster_num)
        Y_norm = Y_count.astype(float) / len(array)
        if i == 0:
            Y_all = Y_norm
            i = 1
        else:
            Y_all = np.vstack((Y_all, Y_norm))
        np.savetxt("mfcc_vector/" + line.replace('\n','') + ".mfcc.csv", Y_norm.transpose())

    print (Y_all)
    print (np.sum(Y_all, axis = 1))

    print "K-means features generated successfully!"
