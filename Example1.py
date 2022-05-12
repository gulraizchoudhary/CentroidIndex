# -*- coding: utf-8 -*-
"""
Created on Wed May 12 19:59:33 2022

@author: gulrch
"""

import numpy as np
from evaluation import CentroidIndex as ci
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#random 2D data set
X=np.random.rand(1000,2)

# number of centroids
k=50

for i in range(5):
    km = KMeans(n_clusters=k, init='random').fit(X)
    kmp = KMeans(n_clusters=k).fit(X)
    
    
    # relative SSE improvement of kmeans++ over kmeans
    imp = 1 - kmp.inertia_/km.inertia_
    print(f"SSE improvement over k-means: {imp:.2%}")
    
    #CI: Number of mismatch cluster from both solutions(kmeans++, kmeans)
    CI = ci.CentroidIndex(km.labels_, kmp.labels_)
    print(f"Mismatch between k-means and k-means++: {CI}")
    
    #plotting the k-means results
    for j in np.unique(km.labels_):
         plt.scatter(X[km.labels_ == j , 0] , X[km.labels_ == j , 1] , label = j)
    plt.scatter(km.cluster_centers_[:,0] , km.cluster_centers_[:,1] , s = 80, color = 'k')
    # displaying the title
    plt.title("k-means results of iteration: "+str(i))
    plt.show()
    
    #plotting the k-means++ results
    for j in np.unique(kmp.labels_):
         plt.scatter(X[kmp.labels_ == j , 0] , X[kmp.labels_ == j , 1] , label = j)
    plt.scatter(kmp.cluster_centers_[:,0] , kmp.cluster_centers_[:,1] , s = 80, color = 'k')
    # displaying the title
    plt.title("k-means++ results of iteration: "+str(i))
    plt.show()
    
    
    
    
