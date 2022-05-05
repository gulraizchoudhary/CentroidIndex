# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 21:24:04 2021

@author: Gulraiz Iqbal Choudhary
"""

import numpy as np

def jaccard(x, y):
    len_x = len(x)
    len_y = len(y)
    fst, snd = (x, y) if len_x < len_y else (y, x)
    num_intersect = len(set(fst).intersection(snd))
    
    #Jaccard score    
    return (num_intersect / (len_x + len_y - num_intersect))

def compare(labels1, labels2, NUMBER_OF_CLUSTERS):
    results = np.zeros((NUMBER_OF_CLUSTERS,NUMBER_OF_CLUSTERS))
    for n1 in range(0, NUMBER_OF_CLUSTERS):
        indexes1 = np.where(labels1 == n1+1)
        for n2 in range(0, NUMBER_OF_CLUSTERS):
            indexes2 = np.where(labels2 == n2+1)
            score = jaccard(indexes1[0], indexes2[0])
            results[n1][n2] = int(score*100)
            
    return results



def CentroidIndex(LABELS1,GT):
    NUMBER_OF_CLUSTERS = len(set(GT))
    matrix = compare(LABELS1, GT, NUMBER_OF_CLUSTERS)
    
    #1 for greeen, 2 for red and 3 for blue- where the value is close to both clusters
    m = np.zeros(shape=(len(matrix), len(matrix)))
    
    for indx in range(0,len(matrix)):
        
        #mark green
        aIndx = matrix[indx,:].argmax() 
        
        #check if it is neighbor of the both clusters
        checkBlue = matrix[:,aIndx].argmax() 
        
        if(indx == checkBlue):
            m[indx,aIndx]=3
        else:
           m[indx,aIndx]=1 
        
        #mark red
        bIndx = matrix[:,indx].argmax()
        
        if m[bIndx,indx]==0:
            m[bIndx,indx]=2
            
    orphanA =0
    orphanB =0
            
    for indx in range(0,len(m)):
        if not (2 in m[indx,:] or 3 in m[indx,:]):
            orphanA+=1
        if not (1 in m[:,indx] or 3 in m[:,indx]):
            orphanB+=1
    
    CI = max(orphanA, orphanB)
    
    return CI
