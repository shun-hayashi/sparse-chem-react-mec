import numpy as np
from itertools import combinations_with_replacement
import itertools

def lr_all(n, n_body):
    comb = []
    for i in range(n_body):
        comb = comb + list(combinations_with_replacement(range(2*n),i+1))
    lr = np.empty((len(comb), 2*n))
    for i in range(lr.shape[0]):
        for j in range(lr.shape[1]):
            lr[i,j] = comb[i].count(j)
    l = lr[:,:n]
    r = lr[:,n:]
    
    lr_index = np.empty(0)
    for i in range(l.shape[0]):
        if np.array_equal(l[i,:], r[i,:]): # 左辺と右辺が完全に一致
            lr_index = np.append(lr_index, i) 
    lr_index = [int(i) for i in lr_index]
    l = np.delete(l, lr_index, axis=0)
    r = np.delete(r, lr_index, axis=0)
    return l, r

def gernerate_full_lr(_conservation, n_body):
    conservation = np.array(_conservation)
    l, r = lr_all(conservation.shape[1], n_body)

    lr_index = np.empty(0)
    for i in range(l.shape[0]):
        if np.sum(l[i,:]*r[i,:]) != 0: # the same chemical species appears on both lhs and rhs.
            lr_index = np.append(lr_index, i)
        for j in range(conservation.shape[0]): # mass conservation
            if np.sum(l[i,:]*conservation[j]) != np.sum(r[i,:]*conservation[j]):
                lr_index = np.append(lr_index, i) 
        if np.sum(l[i,:]) > 2: # one or two reactant reactions
            lr_index = np.append(lr_index, i) 
    lr_index = [int(i) for i in lr_index]
    l = np.delete(l, lr_index, axis=0)
    r = np.delete(r, lr_index, axis=0)
    return l, r

def gernerate_full_lr_with_oxalate(conservation):
    n_body = 5
    l, r = gernerate_full_lr(conservation, n_body)
    
    l[:,3] = 0 # remove oxalic acid (index: 3) from lhs
    
    lr_index = np.empty(0)
    for i in range(l.shape[0]):
        if l[i,4] != 0: 
            lr_index = np.append(lr_index, i) # remove elementary steps that involves CO2 (index: 4) as a reactant
        if np.sum(l[i,:]) == 0: 
            lr_index = np.append(lr_index, i) 
    lr_index = [int(i) for i in lr_index]
    l = np.delete(l, lr_index, axis=0)
    r = np.delete(r, lr_index, axis=0)
    return l, r