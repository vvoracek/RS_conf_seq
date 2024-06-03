from math import log 

import numpy as np 
from statsmodels.stats.proportion import proportion_confint as  pc 
from scipy.stats import binom 

def get_thresholds(p=0.1, alpha=0.001, n=10**7):
    hi = get_upper_threshold(p, alpha, n)
    lo = get_upper_threshold(1-p, alpha, n)
    lo = np.maximum.accumulate([i-j for i,j in enumerate(lo)])
    return list(lo), list(hi)

def get_upper_threshold(p, alpha, n):
    thresholds = [0]*(n)

    A = 0 
    N = 0
    logQ = 0

    while(N <= n-1):

        for A in range(A, N+1):
            logP = A * log(p) + (N-A) * log(1-p)
            if(logQ - logP > log(1/alpha/2) and A >= p*N):
                thresholds[N] = A 
                q = (A+0.5)/(N+1)
                logQ += log(1-q) 

                N += 1
                break 
            if(A != N):
                q = (A + 0.5)/N
                logQ = logQ - log(1-q) + log(q)
            else:
                thresholds[N] = A+n 
                q = (A+0.5)/(N+1)
                logQ += log(1-q) 
                N += 1

    cumin = n*2
    for i in range(len(thresholds))[::-1]:
        cumin = min(thresholds[i], cumin)
        thresholds[i] = cumin 
    return thresholds


def get_thresholds_union_bound(p=0.1, alpha=0.001, n=10**7):
    hi = get_upper_threshold_ub(p, alpha, n)
    lo = get_upper_threshold_ub(1-p, alpha, n)
    lo = np.maximum.accumulate([i-j for i,j in enumerate(lo)])
    return list(lo), list(hi)


def get_upper_threshold_ub(p, alph, n):
    beta = 1.1
    kinit = 11
    alpha = lambda t : 5*alph/(t+4)/(t+5)

    k = kinit

    up = np.arange(n)+100

    last = 0
    for i in range(1, n):
        if(i > beta ** k):
            k += 1
            up[i] = binom.ppf(1-alpha(k-kinit), i,p)+1
    cumin = max(up)    
    for i in range(len(up))[::-1]:
        cumin = min(up[i], cumin)
        up[i] = cumin 
    return up 
