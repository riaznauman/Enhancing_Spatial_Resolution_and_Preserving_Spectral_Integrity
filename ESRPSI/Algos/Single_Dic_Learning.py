import time
import spams
import numpy as np
import sklearn.preprocessing as sk
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy import *
from numpy import matlib
def initDSeqSelection(train,k):
    m, n = train.shape
    r=0
    ichh=list()
    for i in range(k):
        r=(i)%n
        ichh=ichh+[r]
    Dinit = train[:, ichh]
    mn = np.matlib.repmat(np.mean(Dinit, 0), m, 1)
    sb = np.subtract(Dinit, mn)
    Dinit = sk.normalize(sb, norm='l2', axis=0)
    return Dinit
def ompAlgo(Di,train_s,sparsity):
    np.random.seed(0)
    X = train_s
    X = np.asfortranarray(X, dtype=np.float64)
    D = np.copy(Di)
    D = np.asfortranarray(D / np.tile(np.sqrt((D * D).sum(axis=0)), (D.shape[0], 1)), dtype=np.float64)
    eps = 1.0
    numThreads = -1
    Alpha_init=spams.omp(X, D,L=sparsity,return_reg_path=False).toarray()
    return Alpha_init
def ompAlgoPaused(Di,train_s, sparsity):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
    i = 1
    for y in train_s.T:
        y = y[:, np.newaxis]
        omp.fit(Di, y)
        coef = omp.coef_[:, np.newaxis]
        if i == 1:
            i = 0
            Alpha_init = coef
        else:
            Alpha_init = np.hstack([Alpha_init, coef])
    return Alpha_init
########################################################################################################
####(Functions for Gibb Sampling)#####
def sample_Dk(k,Xd, D, Sd, g_eps_d, Z):
    m = Xd.shape[0]
    Xdd = Xd + D.dot(Z.T*Sd.T)
    sig_Dk=1/(m+(g_eps_d)* np.sum((Z*Sd)**2))
    mu_Dk=sig_Dk*g_eps_d*np.sum((Xdd)*(Z.T*Sd.T),1)[:,np.newaxis]
    D = (mu_Dk + np.random.randn(D.shape[0],1)*np.sqrt(sig_Dk))
    Xd = Xdd - D.dot(Z.T*Sd.T)
    return Xd, D
def sample_ZSk(k,Xd, D, Sd, Z, Pid, g_sd, g_eps_d,flag):
    Xdd = Xd + D.dot(Z.T*Sd.T)
    DTD = np.sum(D** 2)
    if flag!=2:
        sigS1d = 1. / (g_sd + (g_eps_d*(Z**2) * DTD))
        SdM=sigS1d * (g_eps_d * Z*((Xdd.T).dot(D)))
        Sd= np.random.randn(Sd.shape[0], 1) * np.sqrt(sigS1d) + SdM
    temp1 = - 0.5 * g_eps_d * ((Sd ** 2) * DTD - 2 * Sd * ((Xdd.T).dot(D)))
    temp=Pid*np.exp(temp1)
    Z[np.random.rand(Z.shape[1], 1) > ((1 - Pid) / (temp + 1 - Pid))] = 1
    Z[np.random.rand(Z.shape[1], 1) <= ((1 - Pid) / (temp + 1 - Pid))] = 0 
    Xd = Xdd - D.dot(Z.T*Sd.T)
    return Xd,Sd, Z
def sample_Pik(k,Z, a0, b0,K):
    aa=((a0)+np.sum(Z))
    bb=((b0 * (K-1)/(1.0*K))-np.sum(Z)+Z.shape[0])
    if bb<=0:
        bb=1.0e-6
    Pid=np.random.beta(aa,bb)
    return Pid
def sample_PikPaused(k,Z,Pid, a0, b0,K):
    aa=((a0/K)+Z-1)
    bb=((b0 * (K-1) /K)-Z)
    aa[aa<=0]=1.0e-6
    Pid=np.random.beta(aa,bb)
    return Pid
def sample_g_sk(Sd,c0,d0,g_s,K):
    a1 = c0
    a2=d0+0.5*np.sum(Sd**2)
    g_s=np.random.gamma(a1, a2)      
    return g_s
def sample_g_eps(X_k, e0, f0):
    e = e0 + 0.5 * X_k.shape[0] * X_k.shape[1]
    f = f0 + 0.5 * np.sum(X_k ** 2)
    g_eps = np.random.gamma(e, f)
    return g_eps
######################Gibbs Sampling main Function######################################    
def dictionary_learning(training_samples,dicSize,sparsity,flag,D,Sd,iterations):
    starttime = time.time()
    a0=1.0
    b0=1.0
    c0=60
    d0=1.0e-6
    e0=1.0e-6
    f0=1.0e-6
    if flag==1:
        D=D
        Xd=np.copy(training_samples)
        Sd=ompAlgo(D, Xd, sparsity)
        Sd = sk.normalize(Sd, norm='l2', axis=0)
    if flag==2:
        Sd=Sd
        D=initDSeqSelection(training_samples,Sd.shape[0])
        Xd=np.copy(training_samples)
    if flag!=1 and flag!=2:
        if dicSize!=0:
            initDictSize=dicSize
        else:
            initDictSize = int(np.floor(1.25 * training_samples.shape[1]))
        D=initDSeqSelection(training_samples,initDictSize)
        Xd = np.copy(training_samples)
        Sd=ompAlgo(D, Xd, sparsity)
    a0=1.0e-6
    b0=1.0e-6
    K=D.shape[1]
    g_sd = 1.0
    g_eps_d=1.0e+9
    Pid = 0.5 * np.ones((1, D.shape[1]), dtype=np.float64)
    Z = np.copy(Sd)
    Z[Z !=0] = 1
    Z = Z.T
    Sd = Sd.T
    Xd = Xd - D.dot(Z.T*Sd.T)
    iter=0
    while iter<iterations:
        print(f"Sparse Cofficients Learning--Iteration no. = {iter+1}/{iterations})")
        for k in range(D.shape[1]):
            g_sd=sample_g_sk(Sd[:,k:k+1],c0,d0,g_sd,K)
            if flag!=1:
                Xd, D[:,k:k+1] = sample_Dk(k,Xd, D[:,k:k+1], Sd[:,k:k+1], g_eps_d, Z[:,k:k+1])
            Xd,Sd[:,k:k+1],Z[:,k:k+1] = sample_ZSk(k,Xd, D[:,k:k+1], Sd[:,k:k+1], Z[:,k:k+1], Pid[0,k], g_sd, g_eps_d,flag)
            Pid[:,k:k+1]= sample_Pik(k,Z[:,k:k+1], a0, b0,K)
        g_eps_d=sample_g_eps(Xd, e0, f0)
        iter=iter+1
    return D,Sd