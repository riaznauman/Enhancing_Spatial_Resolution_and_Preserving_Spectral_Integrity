import time
import spams
import numpy as np
import sklearn.preprocessing as sk
import sklearn.preprocessing as skk
from sklearn.linear_model import OrthogonalMatchingPursuit
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
def initDSeqSelectionPaused(train,k):
    m, n = train.shape
    if k < n + 1:
        ichh = list(range(k))
    elif k > n:
        a = list(range(n))
        b = list(range(k - n))
        ichh = a + b
    Dinit = train[:, ichh]
    mn = np.matlib.repmat(np.mean(Dinit, 0), m, 1)
    sb = np.subtract(Dinit, mn)
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
########################################################################################################
####(Functions for Gibb Sampling)#####
def sample_Dk(k,Xd1, D1, Sd,g_eps_d1,g_eps_d2, Z,Dm,T, Xd2):
    Xdd1 = Xd1 + D1.dot(Z.T*Sd.T)
    Xdd2 = Xd2 + (Dm*T).dot(Z.T*Sd.T)
    sig_Dk=1.0/(D1.shape[0]+(g_eps_d1* np.sum((Z*Sd)**2)))
    mu_Dk=sig_Dk*(g_eps_d1*(np.sum(Xdd1*(Z.T*Sd.T),1))[:,np.newaxis])
    D1 = (mu_Dk + np.random.randn(D1.shape[0],1)*np.sqrt(sig_Dk))
    sig_Dk=1.0/(Dm.shape[0]+(g_eps_d2*(np.sum((Z*Sd)**2)*(T**2))))
    mu_Dk=sig_Dk*(g_eps_d2*T*(np.sum(Xdd2*(Z.T*Sd.T),1))[:,np.newaxis])#[:,np.newaxis]
    Dm = (mu_Dk + np.random.randn(Dm.shape[0],1)*np.sqrt(sig_Dk))
    Xd1 = Xdd1 - D1.dot(Z.T*Sd.T)
    Xd2 = Xdd2 - (Dm*T).dot(Z.T*Sd.T)
    return Xd1,Xd2, D1,Dm
def sample_T(Xd2,Dm,T,Sd,Z,g_eps_d2):
    Xdd2 = Xd2 + (Dm*T).dot(Z.T*Sd.T)
    sig_Dk=1.0/(T.shape[0]+(g_eps_d2*(Dm**2)*np.sum((Z*Sd)**2)))
    mu_Dk=sig_Dk*(T.shape[0]+g_eps_d2*Dm*(np.sum(Xdd2*(Z.T*Sd.T),1))[:,np.newaxis])
    T = (mu_Dk + np.random.randn(T.shape[0],1)*np.sqrt(sig_Dk))
    Xd2 = Xdd2 - (Dm*T).dot(Z.T*Sd.T)
    return Xd2,T
def sample_ZSk(k,Xd1, D1,Dm,T, Sd,Z, Pid, g_sd, g_eps_d1,g_eps_d2,Xd2):
    Xdd1 = Xd1 + D1.dot(Z.T*Sd.T)
    Xdd2 = Xd2 + (Dm*T).dot(Z.T*Sd.T)
    DTD1 = np.sum(D1** 2)
    DTD2 = np.sum((Dm*T)** 2)
    sigS1d = 1. / (g_sd + (g_eps_d1*(Z**2) * DTD1)+(g_eps_d2*(Z**2) * DTD2)) 
    SdM=sigS1d * (g_eps_d1 * Z*((Xdd1.T).dot(D1))+g_eps_d2 * Z*((Xdd2.T).dot(Dm*T)))
    Sd= np.random.randn(Sd.shape[0], 1) * np.sqrt(sigS1d) + SdM
    temp1 = - 0.5 * g_eps_d1* ((Sd ** 2) * DTD1-2 * Sd * ((Xdd1.T).dot(D1)))
    temp2 = - 0.5 * g_eps_d2 * ((Sd ** 2) * DTD2 - 2 * Sd * ((Xdd2.T).dot(Dm*T)))
    temp =Pid*np.exp(temp1+temp2)
    temp=1.0/temp
    bbb=1.0/(1+(1-Pid)*temp)
    Z=np.random.binomial(1,bbb)
    Xd1 = Xdd1 - D1.dot(Z.T*Sd.T)
    Xd2 = Xdd2 - (Dm*T).dot(Z.T*Sd.T)
    return Xd1, Xd2, Sd, Z
def sample_Pik(k,Z,Pid, a0, b0,K):
    aa=((a0)+np.sum(Z*Z))
    bb=((b0 * (K-1) /K)-np.sum(Z)+Z.shape[0])
    if bb<=0:
        bb=1.0e-6
    Pid=np.random.beta(aa,bb)
    return Pid
def sample_PikPaused(k,Z,Pid, a0, b0,K):
    aa=((a0/K)+Z)
    bb=((b0 * (K-1) /K)-Z)
    bb[bb<=0]=1.0e-6
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
def spectral_dics_learning(training_samples1, training_samples2,dicSize,sparsity,itrations):
    starttime = time.time()  # tic
    a0=1.0
    b0=1.0
    c0=60
    d0=1.0e-6
    e0=1.0e-6
    f0=1.0e-6
    if dicSize!=0:
        initDictSize=dicSize
    else:
        initDictSize = int(np.floor(1.25 * training_samples1.shape[1]))
    D1= initDSeqSelection(training_samples1,initDictSize)
    Dm= initDSeqSelection(training_samples2,initDictSize)
    Dm = skk.normalize(Dm, norm='l2', axis=0)
    D1 = skk.normalize(D1, norm='l2', axis=0)
    T=1.0*np.ones(Dm.shape,dtype=np.float32)
    Sd1 = ompAlgo(D1, training_samples1, sparsity)
    Sd2 = ompAlgo(Dm, training_samples2, sparsity)
    Sd=(Sd1+Sd2)/2.0
    Sd = skk.normalize(Sd, norm='l2', axis=0)
    K=initDictSize
    Xd1 = np.copy(training_samples1)
    Xd2 = np.copy(training_samples2)
    g_sd = 1.0
    g_eps_d1=1.0e+9
    g_eps_d2=1.0e+9
    Pid = 0.5 * np.ones((1, D1.shape[1]), dtype=np.float64)
    Z = np.copy(Sd)
    Z[Z !=0] = 1
    Z = Z.T
    Sd = Sd.T
    Xd1 = training_samples1 - D1.dot(Z.T*Sd.T)
    Xd2 = training_samples2- (Dm*T).dot(Z.T*Sd.T)
    for iter in range(itrations):
        print(f"Joint Dictionary and Scaling Factors Matrix Learning--Iteration No. = {iter+1}/{itrations})")
        for k in range(D1.shape[1]):
            g_sd=sample_g_sk(Sd[:,k:k+1],c0,d0,g_sd,K)
            Xd1,Xd2, D1[:,k:k+1],Dm[:,k:k+1] = sample_Dk(k,Xd1, D1[:,k:k+1], Sd[:,k:k+1], g_eps_d1,g_eps_d2, Z[:,k:k+1],Dm[:,k:k+1],T[:,k:k+1],Xd2)
            Xd2,T[:,k:k+1]=sample_T(Xd2,Dm[:,k:k+1],T[:,k:k+1],Sd[:,k:k+1],Z[:,k:k+1],g_eps_d2)
            Xd1, Xd2, Sd[:,k:k+1],Z[:,k:k+1] = sample_ZSk(k,Xd1, D1[:,k:k+1],Dm[:,k:k+1],T[:,k:k+1], Sd[:,k:k+1], Z[:,k:k+1],Pid[0,k],g_sd, g_eps_d1,g_eps_d2, Xd2)
            Pid[:,k:k+1]= sample_Pik(k,Z[:,k:k+1],Pid[0,k], a0, b0,K)
        g_eps_d1=sample_g_eps(Xd1, e0, f0)
        g_eps_d2=sample_g_eps(Xd2, e0, f0)
    return D1,Dm,Sd,T