import time
from datetime import datetime
import sys
import os
sys.path.append('./Algos')
sys.path.append('./Utilities')
from Single_Dic_Learning import dictionary_learning as sparse_coefficient_learning
from Joint_Dic_SFM_Learning import spectral_dics_learning as joint_dics_learning
import utils as util
import numpy as np
import math
import scipy.io
import sklearn.preprocessing as sk
abc = np.load('./Data/Tahoe_preproc_t1.npz')
HSI=abc['HSI']
iterations=1
down_scal=4
nois=30
iter_spect_dict=2000
iter_spect_sparse=2000
dicSize_spectral=30
sparsity_spectral=2
L=HSI.shape[0]
W=HSI.shape[1]
HSI=HSI.transpose(2,0,1).reshape(HSI.shape[2],-1)
HSI = sk.normalize(HSI, norm='l2', axis=0)
HSI=HSI.T.reshape(L,W,-1)
Fh=np.copy(HSI)
Fm=np.copy(Fh)
MSI=abc['MSI']
MSI=MSI.transpose(2,0,1).reshape(MSI.shape[2],-1)
MSI = sk.normalize(MSI, norm='l2', axis=0)
MSI=MSI.T.reshape(L,W,-1)
M=np.copy(MSI)
R=abc['SRF']
Mo=np.copy(M)
H=util.blur_agverage_downSample(HSI,down_scal,np.mean)
l=H.shape[0]
w=H.shape[1]
channels=H.shape[2]
M=util.blur_agverage_downSample(M,down_scal,np.mean)
H=util.Nois_adding3d(nois, H)
Mo=util.Nois_adding3d(nois,Mo)
M=util.Nois_adding3d(nois,M)
HSI=HSI.transpose(2,0,1).reshape(HSI.shape[2],-1)
H=H.transpose(2,0,1).reshape(H.shape[2],-1)
Fm=Fm.transpose(2,0,1).reshape(Fm.shape[2],-1)
Fh=Fh.transpose(2,0,1).reshape(Fh.shape[2],-1)
M=M.transpose(2,0,1).reshape(M.shape[2],-1)
Mo=Mo.transpose(2,0,1).reshape(Mo.shape[2],-1)
samPFm=0.0
psnrPFm=0.0
ergasPFm=0.0
uqiPFm=0.0
samPFh=0.0
psnrPFh=0.0
ergasPFh=0.0
uqiPFh=0.0
Code_DPath="./Results/"
resultFile="Exp3_part1.txt"
tim1=0.0
for i in range(iterations):
    start_time=time.time()
    D1,Dm,Sd12,T=joint_dics_learning(M,H,dicSize_spectral,sparsity_spectral,iter_spect_dict)
    DDD, Sd=sparse_coefficient_learning(Mo,dicSize_spectral,sparsity_spectral,1,D1,0,iter_spect_sparse)
    Fmc=Dm.dot(Sd.T)
    Fhc=(Dm*T).dot(Sd.T)
    end_time=time.time()
    tim1=tim1+end_time-start_time
    sam, psnr,ergas, uqi=util.evaluationsRetValuesOnly(Fm, Fmc,L,W)
    samPFm=samPFm+sam
    psnrPFm=psnrPFm+psnr
    ergasPFm=ergasPFm+ergas
    uqiPFm=uqiPFm+uqi
    sam, psnr,ergas, uqi=util.evaluationsRetValuesOnly(Fh, Fhc,L,W)
    samPFh=samPFh+sam
    psnrPFh=psnrPFh+psnr
    ergasPFh=ergasPFh+ergas
    uqiPFh=uqiPFh+uqi
f = open(Code_DPath+resultFile,"a+")
genral_data = f"\n\nDate and time = {datetime.now()}, down_scal_factor = {down_scal}, iter_spect_dict = {iter_spect_dict}, iter_spect_sparse = {iter_spect_sparse}, \ndicSize_spectral ={dicSize_spectral}, sparsity_spectral ={sparsity_spectral}, Number of Experiment = {iterations}, Average_time Spectral= {tim1/iterations:0f}, nois = {nois} db\n"
f.write(genral_data)
f.write("Spectral result w.r.t. Fm\n")
f.write(f"samPFm = ({samPFm/iterations:5f}, {(samPFm/iterations)*180/np.pi:5f}), psnrPFm = {psnrPFm/iterations:5f}, ergasPFm = {ergasPFm/iterations:5f}, uqiPFm = {uqiPFm/iterations:5f}\n")
f.write("Spectral result w.r.t. Fh\n")
f.write(f"samPFh = ({samPFh/iterations:5f}, {(samPFh/iterations)*180/np.pi:5f}), psnrPFh = {psnrPFh/iterations:5f}, ergasPFh = {ergasPFh/iterations:5f}, uqiPFh = {uqiPFh/iterations:5f}\n")
f.close()
print(genral_data)
print("Spectral result w.r.t. Fm\n")
print(f"samPFm = ({samPFm/iterations:5f}, {(samPFm/iterations)*180/np.pi:5f}), psnrPFm = {psnrPFm/iterations:5f}, ergasPFm = {ergasPFm/iterations:5f}, uqiPFm = {uqiPFm/iterations:5f}\n")
print("Spectral result w.r.t. Fh\n")
print(f"samPFh = ({samPFh/iterations:5f}, {(samPFh/iterations)*180/np.pi:5f}), psnrPFh = {psnrPFh/iterations:5f}, ergasPFh = {ergasPFh/iterations:5f}, uqiPFh = {uqiPFh/iterations:5f}\n")
