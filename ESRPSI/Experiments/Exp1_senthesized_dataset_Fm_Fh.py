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
iterations=1
down_scal=4
nois=25
ps=100
iter_spect_dict=1000
iter_spect_sparse=1000
dicSize_spectral=25 
sparsity_spectral=2
abc = np.load('./Data/synthetic_exp1.npz')
Fm=abc['Fm']
Fh=abc['Fh']
R=abc['R']
print(R.shape)
R=R[0:4,:]
channels=Fh.shape[2]
L=Fh.shape[0]
W=Fh.shape[1]
H=util.blur_agverage_downSample(Fh,down_scal,np.mean)
H=util.Nois_adding3d(nois,H)
l=H.shape[0]
w=H.shape[1]
H=H.transpose(2,0,1).reshape(H.shape[2],-1)
Fm=Fm.transpose(2,0,1).reshape(Fm.shape[2],-1)
M=R.dot(Fm)
M=M.T.reshape(L,W,-1)
Mo=np.copy(M)
Mo=util.Nois_adding3d(nois,Mo)
M=util.blur_agverage_downSample(M,down_scal,np.mean)
M=util.Nois_adding3d(nois, M)
M=M.transpose(2,0,1).reshape(M.shape[2],-1)
Mo=Mo.transpose(2,0,1).reshape(Mo.shape[2],-1)
Fh=Fh.transpose(2,0,1).reshape(Fh.shape[2],-1)
samPFm=0.0
psnrPFm=0.0
ergasPFm=0.0
uqiPFm=0.0
samPFh=0.0
psnrPFh=0.0
ergasPFh=0.0
uqiPFh=0.0
Code_DPath="./Results/"
resultFile="Res_Exp1_senthesized_dataset_Fm_Fh.txt"
tim1=0.0
tim2=0.0
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
#results_dat= f"samPFm = {samPFm}, psnrPFm = {psnrPFm}, ergas = {ergasPFm}, qui = {uqiPFm}"
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