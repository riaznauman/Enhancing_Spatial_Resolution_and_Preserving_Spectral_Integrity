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
abc = np.load('./Data/PaviaU.npz')
HSim=abc['HSim']
HSim=HSim[HSim.shape[0]-320:HSim.shape[0],HSim.shape[1]-320:HSim.shape[1],0:93]
iterations=1
down_scal=16
nois=25
iter_spect_dict=500
iter_spect_sparse=500
dicSize_spectral=25
sparsity_spectral=2
channels=HSim.shape[2]
L=HSim.shape[0]
W=HSim.shape[1]
H=util.blur_gaussian_downSample(HSim,down_scal)
l=H.shape[0]
w=H.shape[1]
channels=HSim.shape[2]
H=util.Nois_adding3d(nois,H)
H=H.transpose(2,0,1).reshape(H.shape[2],-1)
HSim=HSim.transpose(2,0,1).reshape(HSim.shape[2],-1)
Fm=np.copy(HSim)
Fh=np.copy(HSim)
R=abc['R3']
R=R[:,0:93]
M=R.dot(HSim)
M=M.T.reshape(L,W,-1)
Mo=np.copy(M)
Mo=util.Nois_adding3d(nois,Mo)
M=util.blur_gaussian_downSample(M,down_scal)
M=util.Nois_adding3d(nois, M)
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
resultFile="Exp2_part2.txt"
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