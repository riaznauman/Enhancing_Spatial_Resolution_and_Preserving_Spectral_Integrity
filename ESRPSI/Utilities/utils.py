import numpy as np
import skimage
from scipy import interpolate
from scipy.ndimage import generic_laplace,uniform_filter,correlate,gaussian_filter, uniform_filter
import math
import sewar as sw
import sklearn.preprocessing as skk
import cv2
def SAM(Fr, Ft):
    Fr2d=Fr
    Ft2d=Ft
    if Fr.ndim==3:
        Fr2d=Fr.transpose(2,0,1).reshape(Fr.shape[2],-1)
    if Ft.ndim==3:
        Ft2d=Ft.transpose(2,0,1).reshape(Ft.shape[2],-1)
    
    sum=0.0
    for i in range(Fr2d.shape[1]):
        X = skk.normalize(Fr2d[:,i:i+1], norm='l2', axis=0)#[:,np.newaxis]
        Y=skk.normalize(Ft2d[:,i:i+1], norm='l2', axis=0)#[:,np.newaxis]
        dotp=X.T.dot(Y)
        sum=sum+np.arccos(dotp)
    sum=sum/Fr2d.shape[1]
    invcos=sum[0,0]
    return invcos

def upsampleArray(H, upSize):
    x=range(H.shape[1])
    y=range(H.shape[0])
    f = interpolate.interp2d(x, y, H, kind='linear')
    xnew=np.linspace(0,H.shape[1]-1, upSize)
    ynew=np.linspace(0,H.shape[0]-1, H.shape[0])
    znew = f(xnew, ynew)
    return znew
def resizeimage3d(img,W,H):
    imgc=np.zeros((H,W,img.shape[2]), dtype=np.float64)
    for i in range(img.shape[2]):
        imgc[:,:,i]=cv2.resize(img[:,:,i],(W,H),cv2.INTER_CUBIC)
    return imgc
def upsampleArray3d(H, upSize):
    for i in range(H.shape[2]):
        x=range(H.shape[1])
        y=range(H.shape[0])
        f = interpolate.interp2d(x, y, H[:,:,i], kind='linear')
        xnew=np.linspace(0,H.shape[1]-1, upSize)
        ynew=np.linspace(0,H.shape[0]-1, upSize)
        znew = f(xnew, ynew)
        if i==0:
            Hd=np.zeros((znew.shape[0],znew.shape[1],H.shape[2]), dtype=np.float64)
            Hd[:,:,i]=znew
        else:
            Hd[:,:,i]=znew
    return Hd
def downsampleArray(H,downSize):
    x=range(H.shape[1])
    y=range(H.shape[0])
    f = interpolate.interp2d(x, y, H, kind='linear')
    xnew=np.linspace(0,H.shape[1]-1, downSize)
    ynew=np.linspace(0,H.shape[0]-1, H.shape[0])
    znew = f(xnew, ynew)
    return znew
def downsampleArray3D(H,downSize):
    for i in range(H.shape[2]):
        x=range(H.shape[1])
        y=range(H.shape[0])
        f = interpolate.interp2d(x, y, H[:,:,i], kind='linear')
        xnew=np.linspace(0,H.shape[1]-1, downSize)
        ynew=np.linspace(0,H.shape[0]-1, downSize)
        znew = f(xnew, ynew)
        if i==0:
            Hd=np.zeros((znew.shape[0],znew.shape[1],H.shape[2]), dtype=np.float64)
            Hd[:,:,i]=znew
        else:
            Hd[:,:,i]=znew
    return Hd
def blur_gaussian_downSample(img,downsize):
    im_small = 0
    new_img=0
    for i in range(img.shape[2]):
        r = im_blurred = gaussian_filter(img[:,:,i], sigma=3.0)
        if i==0:
            new_img=np.zeros((r.shape[0],r.shape[1],img.shape[2]))
            new_img[:,:,i]=r
        else:
            new_img[:,:,i]=r
    w, h,j = new_img.shape
    w=math.floor(w/downsize)
    h=math.floor(h/downsize)
    im_small=new_img[0:-1:downsize,0:-1:downsize,:]
    return im_small
def PSNR(Fr, Ft):
    M=0
    sum=0.0
    if Fr.ndim==2:
        maxsum=0.0
        for i in range(Fr.shape[0]):
            maxsum=maxsum+np.max(Fr[i,:])
        maxsum=maxsum/Fr.shape[0]
        maxsum=maxsum*maxsum
        for i in range(Fr.shape[0]):
            sum=sum+10*np.log10(Fr.shape[1]*(maxsum)/np.sum((Fr[i,:]-Ft[i,:])**2))
        sum=sum/Fr.shape[0]
    elif Fr.ndim==3:
         M=Fr.shape[0]*Fr.shape[1]
         maxsum=0.0
         for i in range(Fr.shape[2]):
            maxsum=maxsum+np.max(Fr[:,:,i])
         maxsum=maxsum/Fr.shape[2]
         maxsum=maxsum*maxsum
         for i in range(Fr.shape[2]):
            sum=sum+10*np.log10(M*(maxsum)/np.sum((Fr[:,:,i]-Ft[:,:,i])**2))
         sum=sum/Fr.shape[2]
    return sum
def ERGAS(Fr,Ft,Resize_fact):
    sum=0.0
    M=0
    if Fr.ndim==2:
        M=Fr.shape[1]
        for i in range(Fr.shape[0]):
            sum=sum+np.sqrt(np.sum((Fr[i,:]-Ft[i,:])**2))/np.mean(Ft[i,:])**2
        sum=100*np.sqrt(sum/(Fr.shape[0]*M))/Resize_fact

    elif Fr.ndim==3:
        M=Fr.shape[0]*Fr.shape[1]
        for i in range(Fr.shape[2]):
            sum=sum+np.sqrt(np.sum((Fr[:,:,i]-Ft[:,:,i])**2))/np.mean(Ft[:,:,i])**2
        sum=100*np.sqrt(sum/(Fr.shape[2]*M))/Resize_fact
        ergas=sum
    return ergas
def blur_agverage_downSample(img,downsample,operation):
    new_img=0
    for i in range(img.shape[2]):
        r = skimage.measure.block_reduce(img[:, :, i],(downsample, downsample),operation)
        if i==0:
            new_img=np.zeros((r.shape[0],r.shape[1],img.shape[2]))
            new_img[:,:,i]=r
        else:
            new_img[:,:,i]=r
    return new_img
def blur_agverage_downSample2(img,downsize):
    im_small = 0
    new_img=0
    for i in range(img.shape[2]):
        r = uniform_filter(img[:,:,i], downsize)
        if i==0:
            new_img=np.zeros((r.shape[0],r.shape[1],img.shape[2]))
            new_img[:,:,i]=r
        else:
            new_img[:,:,i]=r
    w, h,j = new_img.shape
    w=math.floor(w/downsize)
    h=math.floor(h/downsize)
    im_small=new_img[0:-1:downsize,0:-1:downsize,:]
    return im_small

def chol_sample(mean, cov):
    return mean + np.linalg.cholesky(cov)
def Nois_adding(SNR, M):
    sig_avg_watts=np.mean(M,0)
    sig_avg_db=10*np.log10(sig_avg_watts)
    noise_avg_db=sig_avg_db-SNR
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise=0
    for i in range(M.shape[1]):
        noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts)[i], M.shape[0])[:,np.newaxis]
        M[:,i:i+1] = M[:,i:i+1] + noise
    return M
def Nois_adding3d(SNR, M):
    sig_avg_watts=np.mean(M,2)
    sig_avg_db=10*np.log10(sig_avg_watts)
    noise_avg_db=sig_avg_db-SNR
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise=0
    for i in range(M.shape[2]):
        noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), (M.shape[0],M.shape[1]))#[:,np.newaxis]
        M[:,:,i] = M[:,:,i] + noise
    return M
def Nois_adding2(SNR, M):
    sig_avg_watts=np.mean(M)
    sig_avg_db=10*np.log10(sig_avg_watts)
    noise_avg_db=sig_avg_db-SNR
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise=0
    for i in range(M.shape[1]):
            noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), M.shape[0])[:,np.newaxis]
            M[:,i:i+1] = M[:,i:i+1] + noise
    return M
def evaluationsRetValuesOnly(reference, target,L,W):
    t=0
    reference3d=reference.T.reshape(L,W,-1)
    target3d=target.T.reshape(L,W,-1)
    if t!=0:
        reference3d=reference3d[t:-t,t:-t,:]
        target3d=target3d[t:-t,t:-t,:]
    sam=SAM(reference,target)
    psnr=PSNR(reference, target)
    ergas = ERGAS(reference3d,target3d,16)
    uqi = sw.uqi(reference3d,target3d,8)   
    return sam, psnr,ergas, uqi
def angBvec(X,Y):
    X=X/np.sqrt(np.sum(X*X, 0))
    Y=Y/np.sqrt(np.sum(Y*Y, 0))
    z=np.arccos(np.sum(X*Y,0))*180/np.pi
    return z