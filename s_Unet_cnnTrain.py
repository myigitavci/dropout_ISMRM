# %% load moduals
import os
import scipy.io as sio
import numpy as np
import nibabel as nb
import glob
from matplotlib import pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import qtlib as qtlib
from unet3d_d5_dropout import unet3d_model

# %% set up path
dpRoot = 'path'
os.chdir(dpRoot)

# %% subjects
subjects = sorted(glob.glob(os.path.join(dpRoot, 'mwu*')))

# %% load data 
img_block_all = np.zeros(1)
imgres_block_all = np.zeros(1)
mask_block_all = np.zeros(1)

for ii in np.arange(0,1 ):
    sj = os.path.basename(subjects[ii])
    print(sj)
    
    dpSub = os.path.join(dpRoot, sj);         

    fpImg = os.path.join(dpSub, 'diff-norm', sj + '_diff3_norm.nii.gz')
    img = nb.load(fpImg).get_data()    

    fpImgres = os.path.join(dpSub, 'metrics-norm', sj + '_metrics_norm.nii.gz')
    imgres = nb.load(fpImgres).get_data()    
    
    fpMask = os.path.join(dpSub, 'data', 'MASK_BRAIN_TISSUE.nii.gz')
    mask = nb.load(fpMask).get_data()
    mask = np.expand_dims(mask, 3)

    fpBind = os.path.join(dpSub, 'bind', sj + '_bind_b64.mat')
    bind = sio.loadmat(fpBind)['bind'] - 1

    img_block = qtlib.extract_block(img, bind)
    imgres_block = qtlib.extract_block(imgres, bind)
    mask_block = qtlib.extract_block(mask, bind)
    
    imgres_block = np.concatenate((imgres_block, mask_block), axis=-1)
    
    if imgres_block_all.any():
        img_block_all = np.concatenate((img_block_all, img_block), axis=0)
        imgres_block_all = np.concatenate((imgres_block_all, imgres_block), axis=0)
        mask_block_all = np.concatenate((mask_block_all, mask_block), axis=0)
    else:
        img_block_all = img_block
        imgres_block_all = imgres_block
        mask_block_all = mask_block

# %%
tmp = np.flip(img_block_all, 1)
img_block_all = np.concatenate((img_block_all, tmp), axis=0)

tmp = np.flip(imgres_block_all, 1)
imgres_block_all = np.concatenate((imgres_block_all, tmp), axis=0)

tmp = np.flip(mask_block_all, 1)
mask_block_all = np.concatenate((mask_block_all, tmp), axis=0)

# %% check data
plt.imshow(imgres_block_all[10, :, :, 40, 0], clim=(-2., 2.), cmap='gray')
plt.imshow(imgres_block_all[10, :, :, 40, 1], clim=(-2., 2.), cmap='gray')
plt.imshow(imgres_block_all[10, :, :, 40, 2], clim=(-2., 2.), cmap='gray')

plt.imshow(img_block_all[10, :, :, 40, 0], clim=(-2., 2.), cmap='gray')
plt.imshow(img_block_all[10, :, :, 40, 1], clim=(-2., 2.), cmap='gray')
plt.imshow(img_block_all[10, :, :, 40, 2], clim=(-2., 2.), cmap='gray')
plt.imshow(img_block_all[10, :, :, 40, 3], clim=(-2., 2.), cmap='gray')

plt.imshow(mask_block_all[10, :, :, 40, 0], clim=(-2., 2.), cmap='gray')

# %% set up model
nfilter = 32 ###################################################################################################
nin = 4
nout = 2
unet = unet3d_model(nin, nout, filter_num=nfilter)
unet.summary()

dpCnn = dpRoot+'/unet_1_subj_drop_02' ##########################################################
if not os.path.exists(dpCnn):
    os.mkdir(dpCnn)
    print('create directory')
        
# %% set up adam
adam_opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
unet.compile(loss = qtlib.mean_squared_error_weighted, optimizer = adam_opt)
        
# %% train unet
nbatch = 2
nepoch = 140

for ii in np.arange(0, nepoch):
    
    print('******************************************')
    print(ii)
    
    fnCp = 'unet_drop_lr1e4_ep' + np.str(ii)####################################################################
    fpCp = os.path.join(dpRoot, dpCnn, fnCp + '.h5') 
    checkpoint = ModelCheckpoint(fpCp, monitor='val_loss', save_best_only = True)
    
    history = unet.fit(x = [img_block_all, mask_block_all], 
                        y = imgres_block_all,
                        batch_size = nbatch, 
                        validation_split=0.2,
                        epochs = 1, 
                        callbacks = [checkpoint],
                        verbose = 1, shuffle = True) 
                        
    # save loos
    fpLoss = os.path.join(dpRoot, dpCnn, fnCp + '.mat') 
    sio.savemat(fpLoss, {'loss_train':history.history['loss'], 'loss_val':history.history['val_loss']})    

    if ii >= 15: # delete intermediate checkout files, which are large
        fnCp = 'unet_drop_lr1e4_ep' + np.str(ii - 15) ####################################################################
        fpCp = dpCnn+ '/'+fnCp +'.h5'
        os.remove(fpCp)