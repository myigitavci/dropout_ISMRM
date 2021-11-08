# %% load moduals
import os
import glob
import scipy.io as sio
import numpy as np
import nibabel as nb
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import qtlib as qtlib

# %% set up path
dpRoot = 'path'
os.chdir(dpRoot)

# %% load model
fnCp = 'unet_1_subj_drop_02_weights' #######################################################33
fpCp = os.path.join(dpRoot,fnCp + '.h5') #######################################################
dtnet = load_model(fpCp, custom_objects={'mean_squared_error_weighted': qtlib.mean_squared_error_weighted});

# %%
subjects = sorted(glob.glob(os.path.join(dpRoot, 'mwu*')))


for ii in np.arange(0,1 ):   
    sj = os.path.basename(subjects[ii])
    print(sj)
    
    dpSub = dpRoot+'/'+sj        
    dpPred = dpSub+'/'+'pred'
    dpPred1=dpSub+'/'+'pred'
    dpPred = dpPred+'/'+ '1-subjects-cnn-drop-02'#######################################################
    if not os.path.exists(dpPred1):
       os.mkdir(dpPred1)
    if not os.path.exists(dpPred):
        os.mkdir(dpPred)
        print('create directory')

    fpImg = os.path.join(dpSub, 'diff', sj + '_diff3.nii.gz')
    img = nb.load(fpImg).get_data()    
    
    fpMask = os.path.join(dpSub, 'data', sj + '_diff_mask.nii.gz')
    mask = nb.load(fpMask).get_data()
    mask = np.expand_dims(mask, 3)

    fpBind = os.path.join(dpSub, 'bind', sj + '_bind_b64.mat')
    bind = sio.loadmat(fpBind)['bind'] - 1

    img_block = qtlib.extract_block(img, bind)
    mask_block = qtlib.extract_block(mask, bind)    
    
    for kk in np.arange(0, 100):
        print(kk)
        
        img_block_pred = np.zeros(img_block.shape[:-1] + (2,))
    
        for mm in np.arange(0, img_block.shape[0]):
            tmp = dtnet.predict([img_block[mm:mm+1, :, :, :, :], mask_block[mm:mm+1, :, :, :, :]]) 
            img_block_pred[mm:mm+1, :, :, :, :] = tmp[:, :, :, :, 0:2]
    
        fpPred = os.path.join(dpPred, fnCp + '_img_block_pred' + str(kk).zfill(3) + '.mat')
        sio.savemat(fpPred, {'img_block_pred': img_block_pred * mask_block})
            

























