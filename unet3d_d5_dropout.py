# unet3d.py
#
#
# Qiyuan Tian 2019

from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU,Input, Conv3D, BatchNormalization, Activation, concatenate, MaxPooling3D, UpSampling3D, Dropout, LeakyReLU
from tensorflow.keras.utils import plot_model

def conv3d_dropout_relu(inputs, filter_num):
    inputs = Dropout(0.2)(inputs, training=True) # here is the difference
    conv = Conv3D(filter_num, (3,3,3), 
                  padding='same', 
                  activation=None,
                  kernel_initializer='he_normal')(inputs)
    conv=ReLU()(conv)
    return conv


def conv3d_relu(inputs, filter_num, bn_flag=False):
    conv = Conv3D(filter_num, (3,3,3), 
                  activation=None,
                  padding='same', 
                  kernel_initializer='he_normal')(inputs)
    conv=ReLU()(conv)
    return conv


def unet3d_model(input_ch, output_ch, filter_num=16, kinit_type='he_normal', tag='unet2d'):

    inputs = Input((None, None, None, input_ch))    
    loss_weights = Input((None, None, None, 1))

    p0 = inputs
        
    conv1 = conv3d_relu(p0, filter_num)
    conv1 = conv3d_relu(conv1, filter_num)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    conv2 = conv3d_relu(pool1, filter_num * 2)
    conv2 = conv3d_relu(conv2, filter_num * 2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    conv3 = conv3d_relu(pool2, filter_num * 4)
    conv3 = conv3d_relu(conv3, filter_num * 4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
   
    conv4 = conv3d_relu(pool3, filter_num * 8)
    conv4 = conv3d_relu(conv4, filter_num * 8)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = conv3d_relu(pool4, filter_num * 16)
    conv5 = conv3d_relu(conv5, filter_num * 16)

    up6 = UpSampling3D(size = (2, 2, 2))(conv5)
    merge6 = concatenate([conv4,up6])
    conv6 = conv3d_dropout_relu(merge6, filter_num * 8)
    conv6 = conv3d_dropout_relu(conv6, filter_num * 8)
    
    up7 = UpSampling3D(size = (2, 2, 2))(conv6)
    merge7 = concatenate([conv3,up7])
    conv7 = conv3d_dropout_relu(merge7, filter_num * 4)
    conv7 = conv3d_dropout_relu(conv7, filter_num * 4)

    up8 = UpSampling3D(size = (2, 2, 2))(conv7)
    merge8 = concatenate([conv2,up8])
    conv8 = conv3d_dropout_relu(merge8, filter_num * 2)
    conv8 = conv3d_dropout_relu(conv8, filter_num * 2)

    up9 = UpSampling3D(size = (2, 2, 2))(conv8)
    merge9 = concatenate([conv1,up9])
    conv9 = conv3d_dropout_relu(merge9, filter_num)
    conv9 = conv3d_dropout_relu(conv9, filter_num)
    
    conv = Conv3D(output_ch, (3, 3, 3), padding='same',
                  activation=None, 
                  kernel_initializer='he_normal'
                  )(conv9)
    
    recon = concatenate([conv, loss_weights],axis=-1)
        
    model = Model(inputs=[inputs, loss_weights], outputs=recon) 
#    plot_model(model, to_file='unet3d_d5.png', show_shapes=True)
    
    return model















