import numpy as np
import keras.backend as K
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          DepthwiseConv2D, ZeroPadding2D, Concatenate,
                          Flatten, Input, Reshape)
from tensorflow.python.keras.layers import Layer, InputSpec
from keras.models import Model

def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=1,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def relu6(x):
    return K.relu(x, max_value=6)


def mobilenet(input_tensor):
    #----------------------------主干特征提取网络开始---------------------------#
    # SSD结构,net字典
    net = {} 
    # Block 1
    x = input_tensor
    # 300,300,3 -> 150,150,64
    x = Conv2D(32, (3,3),
            padding='same',
            use_bias=False,
            strides=(2, 2),
            name='conv1')(input_tensor)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6, name='conv1_relu')(x)
    x = _depthwise_conv_block(x, 64, 1, block_id=1)
    
    # 150,150,64 -> 75,75,128
    x = _depthwise_conv_block(x, 128, 1,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, 1, block_id=3)

    
    # Block 3
    # 75,75,128 -> 38,38,256
    x = _depthwise_conv_block(x, 256, 1,
                              strides=(2, 2), block_id=4)
    
    x = _depthwise_conv_block(x, 256, 1, block_id=5)
    net['conv4_3'] = x

    # Block 4
    # 38,38,256 -> 19,19,512
    x = _depthwise_conv_block(x, 512, 1,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, 1, block_id=7)
    x = _depthwise_conv_block(x, 512, 1, block_id=8)
    x = _depthwise_conv_block(x, 512, 1, block_id=9)
    x = _depthwise_conv_block(x, 512, 1, block_id=10)
    x = _depthwise_conv_block(x, 512, 1, block_id=11)

    # Block 5
    # 19,19,512 -> 19,19,1024
    x = _depthwise_conv_block(x, 1024, 1,
                              strides=(1, 1), block_id=12)
    x = _depthwise_conv_block(x, 1024, 1, block_id=13)
    net['fc7'] = x

    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    # 19,19,512 -> 10,10,512
    net['conv6_1'] = Conv2D(256, kernel_size=(1,1), activation='relu',
                                   padding='same',
                                   name='conv6_1')(net['fc7'])
    net['conv6_2'] = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(net['conv6_1'])
    net['conv6_2'] = Conv2D(512, kernel_size=(3,3), strides=(2, 2),
                                   activation='relu',
                                   name='conv6_2')(net['conv6_2'])

    # Block 7
    # 10,10,512 -> 5,5,256
    net['conv7_1'] = Conv2D(128, kernel_size=(1,1), activation='relu',
                                   padding='same', 
                                   name='conv7_1')(net['conv6_2'])
    net['conv7_2'] = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(net['conv7_1'])
    net['conv7_2'] = Conv2D(256, kernel_size=(3,3), strides=(2, 2),
                                   activation='relu', padding='valid',
                                   name='conv7_2')(net['conv7_2'])
    # Block 8
    # 5,5,256 -> 3,3,256
    net['conv8_1'] = Conv2D(128, kernel_size=(1,1), activation='relu',
                                   padding='same',
                                   name='conv8_1')(net['conv7_2'])
    net['conv8_2'] = Conv2D(256, kernel_size=(3,3), strides=(1, 1),
                                   activation='relu', padding='valid',
                                   name='conv8_2')(net['conv8_1'])

    # Block 9
    # 3,3,256 -> 1,1,256
    net['conv9_1'] = Conv2D(128, kernel_size=(1,1), activation='relu',
                                   padding='same',
                                   name='conv9_1')(net['conv8_2'])
    net['conv9_2'] = Conv2D(256, kernel_size=(3,3), strides=(1, 1),
                                   activation='relu', padding='valid',
                                   name='conv9_2')(net['conv9_1'])
    #----------------------------主干特征提取网络结束---------------------------#
    return net


def SSD300(input_shape, num_classes=21):
    #---------------------------------#
    #   典型的输入大小为[300,300,3]
    #---------------------------------#
    input_tensor = Input(shape=input_shape)
    
    #------------------------------------------------------------------------#
    #   net变量里面包含了整个SSD的结构，通过层名可以找到对应的特征层
    #------------------------------------------------------------------------#
    net = mobilenet(input_tensor)
    
    #-----------------------将提取到的主干特征进行处理---------------------------#
    # 对conv4_3的通道进行l2标准化处理 
    # 38,38,512
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv4_3_loc'] = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same', name='conv4_3_loc')(net['conv4_3'])
    net['conv4_3_loc_flat'] = Flatten(name='conv4_3_loc_flat')(net['conv4_3_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv4_3_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv4_3_conf')(net['conv4_3'])
    net['conv4_3_conf_flat'] = Flatten(name='conv4_3_conf_flat')(net['conv4_3_conf'])

    # 对fc7层进行处理 
    # 19,19,1024
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['fc7_mbox_loc']         = Conv2D(num_priors * 4, kernel_size=(3,3),padding='same',name='fc7_mbox_loc')(net['fc7'])
    net['fc7_mbox_loc_flat']    = Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['fc7_mbox_conf']        = Conv2D(num_priors * num_classes, kernel_size=(3,3),padding='same',name='fc7_mbox_conf')(net['fc7'])
    net['fc7_mbox_conf_flat']   = Flatten(name='fc7_mbox_conf_flat')(net['fc7_mbox_conf'])

    # 对conv6_2进行处理
    # 10,10,512
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv6_2_mbox_loc']         = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv6_2_mbox_loc')(net['conv6_2'])
    net['conv6_2_mbox_loc_flat']    = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv6_2_mbox_conf']        = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv6_2_mbox_conf')(net['conv6_2'])
    net['conv6_2_mbox_conf_flat']   = Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])

    # 对conv7_2进行处理
    # 5,5,256
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv7_2_mbox_loc']         = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv7_2_mbox_loc')(net['conv7_2'])
    net['conv7_2_mbox_loc_flat']    = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv7_2_mbox_conf']        = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv7_2_mbox_conf')(net['conv7_2'])
    net['conv7_2_mbox_conf_flat']   = Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])

    # 对conv8_2进行处理
    # 3,3,256
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv8_2_mbox_loc']         = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv8_2_mbox_loc')(net['conv8_2'])
    net['conv8_2_mbox_loc_flat']    = Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv8_2_mbox_conf']        = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv8_2_mbox_conf')(net['conv8_2'])
    net['conv8_2_mbox_conf_flat']   = Flatten(name='conv8_2_mbox_conf_flat')(net['conv8_2_mbox_conf'])

    # 对conv9_2进行处理
    # 1,1,256
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv9_2_mbox_loc']         = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv9_2_mbox_loc')(net['conv9_2'])
    net['conv9_2_mbox_loc_flat']    = Flatten(name='conv9_2_mbox_loc_flat')(net['conv9_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv9_2_mbox_conf']        = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv9_2_mbox_conf')(net['conv9_2'])
    net['conv9_2_mbox_conf_flat']   = Flatten(name='conv9_2_mbox_conf_flat')(net['conv9_2_mbox_conf'])
    
    # 将所有结果进行堆叠
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([net['conv4_3_loc_flat'],
                                                            net['fc7_mbox_loc_flat'],
                                                            net['conv6_2_mbox_loc_flat'],
                                                            net['conv7_2_mbox_loc_flat'],
                                                            net['conv8_2_mbox_loc_flat'],
                                                            net['conv9_2_mbox_loc_flat']])
                                    
    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')([net['conv4_3_conf_flat'],
                                                            net['fc7_mbox_conf_flat'],
                                                            net['conv6_2_mbox_conf_flat'],
                                                            net['conv7_2_mbox_conf_flat'],
                                                            net['conv8_2_mbox_conf_flat'],
                                                            net['conv9_2_mbox_conf_flat']])
    # 8732,4
    net['mbox_loc']     = Reshape((-1, 4), name='mbox_loc_final')(net['mbox_loc'])
    # 8732,21
    net['mbox_conf']    = Reshape((-1, num_classes), name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf']    = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    # 8732,25
    net['predictions']  = Concatenate(axis =-1, name='predictions')([net['mbox_loc'], net['mbox_conf']])

    model = Model(input_tensor, net['predictions'])
    return model