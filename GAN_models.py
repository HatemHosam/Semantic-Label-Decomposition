import os
import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras import layers

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    #result.add(tfa.layers.InstanceNormalization())


    result.add(tf.keras.layers.LeakyReLU())

    return result



def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result




def Generator_Pix2Pix_Concat(dataset == 'NYUV2'):
    
    if dataset == 'NYUV2':
        inputs = tf.keras.layers.Input(shape=[256,256,1])
    elif dataset == 'ADE20K':
        inputs = tf.keras.layers.Input(shape=[256,256,3])
    else:
        print('Invalid dataset choice!!')

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
      ]
        
    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
      ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Generator_Pix2Pix_Add( dataset = 'NYUV2'):
    if dataset == 'NYUV2':
        inputs = tf.keras.layers.Input(shape=[256,256,1])
    elif dataset == 'ADE20K':
        inputs = tf.keras.layers.Input(shape=[256,256,3])
    else:
        print('Invalid dataset choice!!')

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
      ]
        
    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
      ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Add()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def Generator_ResNet_ED_Concat(filter_root=64, depth=8, dataset = 'NYUV2', input_size=(256, 256, 1), activation='relu', batch_norm=True, final_activation='linear'):
    """
    Build ResNet-ED model with ResBlock.
    Args:
        filter_root (int): Number of filters to start with in first convolution.
        depth (int): How deep to go in UNet i.e. how many down and up sampling you want to do in the model. 
                    Filter root and image size should be multiple of 2^depth.
        n_class (int, optional): How many classes in the output layer. Defaults to 2.
        input_size (tuple, optional): Input image size. Defaults to (256, 256, 1).
        activation (str, optional): activation to use in each convolution. Defaults to 'relu'.
        batch_norm (bool, optional): To use Batch normaliztion or not. Defaults to True.
        final_activation (str, optional): activation for output layer. Defaults to 'linear'.
    Returns:
        obj: keras model object
    """
    if dataset == 'NYUV2':
       input_size = [256,256,1]
    elif dataset == 'ADE20K':
        input_size = [256,256,3]
    else:
        print('Invalid dataset choice!!')


    Conv = Conv2D
    MaxPooling = MaxPooling2D
    UpSampling = UpSampling2D
     
    inputs = Input(input_size)
    x = inputs
    # Dictionary for long connections
    long_connection_store = {}

    # Down sampling
    for i in range(depth):
        if i >3:
            j == 3
        else: j = i
            
        out_channel = 2**(j)* filter_root

        # Residual/Skip connection
        res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="Identity{}_1".format(i))(x)

        # First Conv Block with Conv, BN and activation
        conv1 = Conv(out_channel, kernel_size=3, padding='same')(x)
        if batch_norm:
            conv1 = BatchNormalization()(conv1)
        act1 = Activation(activation)(conv1)

        # Second Conv block with Conv and BN only
        conv2 = Conv(out_channel, kernel_size=3, padding='same')(act1)
        if batch_norm:
            conv2 = BatchNormalization()(conv2)

        resconnection = Add()([res, conv2])

        act2 = Activation(activation)(resconnection)

        # Max pooling
        if i < depth - 1:
            long_connection_store[str(i)] = act2
            x = MaxPooling(padding='same')(act2)
        else:
            x = act2

    # Upsampling
    for i in range(depth - 2, -1, -1):
        if i >3:
            j == 3
        else: j = i
        out_channel = 2**(j) * filter_root

        # long connection from down sampling path.
        long_connection = long_connection_store[str(i)]

        up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
        up_conv1 = Conv(out_channel, 2, activation='relu', padding='same')(up1)

        #  Concatenate.
        up_conc = Concatenate(axis--1)([up_conv1, long_connection])   #Add()([up_conv1, long_connection])#

        #  Convolutions
        up_conv2 = Conv(out_channel, 3, padding='same')(up_conc)
        if batch_norm:
            up_conv2 = BatchNormalization()(up_conv2)
        up_act1 = Activation(activation)(up_conv2)

        up_conv2 = Conv(out_channel, 3, padding='same')(up_act1)
        if batch_norm:
            up_conv2 = BatchNormalization()(up_conv2)

        # Residual/Skip connection
        res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False)(up_conc)

        resconnection = Add()([res, up_conv2])

        x = Activation(activation)(resconnection)

    # Final convolution
    output = Conv(3, 1, padding='same', activation=final_activation)(x)

    return Model(inputs, outputs=output, name='ResNet-ED_Concat')

def Generator_ResNet_ED_Add(filter_root=64, depth=8, dataset == 'NYUV2', input_size=(256, 256, 1), activation='relu', batch_norm=True, final_activation='linear'):
    """
    Build ResNet-ED model with ResBlock.
    Args:
        filter_root (int): Number of filters to start with in first convolution.
        depth (int): How deep to go in UNet i.e. how many down and up sampling you want to do in the model. 
                    Filter root and image size should be multiple of 2^depth.
        n_class (int, optional): How many classes in the output layer. Defaults to 2.
        input_size (tuple, optional): Input image size. Defaults to (256, 256, 1).
        activation (str, optional): activation to use in each convolution. Defaults to 'relu'.
        batch_norm (bool, optional): To use Batch normaliztion or not. Defaults to True.
        final_activation (str, optional): activation for output layer. Defaults to 'linear'.
    Returns:
        obj: keras model object
    """
    
    if dataset == 'NYUV2':
        input_size = [256,256,1]
    elif dataset == 'ADE20K':
        input_size = [256,256,3]
    else:
        print('Invalid dataset choice!!')
    
    Conv = Conv2D
    MaxPooling = MaxPooling2D
    UpSampling = UpSampling2D
     
    inputs = Input(input_size)
    x = inputs
    # Dictionary for long connections
    long_connection_store = {}

    # Down sampling
    for i in range(depth):
        if i >3:
            j == 3
        else: j = i
            
        out_channel = 2**(j)* filter_root

        # Residual/Skip connection
        res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="Identity{}_1".format(i))(x)

        # First Conv Block with Conv, BN and activation
        conv1 = Conv(out_channel, kernel_size=3, padding='same')(x)
        if batch_norm:
            conv1 = BatchNormalization()(conv1)
        act1 = Activation(activation)(conv1)

        # Second Conv block with Conv and BN only
        conv2 = Conv(out_channel, kernel_size=3, padding='same')(act1)
        if batch_norm:
            conv2 = BatchNormalization()(conv2)

        resconnection = Add()([res, conv2])

        act2 = Activation(activation)(resconnection)

        # Max pooling
        if i < depth - 1:
            long_connection_store[str(i)] = act2
            x = MaxPooling(padding='same')(act2)
        else:
            x = act2

    # Upsampling
    for i in range(depth - 2, -1, -1):
        if i >3:
            j == 3
        else: j = i
        out_channel = 2**(j) * filter_root

        # long connection from down sampling path.
        long_connection = long_connection_store[str(i)]

        up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
        up_conv1 = Conv(out_channel, 2, activation='relu', padding='same')(up1)

        #  Concatenate.
        up_conc = Add()([up_conv1, long_connection])   #Add()([up_conv1, long_connection])#

        #  Convolutions
        up_conv2 = Conv(out_channel, 3, padding='same')(up_conc)
        if batch_norm:
            up_conv2 = BatchNormalization()(up_conv2)
        up_act1 = Activation(activation)(up_conv2)

        up_conv2 = Conv(out_channel, 3, padding='same')(up_act1)
        if batch_norm:
            up_conv2 = BatchNormalization()(up_conv2)

        # Residual/Skip connection
        res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False)(up_conc)

        resconnection = Add()([res, up_conv2])

        x = Activation(activation)(resconnection)

    # Final convolution
    output = Conv(3, 1, padding='same', activation=final_activation)(x)

    return Model(inputs, outputs=output, name='ResNet-ED')

def Generator_Xception_ED_Concat(dataset = 'NYUV2'):
    
    if dataset == 'NYUV2':
        inputs = tf.keras.layers.Input(shape=[256,256,1])
    elif dataset == 'ADE20K':
        inputs = tf.keras.layers.Input(shape=[256,256,3])
    else:
        print('Invalid dataset choice!!')
    
    connect_layers = []
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual
    connect_layers.append(previous_block_activation)
    k = 1
    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256, 512, 512, 512, 512]:
        k += 1
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        x = layers.Activation("relu")(x)
        previous_block_activation = x  # Set aside next residual
        #if k != 7:
        connect_layers.append(previous_block_activation)
    ### [Second half of the network: upsampling inputs] ###
    print(len(connect_layers))
    j = 0
    connect_layers.reverse()
    for filters in [512, 512, 512, 512, 256, 128, 64, 32]:
        j += 1
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
       
        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        x = layers.Activation("relu")(x)
        previous_block_activation = x  # Set aside next residual
        if j < 7:
            x = layers.Concatenate()([previous_block_activation, connect_layers[j]])
        

    outputs = layers.Conv2D(3, 3, activation="linear", padding="same")(x)

    # Define the model
    model = Model(inputs, outputs)
    return model
    
def Generator_Xception_ED_Add(dataset = 'NYUV2'):
    
    if dataset == 'NYUV2':
        inputs = tf.keras.layers.Input(shape=[256,256,1])
    elif dataset == 'ADE20K':
        inputs = tf.keras.layers.Input(shape=[256,256,3])
    else:
        print('Invalid dataset choice!!')
    
    connect_layers = []
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual
    connect_layers.append(previous_block_activation)
    k = 1
    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256, 512, 512, 512, 512]:
        k += 1
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        x = layers.Activation("relu")(x)
        previous_block_activation = x  # Set aside next residual
        #if k != 7:
        connect_layers.append(previous_block_activation)
    ### [Second half of the network: upsampling inputs] ###
    print(len(connect_layers))
    j = -1
    connect_layers.reverse()
    for filters in [512, 512, 512, 512, 256, 128, 64, 32]:
        j += 1
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
       
        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        x = layers.Activation("relu")(x)
        previous_block_activation = x  # Set aside next residual
        if j < 7:
            x = layers.Add()([previous_block_activation, connect_layers[j]])
        

    outputs = layers.Conv2D(3, 3, activation="linear", padding="same")(x)

    # Define the model
    model = Model(inputs, outputs)
    return model

def Discriminator_Pix2Pix(dataset = 'NYUV2'):
    initializer = tf.random_normal_initializer(0., 0.02)
    if dataset == 'NYUV2':
        inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
    elif dataset == 'ADE20K':
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    else:
        print('Invalid dataset choice!!')
        
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer, activation='sigmoid')(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
    

def Discriminator_ResNet():
    initializer = tf.random_normal_initializer(0., 0.02)

    if dataset == 'NYUV2':
        inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
    elif dataset == 'ADE20K':
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    else:
        print('Invalid dataset choice!!')
        
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)
    
    Conv = Conv2D
    MaxPooling = MaxPooling2D
    UpSampling = UpSampling2D
    
    for i in range(3):
            
        out_channel = 2**(i)* 32

        # Residual/Skip connection
        res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="disc_Identity{}_1".format(i))(x)

        # First Conv Block with Conv, BN and activation
        conv1 = Conv(out_channel, kernel_size=3, padding='same')(x)
        conv1 = BatchNormalization()(conv1)
        act1 = Activation('relu')(conv1)

        # Second Conv block with Conv and BN only
        conv2 = Conv(out_channel, kernel_size=3, padding='same')(act1)
        conv2 = BatchNormalization()(conv2)

        resconnection = Add()([res, conv2])

        act2 = Activation('relu')(resconnection)

        # Max pooling
        x = MaxPooling(padding='same')(act2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(x) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(256, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer, activation='sigmoid')(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
    
    
def Discriminator_Xception():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)
    
    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64,64]:
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        x = layers.Activation("relu")(x)
        previous_block_activation = x  # Set aside next residual

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(x) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(128, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer, activation='sigmoid')(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)