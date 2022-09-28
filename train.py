import os
import numpy as np
import tensorflow as tf
import time
import glob
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Add, BatchNormalization, Input, Activation, Lambda, Concatenate
from GAN_models import Generator_Pix2Pix_Concat, Generator_Pix2Pix_Add, Generator_ResNet_ED_Concat, Generator_ResNet_ED_Add, Generator_Xception_ED_Concat, Generator_Xception_ED_Add, Discriminator_Pix2Pix, Discriminator_ResNet, Discriminator_Xception

# physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
# for device in physical_devices:
    # tf.config.experimental.set_memory_growth(device, True)
    

#mirrored_strategy = tf.distribute.MirroredStrategy()

BUFFER_SIZE = 1000
BATCH_SIZE = 64
IMG_WIDTH = 256
IMG_HEIGHT = 256


# In[ ]:



n_gpu = 2

seg_path = 'D:/NYUV2 GAN/imgs_seg40/'
image_path = 'D:/NYUV2 GAN/images/'

def imshow(image, figsize=(6,6)):
    image = np.uint8(image)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(image)
    

def read_image(file):
    print(file)
    #for file in seg_path:
    input_image = tf.io.read_file(seg_path+file+'.png')
    input_image = tf.image.decode_png(input_image, channels=1)
    #print(image_path+str(int(file.split('.')[0].split('_')[-1]))+'.png')
    target_image = tf.io.read_file(image_path+file+'.png')
    target_image = tf.image.decode_png(target_image, channels=3)
    
    input_image = tf.cast(input_image, dtype=tf.float32)
    target_image = tf.cast(target_image, dtype=tf.float32)
    return input_image, target_image


def random_jittering_mirroring(input_image, target_image, height=256, width=256):
    
    #resizing to 286x286
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_image = tf.image.resize(target_image, [height, width],method=tf.image.ResizeMethod.BILINEAR)
    
    if tf.random.uniform(()) > 0.5:
    # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)
        
        
    return input_image, target_image

def normalize(input_image, target_image):
    input_image = input_image #/ 127.5) - 1
    target_image = (target_image / 127.5) - 1
    return input_image, target_image



def preprocess_fn(image_path):
    input_image, target_image = read_image(image_path)
    input_image, target_image = random_jittering_mirroring(input_image, target_image)
    input_image, target_image = normalize(input_image, target_image)
    return input_image, target_image    


# In[10]:

with open('train.txt','r') as f:
    train_path = f.read().splitlines()  

with open('val.txt','r') as f:
    val_path = f.read().splitlines() 
 

# train_path = glob.glob('images_seg/*')


# # In[11]:


# val_path = glob.glob('images_seg/*')


# # In[12]:


# In[13]:
AUTOTUNE = tf.data.experimental.AUTOTUNE

batch_size = 16

train_dataset = tf.data.Dataset.from_tensor_slices(train_path)
#print(train_dataset)
train_dataset = train_dataset.map(preprocess_fn,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(batch_size)

#train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)



batch_size = 16

val_files = tf.data.Dataset.from_tensor_slices(val_path)
val_files = val_files.shuffle(256)
val_dataset = val_files.map(preprocess_fn)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)



#with mirrored_strategy.scope():
generator = Generator_ResNet_ED_Add()
discriminator = Discriminator_ResNet()
generator_optimizer = tf.keras.optimizers.Adam(4e-4, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)

generator.summary()
discriminator.summary()

loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


# In[31]:


def generator_loss(disc_generated_output, gen_output, target, real_labels):
    Lambda =  100
    bce_loss = loss(real_labels, disc_generated_output)

    gan_loss = tf.reduce_mean(bce_loss)
    #gan_loss = gan_loss/ n_gpu

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    #l1_loss = l1_loss / n_gpu
    #print(l1_loss)

    total_gen_loss = gan_loss + (Lambda * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


# In[32]:


def discriminator_loss(disc_real_output, disc_generated_output, real_labels, fake_labels):
    bce_loss_real = loss(real_labels, disc_real_output)
    real_loss = tf.reduce_mean(bce_loss_real)
    #real_loss = real_loss / n_gpu

    bce_loss_generated = loss(fake_labels, disc_generated_output)
    generated_loss = tf.reduce_mean(bce_loss_generated)
    #generated_loss = generated_loss / n_gpu

    total_disc_loss = real_loss + generated_loss
    #total_disc_loss = total_disc_loss / 2
    return total_disc_loss




def train_step(inputs):
    input_image, target = inputs
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        real_targets = tf.ones_like(disc_real_output)
        fake_targets = tf.zeros_like(disc_real_output)

        gen_total_loss, gen_gan_loss, l1_loss = generator_loss(disc_generated_output, gen_output, target, real_targets)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output, real_targets, fake_targets)
        

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
                                              
  
    return gen_gan_loss, l1_loss, disc_loss                                         



# In[35]:


EPOCHS = 500

# In[36]:



def distributed_train_step(dist_inputs):
    gan_l, l1_l, disc_l = mirrored_strategy.run(train_step, args=(dist_inputs,))
    gan_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, gan_l, axis=None)
    l1_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, l1_l, axis=None)
    disc_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, disc_l, axis=None) 
    return gan_loss, l1_loss, disc_loss 

l1 = []
gl = []
dl = []

def plotLoss(epoch, gLosses, dLosses, l1_loss):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.plot(l1_loss, label='L1 loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('train_improvement_ResUnet41_concat/gan_loss_epoch_%d.png' % epoch)
    plt.close()
    
def fit():
    for epoch in range(1,EPOCHS+1):
        n = 0
        gan_loss, l1_loss, disc_loss = 0, 0, 0
        for dist_inputs in train_dataset:
            n += 1
            gan_l, l1_l, disc_l = train_step(dist_inputs)
            gan_loss += gan_l
            l1_loss += l1_l
            disc_loss += disc_l
            
        gan_loss = gan_loss / n
        l1_loss = l1_loss / n
        disc_loss = disc_loss / n
        l1.append(l1_loss)
        gl.append(gan_loss)
        dl.append(disc_loss)
        plotLoss(epoch, gl, dl, l1)
        plt.close()
        print('Epoch: [%d/%d]: D_loss: %.3f, G_loss: %.3f, L1_loss: %.3f'  % (
            (epoch), EPOCHS, disc_loss, gan_loss, l1_loss))
        if epoch % 5 ==0:
            generator.save_weights('model_objective_ResUnet41_concat/gen_'+ str(epoch) + '.h5')
            discriminator.save_weights('model_objective_ResUnet41_concat/disc_'+ str(epoch) + '.h5')


fit()