import tensorflow as tf, os, sys
import data

def conv_block(layer, fsize, training, name, pool=True):
    """
    Method to perform basic CNN convolution block pattern
    
      [ CONV --> BN --> RELU ] x2 --> POOL (optional)
    :params
      
      (tf.Tensor) layer : input layer
      (int) fsize : output filter size
      (tf.Tensor) training : boolean value regarding train/valid cohort
      (str) name : name of block 
      (bool) pool : if True, pooling is performed
    :return
      (tf.Tensor) layer : output layer 
    """
    with tf.variable_scope(name):

        for i in range(1, 3):

            layer = tf.layers.conv2d(layer, filters=fsize, kernel_size=(3, 3), padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-1), name='conv-%i' % i)
            layer = tf.layers.batch_normalization(layer, training=training, name='norm-%s' % i)
            layer = tf.nn.relu(layer, name='relu-%i' % i)

        if pool:
            pool = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='pool-%i' % i)

        return layer, pool

def convt_block(layer, concat, fsize, name):
    """
    Method to perform basic CNN convolutional-transpose block pattern
      CONVT (applied to `layer`) --> CONCAT (with `concat`) 
    :params
      
      (tf.Tensor) layer : input layer 
      (tf.Tensor) concat : tensor to be concatenated
      (str) name : name of block 
    :return
      (tf.Tensor) layer : output layer
    """
    with tf.variable_scope(name):

        layer = tf.layers.conv2d_transpose(layer, filters=fsize, kernel_size=2, strides=2, 
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-1),  name='convt')
        layer = tf.concat([layer, concat], axis=-1, name='concat')

        return layer

def create_unet(X, training, name='final'):
    """
    Method to implement U-net
    :params
      (tf.Tensor) X : input tensor
      (tf.Tensor) training : boolean value regarding train/valid cohort
    :return
      (tf.Tensor) layer : output layer
    Original Paper: https://arxiv.org/abs/1505.04597
    """
    # --- Contracting arm
    block1, pool1 = conv_block(X, 64, training, name='block01')
    block2, pool2 = conv_block(pool1, 128, training, name='block02')
    block3, pool3 = conv_block(pool2, 256, training, name='block03')
    block4, pool4 = conv_block(pool3, 512, training, name='block04')
    block5, pool5 = conv_block(pool4, 1024, training, name='block05', pool=False)
    tf.add_to_collection('checkpoints', block5)
    # --- Expanding arm
    block6 = convt_block(block5, block4, 512, name='block06')
    block7, _ = conv_block(block6, 512, training, name='block07', pool=False)

    block8 = convt_block(block7, block3, 256, name='block08')
    block9, _ = conv_block(block8, 256, training, name='block09', pool=False)

    block10 = convt_block(block9, block2, 128, name='block10')
    block11, _ = conv_block(block10, 128, training, name='block11', pool=False)

    block12 = convt_block(block11, block1, 64, name='block12')
    block13, _ = conv_block(block12, 64, training, name='block13', pool=False)

    # --- Collapse to number of classes
    # 1x1 convolution to collapse channels
    pred =tf.layers.conv2d(block13, 2, (1, 1), name=name, activation = tf.nn.softmax, padding='same')

    return pred

def get_train_layers():
    return ['block01', 'block02', 'block03', 'block04', 'block05', 'block06', 'block07', 'block08', 'block09', 'block10', 'block11', 'block12', 'block13']

def dice_score(y_pred, y_true, smooth=1e-7):
    intersection = 2 * tf.reduce_sum(y_pred * y_true, axis=[1,2,3])
    union = tf.reduce_sum(y_pred, axis=[1,2,3]) + tf.reduce_sum(y_true, axis=[1,2,3])
    return tf.div(2. * intersection + smooth, union + smooth)

def loss_dice(y_pred, y_true, smooth=1e-7):
    """
    Method to approximate Dice score loss function
      Dice (formal) = 2 x (y_pred UNION y_true) 
                      -------------------------
                       | y_pred | + | y_true | 
      Dice (approx) = 2 x (y_pred * y_true) + d 
                      -------------------------
                     | y_pred | + | y_true | + d 
      where d is small delta == 1e-7 added both to numerator/denominator to
      prevent division by zero.
    :params
        (tf.Tensor) y_pred : predictions 
        (tf.Tensor) y_true : ground-truth 
    :return
        (dict) scores : {
          'final': final weighted Dice score,
          0: score for class 1,
          1: score for class 2, ...
        }
    """
    
    dice_score_raw = dice_score(y_pred, y_true, smooth)
    # average across batches/channels
    return tf.reduce_mean(dice_score_raw, axis=-1)


