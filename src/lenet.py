from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#from tensorflow.contrib.layers import flatten

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else -1,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  

 # unary = {"1":lambda x:x ,"2":lambda x: -x, "3":lambda x:tf.abs, "4":lambda x : tf.pow(x,2),"5":lambda x : tf.pow(x,3),
  #         "6":lambda x:tf.sqrt,"7":lambda x: tf.Variable(tf.truncated_normal([1], stddev=0.08))*x,
   #        "8":lambda x : x + tf.Variable(tf.truncated_normal([1], stddev=0.08)),"9":lambda x: tf.log(tf.abs(x)+10e-8),
    #       "10":lambda x:tf.exp,"11":lambda x:tf.sin,"12":lambda x:tf.sinh,"13":lambda x:tf.cosh,"14":lambda x:tf.tanh,"15":lambda x:tf.asinh,"16":lambda x:tf.atan,"17":lambda x: tf.sin(x)/x,
     #      "18":lambda x : tf.maximum(x,0),"19":lambda x : tf.minimum(x,0),"20":tf.sigmoid,"21":lambda x:tf.log(1+tf.exp(x)),
      #     "22":lambda x:tf.exp(-tf.pow(x,2)),"23":lambda x:tf.erf,"24":lambda x: tf.Variable(tf.truncated_normal([1], stddev=0.08))}
       
 # binary = {"1":lambda x,y: x+y,"2":lambda x,y:x*y,"3":lambda x,y:x-y,"4":lambda x,y:x/(y+10e-8),
  #       "5":lambda x,y:tf.maximum(x,y),"6":lambda x,y: tf.sigmoid(x)*y,"7":lambda x,y:tf.exp(-tf.Variable(tf.truncated_normal([1], stddev=0.08))*tf.pow(x-y,2)),
   #      "8":lambda x,y:tf.exp(-tf.Variable(tf.truncated_normal([1], stddev=0.08))*tf.abs(x-y)),
    #     "9":lambda x,y: tf.Variable(tf.truncated_normal([1], stddev=0.08))*x + (1-tf.Variable(tf.truncated_normal([1], stddev=0.08)))*y}
        


  unary = {"1":lambda x:x ,"2":lambda x: -x, "3": lambda x: tf.maximum(x,0), "4":lambda x : tf.pow(x,2),"5":lambda x : tf.tanh(tf.cast(x,tf.float32))}
  binary = {"1":lambda x,y: tf.add(x,y),"2":lambda x,y:tf.multiply(x,y),"3":lambda x,y:tf.add(x,-y),"4":lambda x,y:tf.maximum(x,y),"5":lambda x,y: tf.sigmoid(x)*y}
  input_fun = {"1":lambda x:tf.cast(x,tf.float32) , "2":lambda x:tf.zeros(tf.shape(x)), "3": lambda x:2*tf.ones(tf.shape(x)),"4": lambda x : tf.ones(tf.shape(x)), "5": lambda x: -tf.ones(tf.shape(x))}

  with open("tmp","r") as f:
      activation = f.readline()
      activation = activation.split(" ")

  #inputs = binary[activation[8]](unary[activation[5]](binary[activation[4]](unary[activation[2]](input_fun[activation[0]](inputs)),unary[activation[3]](input_fun[activation[1]](inputs)))),unary[activation[7]](input_fun[activation[6]](inputs)))
  #inputs = binary[activation[4]]((unary[activation[2]](input_fun[activation[0]](inputs))),(unary[activation[3]](input_fun[activation[1]](inputs))))   #b[4](u1[2](x1[0]),u2[3](x2[1])) #core unit
  #inputs = binary[activation[2]]((unary[activation[0]](inputs)),(unary[activation[1]](inputs)))   #b[2](u1[0](x),u2[1](x)) #core unit

  inputs = tf.nn.relu(inputs)
  functions = open("./functions.txt", "a")
  functions.write(str(inputs) +  "\n")
  
  return inputs

def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
     inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

def cifar10_lenet5_generator(num_classes, data_format=None):
  """Generator for CIFAR-10 lenet models.
  """
  if data_format is None:
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
  def model(inputs, is_training):
    """Constructs the lenet model given the inputs."""
       # Hyperparameters
    #mu = 0
    #sigma = 0.1
    #x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')

    with tf.device("/gpu:0"):
        # Filter = [filter_height, filter_width, in_channels, out_channels]
        #conv1_filter = tf.Variable(tf.truncated_normal(shape = [5,5,3,6],mean = mu, stddev = sigma))
        #conv2_filter = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
        
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=6, kernel_size=5 , strides=1 ,data_format=data_format)
        inputs = tf.identity(inputs, 'initial_conv')

        inputs = batch_norm_relu(inputs, is_training, data_format)

        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=2 , strides=2 , padding='VALID',data_format=data_format)
        inputs = tf.identity(inputs, 'first_avg_pool')

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=16, kernel_size=5 , strides=1 ,data_format=data_format)
        inputs = tf.identity(inputs, 'Second_conv')

        inputs = batch_norm_relu(inputs, is_training, data_format)

        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=2 , strides=2 , padding='same',data_format=data_format)
        inputs = tf.identity(inputs, 'second_avg_pool')
        
        #inputs = tf.reshape(inputs, [-1, 400])
        inputs = tf.layers.flatten(inputs)

        inputs = tf.layers.dense(inputs=inputs, units=120)
        inputs = tf.identity(inputs, 'first_dense')

        inputs = batch_norm_relu(inputs, is_training, data_format)

        inputs = tf.layers.dense(inputs=inputs, units=84)
        inputs = tf.identity(inputs, 'second_dense')

        inputs = batch_norm_relu(inputs, is_training, data_format)

        inputs = tf.layers.dense(inputs=inputs, units=num_classes)
        inputs = tf.identity(inputs, 'third_dense')
    
    return inputs

  return model
