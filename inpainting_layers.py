import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import tensorflow.keras.layers as layers

class InstanceNorm2D(layers.Layer):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-3
 
    def build(self, inputs_shape):
        self.beta = tf.Variable(tf.zeros((1, 1, 1, inputs_shape[-1]), dtype=tf.float32), 
                                aggregation=tf.VariableAggregation.MEAN, trainable=True, name="beta")
        self.gamma = tf.Variable(tf.ones((1, 1, 1, inputs_shape[-1]), dtype=tf.float32),
                                 aggregation=tf.VariableAggregation.MEAN, trainable=True, name="gamma")
        super().build(inputs_shape)

    def call(self, inputs):
        mu = tf.reduce_mean(inputs, axis=(1, 2), keepdims=True) # mean
        sigma = tf.reduce_mean((inputs - mu)** 2, axis=(1, 2), keepdims=True)  # variance
        out = tf.nn.batch_normalization(inputs, mean=mu, variance=sigma,
                offset=self.beta, scale=self.gamma, variance_epsilon=self.epsilon)
        return out

    def get_config(self):
        return super().get_config()

def _region_moments(inputs, flag, eps):
    flag_sum = tf.reduce_sum(tf.sign(flag), axis=(1, 2), keepdims=True) + eps
    mu = tf.reduce_sum(inputs * flag, axis=(1, 2), keepdims=True) / flag_sum
    sigma = tf.reduce_sum((inputs - mu)** 2 * flag, axis=(1, 2), keepdims=True) / flag_sum
    return mu, sigma

class RegionNormB(layers.Layer):
    def __init__(self):
        super().__init__()
        self.eps = 1e-3
 
    def build(self, inputs_shape):
        image_shape, _ = inputs_shape
        self.beta_mask = tf.Variable(tf.zeros((1, 1, 1, image_shape[-1]), dtype=tf.float32), 
                                    aggregation=tf.VariableAggregation.MEAN, trainable=True, name="beta_mask")
        self.beta_valid = tf.Variable(tf.zeros((1, 1, 1, image_shape[-1]), dtype=tf.float32), 
                                    aggregation=tf.VariableAggregation.MEAN, trainable=True, name="beta_valid")
        self.gamma_mask = tf.Variable(tf.ones((1, 1, 1, image_shape[-1]), dtype=tf.float32),
                                    aggregation=tf.VariableAggregation.MEAN, trainable=True, name="gamma_mask")
        self.gamma_valid = tf.Variable(tf.ones((1, 1, 1, image_shape[-1]), dtype=tf.float32),
                                    aggregation=tf.VariableAggregation.MEAN, trainable=True, name="gamma_valid")
        super().build(inputs_shape)
        
    def call(self, inputs):
        image, mask = inputs
        # Instance normalization for valid area (1 for valid, 0 for hole)
        mu, sigma = _region_moments(image, mask, self.eps)
        x_valid = tf.nn.batch_normalization(image, mean=mu, variance=sigma,
                    offset=self.beta_valid, scale=self.gamma_valid, variance_epsilon=self.eps)
        # x_valid = x_valid * mask
        # Instance normalization for masked area
        mu, sigma = _region_moments(image, 1.0- mask, self.eps)
        x_mask = tf.nn.batch_normalization(image, mean=mu, variance=sigma,
                    offset=self.beta_mask, scale=self.gamma_mask, variance_epsilon=self.eps)
        # x_mask = x_mask * (1.0- mask)
        return x_valid + x_mask

    def get_config(self):
        return super().get_config()
        
class RegionNormL(layers.Layer):
    def __init__(self):
        super().__init__()
        self.spatial_response_conv = layers.Conv2D(1, 3, padding="same", activation="sigmoid")
        self.gamma_conv = layers.Conv2D(1, 3, padding="same")
        self.beta_conv = layers.Conv2D(1, 3, padding="same")
        self.threshold = 0.8
        self.eps = 1e-3

    def call(self, inputs):
        # channel wise pooling
        maxpool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        avgpool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        x = tf.concat([maxpool, avgpool], axis=-1)
        # spatial response
        x = self.spatial_response_conv(x)
        # region mask
        region_mask = tf.sign(tf.nn.relu(x - self.threshold))
        # beta, gamma
        beta = self.beta_conv(x)
        gamma = self.gamma_conv(x)
        # region norm
        # valid region
        mu, sigma = _region_moments(inputs, region_mask, self.eps)
        x_valid = (inputs - mu) / tf.sqrt(sigma + self.eps)
        x_valid = x_valid * region_mask
        # mask region
        mu, sigma = _region_moments(inputs, 1.0- region_mask, self.eps)
        x_mask = (inputs - mu) / tf.sqrt(sigma + self.eps)
        x_mask = x_mask * (1.0- region_mask)
        # fusion
        x = x_valid + x_mask
        # pixelwise affine transform
        return x * beta + gamma

    def get_config(self):
        return super().get_config()

class RegionWiseConv(layers.Layer):
    def __init__(self, ch, strides=1, kernel_size=3, dilation_rate=1, conv="conv2d"):
        super().__init__()
        if conv == "conv2d":
            conv_layer = layers.Conv2D

        self.valid_conv = conv_layer(ch, kernel_size, strides=strides,
                               dilation_rate=dilation_rate, padding="same")
        self.mask_conv = conv_layer(ch, kernel_size, strides=strides,
                               dilation_rate=dilation_rate, padding="same")
        self.config = {
            "ch": ch,
            "strides": strides,
            "kernel_size": kernel_size,
            "dilation_rate": dilation_rate,
            "conv": conv
        }
        
    def call(self, inputs):
        image, mask = inputs
        x_valid = self.valid_conv(image)
        x_mask = self.mask_conv(image)
        x = x_valid * mask + x_mask * (1.0- mask)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, **self.config}        

def upsampling2d_tpu(inputs, scale=2):
    x = K.repeat_elements(inputs, scale, axis=1)
    x = K.repeat_elements(x, scale, axis=2)
    return x

# https://github.com/MathiasGruber/PConv-Keras
class PConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1, activation=None,
                 use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zeros', bias_regularizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):        
        """Adapted from original _Conv() layer of Keras        
        param input_shape: list of dimensions for [img, mask]
        """
                    
        if input_shape[0][-1] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
            
        self.input_dim = input_shape[0][-1]
        
        # Image kernel
        kernel_shape = (self.kernel_size, self.kernel_size, self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer)
        # Mask kernel
        self.kernel_mask = K.ones(shape=kernel_shape)

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size-1)/2), int((self.kernel_size-1)/2)), 
            (int((self.kernel_size-1)/2), int((self.kernel_size-1)/2)), 
        )

        # Window size - used for normalization
        self.window_size = self.kernel_size ** 2        
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        '''
        We will be using the Keras conv2d method, and essentially we have
        to do here is multiply the mask with the input X, before we apply the
        convolutions. For the mask itself, we apply convolutions with all weights
        set to 1.
        Subsequently, we clip mask values to between 0 and 1
        ''' 

        # Both image and mask must be supplied
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('PartialConvolution2D must be called on a list of two tensors [img, mask]. Instead got: ' + str(inputs))

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding)

        # Apply convolutions to mask
        mask_output = K.conv2d(
            masks, self.kernel_mask, 
            strides=self.strides,
            padding='valid',
            dilation_rate=self.dilation_rate
        )

        # Apply convolutions to image
        img_output = K.conv2d(
            (images*masks), self.kernel, 
            strides=self.strides,
            padding='valid',
            dilation_rate=self.dilation_rate
        )        

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)

        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output

        # Normalize iamge output
        img_output = img_output * mask_ratio

        # Apply bias only to the image (if chosen to do so)
        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias)
        
        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)
            
        return [img_output, mask_output]

    def get_config(self):
        base_config = super().get_config()
        current_config = {
            "filters" : self.filters,
            "kernel_size" : self.kernel_size,
            "strides" : self.strides,
            "dilation_rate" : self.dilation_rate,
            "activation" : self.activation,
            "use_bias" : self.use_bias,
            "kernel_initializer" : self.kernel_initializer,
            "kernel_regularizer" : self.kernel_regularizer,
            "bias_initializer" : self.bias_initializer,
            "bias_regularizer" : self.bias_regularizer            
        }
        return {**base_config, **current_config}
