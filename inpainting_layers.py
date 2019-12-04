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