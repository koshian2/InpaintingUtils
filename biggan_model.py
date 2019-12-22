import tensorflow as tf
import tensorflow.keras.layers as layers
from inpainting_layers import InstanceNorm2D, upsampling2d_tpu, ConvSN2D
from utils import distributed, Reduction

# conditional batch norm -> instance norm
def biggan_residual_block(inputs, ch, downsampling, use_normalize):
    # main path 
    if use_normalize:
        x = InstanceNorm2D()(inputs)
    else:
        x = inputs
    x = layers.ReLU()(x)
    if downsampling < -1:
        x = layers.Lambda(upsampling2d_tpu, arguments={"scale": abs(downsampling)})(x)
    x = ConvSN2D(ch, 3, padding="same")(x)
    if use_normalize:
        x = InstanceNorm2D()(x)
    x = layers.ReLU()(x)
    x = ConvSN2D(ch, 3, padding="same")(x)
    if downsampling > 1:
        x = layers.AveragePooling2D(downsampling)(x)
    # residual path
    if downsampling < -1:
        r = layers.Lambda(upsampling2d_tpu, arguments={"scale": abs(downsampling)})(inputs)
    else:
        r = inputs
    r = ConvSN2D(ch, 1)(r)
    if downsampling > 1:
        r = layers.AveragePooling2D(downsampling)(r)
    return layers.Add()([x, r])

def self_attention_non_local(inputs, conv_layer):
    batch, height, width, ch = inputs.shape
    # theta path
    theta = conv_layer(ch // 8, kernel_size=1)(inputs)
    theta = tf.reshape(theta, (-1, height * width, ch // 8)) #[B, HW, C/8]
    # phi path
    phi = conv_layer(ch // 8, kernel_size=1)(inputs)
    phi = layers.MaxPooling2D(2)(phi)  # [B, H/2, W/2, C/8]
    phi = tf.transpose(tf.reshape(phi, (-1, height * width // 4, ch // 8)), (0, 2, 1))  # [B, C/8, HW/4]
    # attention
    attn = tf.matmul(theta, phi)  # [B, HW, HW/4]
    attn = layers.Activation("softmax")(attn)
    # g path
    g = conv_layer(ch // 2, 1)(inputs)
    g = layers.MaxPooling2D(2)(g)  # [B, H/2, W/2, C/2]
    g = tf.reshape(g, (-1, height * width // 4, ch // 2))  # [B, HW/4, C/2]
    
    attn_g = tf.matmul(attn, g)  # [B, HW, C/2]
    attn_g = tf.reshape(attn_g, (-1, height, width, ch // 2))  # [B, H, W, C/2]
    attn_g = conv_layer(ch, 1)(attn_g)  # [B, H, W, C]
    sigma_ratio = tf.Variable(tf.zeros(1), name="sigma_ratio", trainable=True)
    return inputs + sigma_ratio * attn_g

def noise_to_image_generator():
    inputs = layers.Input((128,))
    x = layers.Reshape((4, 4, 8))(inputs)  # 本当はSN＋Dense
    x = ConvSN2D(1024, 1)(x)  # (4, 4, 1024)
    x = biggan_residual_block(x, 512, -2, True) # (8, 8, 512)
    x = biggan_residual_block(x, 256, -2, True)  # (16, 16, 256)
    x = biggan_residual_block(x, 128, -2, True)  # (32, 32, 256)
    x = self_attention_non_local(x, ConvSN2D)
    x = biggan_residual_block(x, 64, -2, True)  # (64, 64, 64)
    x = InstanceNorm2D()(x)
    x = layers.ReLU()(x)
    x = ConvSN2D(3, 3, padding="same", activation="tanh")(x)
    model = tf.keras.models.Model(inputs, x)
    return model

def image_to_image_generator():
    inputs = layers.Input((64, 64, 3))
    x = inputs
    for ch in [64, 128, 256]:
        x = ConvSN2D(ch, 3, padding="same", strides=2)(x)
        x = InstanceNorm2D()(x)
        x = layers.ReLU()(x)
    for d in [1, 1, 2, 2]:
        x = ConvSN2D(512, 3, padding="same", dilation_rate=d)(x)
        x = InstanceNorm2D()(x)
        x = layers.ReLU()(x) # (8, 8, 512)
    x = biggan_residual_block(x, 256, -2, True)  # (16, 16, 256)
    x = biggan_residual_block(x, 128, -2, True)  # (32, 32, 128)
    x = self_attention_non_local(x, ConvSN2D)
    x = biggan_residual_block(x, 64, -2, True)  # (64, 64, 64)
    x = InstanceNorm2D()(x)
    x = layers.ReLU()(x)
    x = ConvSN2D(3, 3, padding="same", activation="tanh")(x)
    model = tf.keras.models.Model(inputs, x)
    return model

def discriminator(use_patch_gan):
    inputs = layers.Input((64, 64, 3))
    x = biggan_residual_block(inputs, 128, 2, False) # (32, 32, 128)
    x = self_attention_non_local(x, ConvSN2D)
    x = biggan_residual_block(x, 256, 2, False)  # (16, 16, 256)
    x = biggan_residual_block(x, 512, 2, False)  # (8, 8, 512)
    x = biggan_residual_block(x, 512, 1, False)
    x = layers.ReLU()(x)
    if not use_patch_gan: # sum pooling
        x = layers.Lambda(lambda  z: tf.reduce_sum(z, axis=(1,2), keepdims=True))(x)
    x = ConvSN2D(1, 1)(x)
    if not use_patch_gan:
        x = layers.Reshape((1,))(x)
    model = tf.keras.models.Model(inputs, x)
    return model


