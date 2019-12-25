import tensorflow as tf
import tensorflow.keras.layers as layers
from inpainting_layers import InstanceNorm2D, upsampling2d_tpu
from utils import distributed, Reduction

def generator_residual_block(inputs, ch, downsampling):
    # main path 
    x = InstanceNorm2D()(inputs)
    x = layers.ReLU()(x)
    if downsampling < -1:
        x = layers.Lambda(upsampling2d_tpu, arguments={"scale": abs(downsampling)})(x)

    x = layers.Conv2D(ch, 3, padding="same")(x)
    x = InstanceNorm2D()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(ch, 3, padding="same")(x)

    if downsampling > 1:
        x = layers.AveragePooling2D(downsampling)(x)
    # residual path
    if downsampling < -1:
        r = layers.Lambda(upsampling2d_tpu, arguments={"scale": abs(downsampling)})(inputs)
    else:
        r = inputs
    r = layers.Conv2D(ch, 1)(r)
    if downsampling > 1:
        r = layers.AveragePooling2D(downsampling)(r)
    return layers.Add()([x, r])

def noise_to_image_generator():
    inputs = layers.Input((128,))
    x = layers.Reshape((4, 4, 8))(inputs)
    x = layers.Conv2D(1024, 1)(x)  # (4, 4, 1024)
    x = generator_residual_block(x, 512, -2) # (8, 8, 512)
    x = generator_residual_block(x, 256, -2)  # (16, 16, 256)
    x = generator_residual_block(x, 128, -2)  # (32, 32, 256)
    x = generator_residual_block(x, 64, -2)  # (64, 64, 64)
    x = InstanceNorm2D()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(3, 3, padding="same", activation="tanh")(x)
    model = tf.keras.models.Model(inputs, x)
    return model

def image_to_image_generator():
    inputs = layers.Input((64, 64, 3))
    x = inputs
    for ch in [64, 128, 256]:
        x = layers.Conv2D(ch, 3, padding="same", strides=2)(x)
        x = InstanceNorm2D()(x)
        x = layers.ReLU()(x)
    for d in [1, 1, 2, 2]:
        x = layers.Conv2D(512, 3, padding="same", dilation_rate=d)(x)
        x = InstanceNorm2D()(x)
        x = layers.ReLU()(x) # (8, 8, 512)
    x = generator_residual_block(x, 256, -2)  # (16, 16, 256)
    x = generator_residual_block(x, 128, -2)  # (32, 32, 128)
    x = generator_residual_block(x, 64, -2)  # (64, 64, 64)
    x = InstanceNorm2D()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(3, 3, padding="same", activation="tanh")(x)
    model = tf.keras.models.Model(inputs, x)
    return model

def image_to_image_unet_generator():
    inputs = layers.Input((64, 64, 3))
    x = inputs
    encoders = []
    for ch in [64, 128, 256]:
        x = layers.Conv2D(ch, 3, padding="same", strides=2)(x)
        x = InstanceNorm2D()(x)
        x = layers.ReLU()(x)
        encoders.append(x)
    for d in [1, 1, 2, 2]:
        x = layers.Conv2D(512, 3, padding="same", dilation_rate=d)(x)
        x = InstanceNorm2D()(x)
        x = layers.ReLU()(x)  # (8, 8, 512)
    x = layers.Concatenate()([x, encoders[-1]])
    x = generator_residual_block(x, 256, -2)  # (16, 16, 256)
    x = layers.Concatenate()([x, encoders[-2]])
    x = generator_residual_block(x, 128, -2)  # (32, 32, 128)
    x = layers.Concatenate()([x, encoders[-3]])
    x = generator_residual_block(x, 64, -2)  # (64, 64, 64)
    x = InstanceNorm2D()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(3, 3, padding="same", activation="tanh")(x)
    model = tf.keras.models.Model(inputs, x)
    return model

def discriminator():
    inputs = layers.Input((64, 64, 3))
    x = inputs
    for i, (ch, stride) in enumerate(zip([128, 256, 512, 512], [2, 2, 2, 1])):
        x = layers.Conv2D(ch, 3, strides=stride, padding="same")(x)
        if i != 0:
            x = InstanceNorm2D()(x)
        x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(1, 1)(x)
    model = tf.keras.models.Model(inputs, x)
    return model