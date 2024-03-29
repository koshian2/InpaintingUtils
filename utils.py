from enum import Enum
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
import os

class Reduction(Enum):
    NONE = 0
    SUM = 1
    MEAN = 2
    CONCAT = 3

def distributed(*reduction_flags):
    def _decorator(fun):
        def per_replica_reduction(z, flag):
            if flag == Reduction.NONE:
                return z
            elif flag == Reduction.SUM:
                return strategy.reduce(tf.distribute.ReduceOp.SUM, z, axis=None)
            elif flag == Reduction.MEAN:
                return strategy.reduce(tf.distribute.ReduceOp.MEAN, z, axis=None)
            elif flag == Reduction.CONCAT:
                z_list = strategy.experimental_local_results(z)
                return tf.concat(z_list, axis=0)
            else:
                raise NotImplementedError()

        @tf.function
        def _decorated_fun(*args, **kwargs):
            fun_result = strategy.run(fun, args=args, kwargs=kwargs)
            if len(reduction_flags) == 0:
                assert fun_result is None
                return
            elif len(reduction_flags) == 1:
                assert type(fun_result) is not tuple and fun_result is not None
                return per_replica_reduction(fun_result, *reduction_flags)
            else:
                assert type(fun_result) is tuple
                return tuple((per_replica_reduction(fr, rf) for fr, rf in zip(fun_result, reduction_flags)))
        return _decorated_fun
    return _decorator

def make_grid(imgs, nrow, padding=0):
    assert imgs.ndim == 4 and nrow > 0
    batch, height, width, ch = imgs.shape
    n = nrow * (batch // nrow + np.sign(batch % nrow))
    ncol = n // nrow
    pad = np.zeros((n - batch, height, width, ch), imgs.dtype)
    x = np.concatenate([imgs, pad], axis=0)
    # border padding if required
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, padding), (0, padding), (0, 0)),
                   "constant", constant_values=(0, 0)) # 下と右だけにpaddingを入れる
        height += padding
        width += padding
    x = x.reshape(ncol, nrow, height, width, ch)
    x = x.transpose([0, 2, 1, 3, 4])  # (ncol, height, nrow, width, ch)
    x = x.reshape(height * ncol, width * nrow, ch)
    if padding > 0:
        x = x[:(height * ncol - padding),:(width * nrow - padding),:] # 右端と下端のpaddingを削除
    return x

def make_citation(data_seq, output_size):
    outs = []
    for data in data_seq:
        ## Citattion
        with Image.new("RGB", (output_size, output_size), color=(255, 255, 255)) as cite:
            id = data["asset"]["name"].split("_")[0]
            if os.name == "nt":
                font = ImageFont.truetype("arial.ttf")
            else:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 18)
            draw = ImageDraw.Draw(cite)
            draw.text((10, 50), "https://www.pixiv.net/artworks/", (32, 32, 32), font=font)
            draw.text((150, 75), id, (32, 32, 32), font=font)
            draw.text((10, 130), data["asset"]["name"], (32, 32, 32), font=font)
            outs.append(np.asarray(cite, np.uint8))
    return np.stack(outs, axis=0)
