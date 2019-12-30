import os
import tensorflow as tf
import numpy as np
from scipy import linalg

def get_gpu_strategy():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"],
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    return strategy

def load_evaluation_dataset(batch_size, broadcast_mask, scale_tanh, mosaic_flag):
    def preprocess(inputs):
        if broadcast_mask and inputs.shape[-1] == 1:
            x = tf.broadcast_to(inputs, (*inputs.shape[:-1], 3))
        else:
            x = inputs
        x = tf.image.convert_image_dtype(x, tf.float32)
        if scale_tanh and inputs.shape[-1] == 3:            
            x = (x * 2) - 1.0 # [0,1] -> [-1,1]
        return x

    if mosaic_flag:
        val = np.load("../data/test_mosaic.npz", allow_pickle=True)
    else:
        val = np.load("../data/test_whiten.npz", allow_pickle=True)
    valset = tf.data.Dataset.from_tensor_slices((val["image"], val["mosaic"], val["mask"]))
    valset = valset.map(
        lambda *args: [preprocess(a) for a in args]
    ).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)    
    return valset, val["json"]

def fid(inception_a, inception_b):
    mu1 = np.mean(inception_a, axis=0)
    mu2 = np.mean(inception_b, axis=0)
    sigma1 = np.cov(inception_a, rowvar=False)
    sigma2 = np.cov(inception_b, rowvar=False)
    diff = mu1 - mu2

    # https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)    


def concatenate_batch_outputs(outputs, json, img_citation, contains_image, output_path):
    for i in range(len(outputs)):
        outputs[i] = np.concatenate(outputs[i], axis=0)[:2542]

    # マスク比率
    valid_sum = np.mean(np.sum(outputs[3], axis=(1, 2)), axis=-1)
    mask_ratio = 1.0 - valid_sum / outputs[3].shape[1] / outputs[3].shape[2]

    # float32 -> uint8
    for i in range(4):
        if scale_tanh and i < 3:                
            outputs[i] = (outputs[i] + 1.0) * 127.5
        else:
            outputs[i] = outputs[i] * 255.0
        outputs[i] = outputs[i].astype(np.uint8)

    result = {}
    result_keys = ["img_gt", "img_mosaic", "img_comp", "img_mask",
                    "inception_gt", "inception_mosaic", "inception_comp",
                    "mosaic_psnr", "mosaic_msssim", "comp_psnr", "comp_msssim"]
    for i in range(len(outputs)):
        result[result_keys[i]] = outputs[i]
    result["json"] = json
    result["mask_ratio"] = mask_ratio

    # fid
    result["mosaic_fid"] = fid(result["inception_gt"], result["inception_mosaic"])
    result["comp_fid"] = fid(result["inception_gt"], result["inception_comp"])

    # 不要なら推論画像以外を消す（容量対策）
    if not contains_image:
        result["img_gt"], result["img_mosaic"], result["img_comp"] = None, None, None
    else:
        result_branch = {"img_gt": result["img_gt"],
                         "img_mosaic": result["img_mosaic"],
                         "img_comp": result["img_comp"],
                         "img_citation": result["img_citation"]}                    

    # 確認用
    print("comp_psnr =", np.mean(result["comp_psnr"]), "mosaic_psnr =", np.mean(result["mosaic_psnr"]))
    print("comp_msssim =", np.mean(result["comp_msssim"]), "mosaic_msssim =", np.mean(result["mosaic_msssim"]))
    print("comp_fid =", np.mean(result["comp_fid"]), "mosaic_fid =", np.mean(result["mosaic_fid"]))

    base_dir = os.path.dirname(output_path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    np.savez_compressed(output_path, result)
    if contains_image:
        np.savez_compressed(base_dir+"/images.npz", result_branch)
