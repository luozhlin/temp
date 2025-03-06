import sys

import os
# 获取当前脚本的绝对路径（/scripts/DiffUn_w_re.py）
current_script_path = os.path.abspath(__file__)
# 定位到项目根目录（DiffUn/）
project_root = os.path.dirname(os.path.dirname(current_script_path))
# 将根目录加入模块搜索路径
sys.path.insert(0, project_root)

import argparse

import blobfile as bf
import numpy as np
import torch as th
import yaml
import matplotlib.pyplot as plt

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from guided_diffusion.create import create_model_and_diffusion_RS

from unmixing_utils import UnmixingUtils, denoising_fn

from guided_diffusion import gaussian_diffusion_re as gdre

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR

# 加载数据，npz文件
def load_data(data_dir):
    data = np.load(data_dir)
    W = data["W"]
    H = data["H"]
    Y = data["Y"]
    X = data["X"]
    A = data["A"]  # spectral library
    sigma = data["sigma"]
    return W, H, X, sigma, Y, th.from_numpy(A).to(dist_util.dev()).float()

def main():
    # 读取命令行参数, 生成一个字典
    args = create_argparser().parse_args()  # 命令行读取参数
    if args.model_config is not None:  # yaml文件读取参数
        upgrade_by_config(args)

    # 设置设备
    GPU_ID = 1
    dist_util.setup_dist()

    filename = args.input_hsi  # 输入的hsi文件名字
    filename = filename.split("/")[-1]  # 提取文件名，不包含路径
    filename = filename.split(".")[:-1]  # 去除文件扩展名
    filename = ".".join(filename)  # 重新构建文件名
    logger.configure(dir=bf.join(args.save_dir, f"{filename}"))
    #logger.log(args)

    logger.log("creating model...")

    ###################### 创建endmember模型和diffusion, 应用DiffUn的设置
    endmember_path = "config/endmember_models_config.yaml"
    endmember_config = load_yaml(endmember_path)
    model_W, DiffUn = create_model_and_diffusion(
        **args_to_dict(endmember_config, model_and_diffusion_defaults().keys()) # 后续更改
    )
 
    # `create_model_and_diffusion` 是一个自定义函数，用于创建模型和扩散对象。
    # `args_to_dict(args, model_and_diffusion_defaults().keys())` 将命令行参数 `args` 转换为字典，只保留 `model_and_diffusion_defaults()` 中定义的键。
    # `**` 是字典解包操作符，将字典中的键值对作为关键字参数传递给 `create_model_and_diffusion` 函数。

    # 从指定路径加载模型的状态字典，并将其映射到 CPU 上
    model_path = endmember_config["model_path"]
    model_dict = model_W.state_dict()
    pretrained_dict = th.load(model_path, map_location='cpu')

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model_W.load_state_dict(model_dict)
       #dist_util.load_state_dict(model_path, map_location="cpu"))
    #)  # 预训练模型路径
    # `dist_util.load_state_dict` 是一个自定义函数，用于从指定路径加载模型的状态字典。
    # `map_location="cpu"` 表示将加载的模型参数映射到 CPU 上。
    # `model.load_state_dict` 方法将加载的状态字典应用到模型上，更新模型的参数。

    model_W.to(dist_util.dev())
    if args.use_fp16:
        model_W.convert_to_fp16()
    model_W.eval()

    ################################ 创建abundance模型和diffusion, 应用HIRdiff的设置
    abundance_path = "config/abundance_models_config.yaml"
    abundance_config = load_yaml(abundance_path)

    abundance_config["eta1"] *= 256*64
    abundance_config["eta2"] *= 8*64
    abundance_config["beta_linear_start"] = float(abundance_config["beta_linear_start"])
    abundance_config["beta_linear_end"] = float(abundance_config["beta_linear_end"])

    model_H, diff_H = create_model_and_diffusion_RS(abundance_config)    

    cks = th.load(abundance_config['resume_state'])
    import collections
    new_cks = collections.OrderedDict()
    for k, v in cks.items():
        newkey = k[11:] if k.startswith('denoise_fn.') else k
        new_cks[newkey] = v
    model_H.load_state_dict(new_cks, strict=False)
    model_H.to(dist_util.dev())

    if args.use_fp16:
        model_H.convert_to_fp16()
    model_H.eval()

    # 加载WDC数据
    # data = np.load(args.input_hsi)
    # X_t = data["X"]  # (256, 256, 124)
    # Y = data["Y"]       # (256, 256, 124)
    # X_t = X_t.reshape(256*256, 124)
    # Y = Y.reshape(256*256, 124)

    W_t, H_t, X_t, sigma, Y, _ = load_data(args.input_hsi)  # W_t (6, 224)  H_t (4096,6)  X_t (4096,224)  Y (4096,224)
    hyper_utils = UnmixingUtils(W_t.T, H_t)  # 真实的endmember和abundance

    logger.log("creating samples...")
    SRE = 0  # 重构误差
    R = 3  # diffusion的纬度

    # 核心函数
    sample, H = gdre.double_sample(
        model_W,
        DiffUn,
        model_H,
        diff_H,
        gt = X_t,
        R = R,  # 分解后的维度
        bands = 224,  # 原始维度
        input_hsi = Y.T,  # 输入观测值
        denoising_fn=denoising_fn(),  # 对输入数据进行高斯模糊去噪的滤波器函数
        progress=True,  # output the message
        cache_H=args.cache_H, #whether to cache to H. Setting to True for accelerating. Defaults to False.
    )
    # sample endmeber (3,224)
    # H abundance (3,4096)
    
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    Distance, meanDistance, P = hyper_utils.hyperSAD(sample.T)
    axes[0].plot(sample.T @ P.T)
    axes[1].plot(W_t.T)
    plt.savefig(bf.join(logger.get_dir(), f"W.png"), dpi=500)
    print("P",P.shape) # (6,3)
    print("H",H.shape) # (3,4096)
    rmse = hyper_utils.hyperRMSE(H, P)

    output_data = "L21: SAD ", str(meanDistance), str(Distance), "aRMSE: ", str(rmse)
    logger.log(output_data)
    SRE = 10*np.log10(np.sum((X_t)**2)/np.sum((H.T@sample - Y)**2))
    logger.log("SNR: ", SRE)
    SAM = np.mean(np.arccos(np.sum(X_t * (H.T@sample), axis=1)/np.linalg.norm(X_t, axis=1)/np.linalg.norm(H.T@sample, axis=1)))
    logger.log("SAM: ", SAM)
    print("PSNR:", PSNR(X_t, H.T@sample, data_range=1))
    print("SSIM", compare_ssim(X_t, H.T@sample, data_range=1))

    logger.log("sampling complete")
    np.savez(bf.join(logger.get_dir(), "result.npz"), H=H, W=sample)

    ## PSNR and SSIM

    # # Y_e = H.T @ sample
    # # Y_e = Y_e.reshape(256*256, 124)




def create_argparser():
    defaults = dict(
        clip_denoised=True,
        range_t=0,
        use_ddim=False,
        input_hsi="",
        model_path="",
        save_dir="",
        model_config=None,
        cache_H=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def upgrade_by_config(args):
    model_config = load_yaml(args.model_config) # 返回一个字典
    for k, v in model_config.items():
        setattr(args, k, v)  # setattr将k做属性名，v做属性值，添加到args对象中


if __name__ == "__main__":
    main()