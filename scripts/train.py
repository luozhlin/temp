
"""
Train a diffusion model on spectral.
"""
import sys
sys.path.append("../")  # 没有这一步

import os
current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录
current_directory = os.path.dirname(current_script_path)
# 获取项目根目录
project_root = os.path.dirname(current_directory)

# 将项目根目录添加到 sys.path 中
sys.path.insert(0, project_root)

import argparse
import yaml
from guided_diffusion import dist_util, logger
from guided_diffusion.spectral_datasets import load_spectral_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

GPU_ID = 0

def main():
    args = create_argparser().parse_args()  # 解析命令行参数
    if args.model_config is not None:
        upgrade_by_config(args)

    dist_util.setup_dist()#GPU_ID)  # 初始化分布式训练环境
    logger.configure(dir=args.log_direct) # 配置日志记录器，将日志输出到命令行参数指定的目录

    logger.log("creating model and diffusion...")  #记录一条日志信息
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )  # model_and_diffusion_defaults() 函数返回一个包含模型和扩散过程默认参数的字典
       # model_and_diffusion_defaults().keys() 提取这个字典的所有键
       # args_to_dict 将 args 对象（命令行参数解析后的结果）转换为字典，只包含 model_and_diffusion_defaults() 中定义的键对应的参数
       # ** 是 Python 中的解包操作符，将字典中的键值对作为关键字参数传递给函数
    model.to(dist_util.dev()) # 将模型移动到指定的设备上
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)  # 采样

    logger.log("creating data loader...")
    data = load_spectral_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )  # 加载数据

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model_config=None,
        log_direct="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def upgrade_by_config(args):
    model_config = load_yaml(args.model_config)
    for k, v in model_config.items():
        setattr(args, k, v)


if __name__ == "__main__":
    main()
