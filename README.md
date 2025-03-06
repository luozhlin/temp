
DiffUn的运行命令如下：

```
python scripts/DiffUn.py --input_hsi data/A4_P1500.npz --save_dir "results/unmixing/" --model_config models/A4_P1500/model_config.yaml --in_channels 1 --range_t 0 --diffusion_steps 1000 --rescale_timesteps True --cache_H True
```

Ours的运行命令如下：

```
python scripts/DiffUn_w_re.py --input_hsi data/A4_P1500.npz --save_dir "results/unmixing/" --in_channels 1 --range_t 0 --diffusion_steps 1000 --rescale_timesteps True --cache_H True
```

主要修改点：
1. DiffUn_w_re.py中配置读取，新增输出PSNR和SSIM，配置读取后续还需精简
2. gaussian_diffusion_re.py中最下方新增使用的主要函数doublediffusion()
3. rsfac_grad_gaussian_diffusion.py中新增q_sample()函数，用于abundance diffusion的向前加噪初始化
4. unmixing_utils.py()中新增calgradient，用于计算梯度
5. guided_diffusion中新增的文件是HIRdiff中会用到的

现在只是程序能够正常运行，后续两个的diffusion功能可以整合，精简文件。如果要跑其他数据集，注意数据的读取方式和形状大小，会涉及到模型unet的一些修改，这里只以A4_P1500为例。

模型及数据在百度云下载
通过网盘分享的文件：DiffUn
链接: https://pan.baidu.com/s/1z2qj9UcXMQrk1LGw1yttGw 提取码: u6xn 
--来自百度网盘超级会员v7的分享

* A4_P1500.npz放在data中
* model010000.pt放在models/A4_P1500中
* I1900000_E97_gen.pth放在checkpoints中
WDC数据如要使用，需要修改代码

后续修改点
1. 梯度的计算还能优化，原方法是自己计算，可以尝试自动求导
2. 代码结构优化
3. 解决纬度R过小的问题
