"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import os

import numpy as np
import torch as th

from .nn import mean_flat, nan_mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from unmixing_utils import UnmixingUtils, cal_conditional_gradient_W, vca, cal_gradient
from pysptools.abundance_maps.amaps import FCLS
from functools import partial

import torch as th


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    args:
        schedule_name: the name of the beta schedule to use.
        num_diffusion_timesteps: the number of diffusion steps to use.

    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":  # 返回线性的beta线性数组
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = 1e-6 #/scale 
        beta_end = 1e-2 #/scale
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":  # cos数组设置
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL  # 是一个实例方法，用于判断当前枚举成员是否属于变分下界类型的损失函数


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000). 一个布尔值，如果为 True，则将浮点型的时间步传入模型，使其始终按照原论文中的比例（0 到 1000）进行缩放。
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)  # 它是一个一维数组，其中每个元素表示从开始到当前时间步长的 $\alpha$ 值的累积乘积。
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])  # 它表示前一个时间步长的 $\alpha$ 值的累积乘积
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)   # 它表示下一个时间步长的 $\alpha$ 值的累积乘积
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)  # 计算系数要用到参数

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # DDPM中x0的系数
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )  # DDPM中xt的系数

    def q_mean_variance(self, x_start, t): # 计算前向加噪要用到的均值, 方差
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):  # 前向加噪采样
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:  
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise  # _extract_into_tensor 函数用于从一维数组中提取对应时间步 t 的值，并将其广播到 x_start 的形状。
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,), f"excepted t.shape to be ({B},) but received {t.shape}"
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"], 'log_variance':out['log_variance']}

    def unmixing(
        self,
        model_W,
        R,  
        bands,
        input_hsi,
        gradient_type='diffun',
        band_mask=None,
        t0=200,
        measure_sigma=0,
        range_t=0,
        clip_denoised=True,
        progress=False,
        K=5,
        denoising_fn=None,
        cache_H=False,
    ):
        """The reverse process for hyperspectral unmixing.

        Args:
            model (nn.Module): spectral denoising model
            R (int): number of endmembers
            bands (int): number of bands
            input_hsi (np.ndarray): hyperspectral image with the shape of (bands, N)
            gradient_type (str, optional): type of gradient. Defaults to 'diffun'.
            band_mask (np.ndarray, optional): band mask to mask some noisy band. Defaults to None.
            t0 (int, optional): starting of t0. Defaults to 200.
            measure_sigma (float, optional): measure noise. Defaults to 0.
            range_t (int, optional): end of caculating the gradient. Defaults to 0.
            clip_denoised (bool, optional): clip the spectral to [0,1]. Defaults to True.
            progress (bool, optional): output the message. Defaults to False.
            K (int, optional): number of iteration. Defaults to 5.
            denoising_fn (function, optional): additional spectral denoising function. Defaults to None.
            cache_H (bool, optional): whether to cache to H. Setting to True for accelerating. Defaults to False.

        Returns:
            sample: Endmembers
            H: Abundance maps
        """
        device = next(model_W.parameters()).device # next() 函数用于从生成器中获取下一个元素。在这里，它获取了模型的第一个可训练参数。
        # partial 函数的作用是将一个函数的部分参数预先绑定，从而创建一个新的函数，这个新函数在调用时只需要提供剩余的参数
        measure_fn = partial(cal_conditional_gradient_W, type=gradient_type, __cache_H=cache_H)
        sre = -np.inf
        for repeat in range(K):
            W0, _, _, H0 = vca(input_hsi.T, R)  # 初始化endmember和abundance
            # 将 W0 转换为 PyTorch 张量并移动到指定设备上
            W0 = th.from_numpy(W0.T).to(device).float()[:, None]
            _samplek, _H = self.p_sample_loop(
                model_W,
                shape_W = (R, 1, bands),  # shape
                clip_denoised=clip_denoised,
                model_kwargs={"ref_img": th.from_numpy(input_hsi).to(device).float()},
                range_t=range_t,
                progress=False,
                measure_sigma=measure_sigma,
                measurement=measure_fn,  # cal_conditional_gradient_W
                W0=W0,
                t0=t0,
                mask=band_mask,
                denoised_fn=denoising_fn
            )  # 返回最后的endmember和abundance
            _H = _H.cpu().detach().numpy()[:, 0]
            _sample = _samplek.cpu().detach().numpy()[:, 0]
            _sample = (_sample + 1)/2
            # 计算measurement
            _sre = 10*np.log10(np.sum((input_hsi)**2)/np.sum((_H@_sample - input_hsi)**2))
            if progress:
                print(f"Repeat {repeat+1}/{K}: SRE = {_sre}")
            if _sre > sre:
                sre = _sre
                sample = _sample
        H = FCLS(input_hsi, sample)  # 最后根据endmember求abundance
        return sample, H

    def p_sample_loop(
        self,
        model_W,
        shape_W,
        noise_W=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        range_t=0,
        measurement=None,
        measure_sigma=0,
        W0=None,
        t0=None,
        mask=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).???
        :param noise_W: if specified, the noise from the endmember to sample.
                      Should be of the same shape as `shape`.
        :param noise_H: if specified, the noise from the abundance to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        log = {}
        for sample in self.p_sample_loop_progressive(
            model_W,
            shape_W,
            noise_W = noise_W,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            range_t=range_t,
            measure_sigma=measure_sigma,
            measurement=measurement,
            W0=W0,
            t0=t0,
            mask=mask
        ):
            final = sample

        return final["sample"], final["H"]

    def p_sample_loop_progressive(
        self,
        model_W,
        shape_W,
        noise_W=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        range_t=0,
        measurement=None,
        measure_sigma=0,
        W0=None,
        t0=None,
        mask=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model_W.parameters()).device

        assert isinstance(shape_W, (tuple, list))


        # endmember W
        if noise_W is not None:
            img_W = noise_W
        else:
            img_W = th.randn(*shape_W, device=device)
        
        # 采样时刻
        indices = list(range(self.num_timesteps))[::-1]
        
        # endmember起始时刻，如果有初始值
        if W0 is not None: 
            img_W = self.q_sample(W0 * 2 - 1, th.tensor([t0] * shape_W[0], device=device), img_W)

        # 是否记录过程
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        noise_var = measure_sigma ** 2

        for i in indices:
            if t0 is not None and i > t0:  # t0时刻才开始采样
                continue
            t = th.tensor([i] * shape_W[0], device=device)  # i时刻张量，用于和img能够对应元素处理

            with th.no_grad():  # 不计算梯度
                # endmember下一步采样, 返回W_{t-1}
                out_W = self.p_sample(
                    model_W,
                    img_W,
                    t,
                    clip_denoised=
                    clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )

              
                # cal_conditional_gradient_W计算梯度g1和abundance，要修改
                if measurement is not None:
                    g1, g2, _ = measurement(out_W["pred_xstart"], model_kwargs["ref_img"],
                                                                self.alphas_cumprod[i], 1-self.betas[i], noise_var, i, mask)
                else:
                    g1 = 0
                    H = None

                # 计算出各自的loss，然后加上各自的梯度

                if i > range_t: # range_t =0 所以每一步都加这个梯度
                    if mask is not None:
                        out_W["sample"][:, :, ~mask] += g1
                    else:
                        out_W["sample"] += g1  # 直接加梯度
                    
                
                # ？？？ 要修改H的计算
                # out["H"] = H
                yield out_W
                img_W = out_W["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"], 'log_variance':out['log_variance']}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample

        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"], 'log_variance':out['log_variance']}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, target=None, pad=False):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            if target is not None:
                x_start = target
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            if pad:
                assert False, "NO SUPPORT NOW"
                target = target[..., 6:-6]
                model_output = model_output[..., 6:-6]
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

######################################## Main function ###########################################
# 同步采样函数
def double_sample(
        model_W, 
        diffusion_W, 
        model_H, 
        diffusion_H, 
        R, 
        bands,
        input_hsi,
        gradient_type="denoise",
        band_mask=None,
        t0 = 100,
        measure_sigma=0,
        range_t=0,
        clip_denoised=True,
        progress = True,
        K=5,
        denoising_fn=None,
        cache_H=False, 
        ):
    '''
    Sampling function to sample abundance and endmembers from their own diffusion models.
    Input:
        model_W: endmember model - unet
        diffusion_W: endmember diffusion - sampling
        model_H: abundance diffusion model
        diffusion_H: abundance diffusion -sampling
        R: number of endmembers
        bands: number of bands, L
        input_hsi: input hyperspectral image, the shape is (L, N) where L is bands and N is pixels
        gradient_type: gradient type to use
        band_mask: band mask to use
        t0: start time step
        measure_sigma: measurement noise
        range_t: range of time steps to use
        clip_denoised: whether to clip denoised samples
        progress: whether to print progress
        K: number of repeats
        denoising_fn: denoising function to use
        cache_H: whether to cache H

    Output:
        W: endmembers, the shape is (R, bands)
        H: abundance, the shape is (R, N)
    '''
    device = next(model_W.parameters()).device 
    measure_fn = partial(cal_gradient, type=gradient_type, __cache_H=cache_H)  # update parameters of gradient function
    sre = -np.inf

    for repeat in range(1):
        W0, _, _= vca(input_hsi, R)   # Initialize W0, the shape is (L, R)
        import unmixing_utils
        H0 = unmixing_utils.solve_H(input_hsi.T, W0.T, 0)  # Initialize H0, the shape is (N, R)
        H0 = H0.numpy()
        Nn, Rr  = H0.shape

        W0 = th.from_numpy(W0.T).to(device).float()[:, None]  # add a dimension to W0, the shape is (R, 1, L)
        # H0 = th.from_numpy(H0.T).to(device).float()[:, None]  # 可能可以不用增加纬度

        _sampleW, _sampleH =  double_sample_loop(
            model_W,
            diffusion_W,
            model_H,
            diffusion_H,
            #shape_W = (R,1,bands),
            clip_denoised=clip_denoised,
            model_kwargs={"ref_img":th.from_numpy(input_hsi).to(device).float()},
            range_t=range_t,
            measure_sigma=measure_sigma,
            measurement=measure_fn,
            W0=W0,
            H0=H0,
            t0=t0,
            mask=band_mask,
            denoised_fn= denoising_fn,
        )
        _H = _sampleH.cpu().detach().numpy()
        _H = (_H+1)/2

        _H = _H.reshape(Rr,Nn)  ###?????? 待修改

        _W = _sampleW.cpu().detach().numpy()[:,0]
        _W = (_W+1)/2
      #
        _sre = 10*np.log10(np.sum((input_hsi)**2)/np.sum((_W.T@_H - input_hsi)**2))

        if progress:
            print(f"Repeat {repeat+1}/{K}: SRE = {_sre}")
        if _sre > sre:
            sre = _sre
            W = _W
            H = _H
    return W, H 

def double_sample_loop(
        model_W,
        diffusion_W,
        model_H,
        diffusion_H,
        noise_W = None,
        noise_H = None,
        clip_denoised=True,
        cond_fn=None,
        model_kwargs=None,
        device = None,
        progress=True,
        range_t=0,
        measurement=None,
        measure_sigma=0,
        W0=None,
        H0=None,
        t0=None,
        mask=None,
        denoised_fn=None,
):
    '''
    Sampling function to sample abundance and endmembers from their own diffusion models at the end.

    Output:
        W: endmembers, the shape is (R, 1, bands)
        H: abundance, the shape is (R, 1, N)
    '''
    finalW = None
    finalH = None
    log = {}

    for sampleW,sample_H in double_sample_loop_progressive(
        model_W=model_W,
        diffusion_W=diffusion_W,
        model_H=model_H,
        diffusion_H=diffusion_H,
        noise_W=noise_W,
        noise_H = noise_H,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        cond_fn=cond_fn,
        model_kwargs=model_kwargs,
        device =device,
        progress=progress,
        range_t=range_t,
        measurement=measurement,
        measure_sigma=measure_sigma,
        W0=W0,
        H0=H0,
        t0=t0,
        mask=mask,
    ):
        finalW = sampleW
        finalH = sample_H

    return finalW["sample"], finalH["sample"]





def double_sample_loop_progressive(
    model_W,
    diffusion_W,
    model_H,
    diffusion_H,
    noise_W=None,
    noise_H=None,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=True,
    range_t=0,
    measurement=None,
    measure_sigma=0,
    W0=None,
    H0=None,
    t0=None,
    mask=None,
):
    '''
    The reverse process of diffusion.
    '''
    if device is None:
        device = next(model_W.parameters()).device
    assert isinstance(W0.shape, (tuple, list))

    # Initialize the start noise 
    if noise_W is not None:
        img_W = noise_W
    else:
        img_W = th.randn(*W0.shape, device=device)
    
    N, Rr = H0.shape
    H0 = H0.reshape(1,Rr,int(np.sqrt(N)),int(np.sqrt(N)))

    if noise_H is not None:
        img_H = noise_H
    else:
        img_H = th.randn(*H0.shape, device=device)

    indices = list(range(diffusion_W.num_timesteps))[::-1]  
    indices_next = indices[1:] + [-1]

    if progress:  # 进度条
        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    noise_var = measure_sigma**2  # 与梯度计算有关，待修改？？？？？？？？？
    for iteration, (i,j) in enumerate(zip(indices, indices_next)):
        tw = th.tensor([i] * W0.shape[0], device=device)  # endmember diffusion noise schedule, R - batch size

        t = th.tensor([i] * 1, device=device)  # abundance diffusion noise schedule, B - batch size
        t_next = th.tensor([j] * 1, device=device) 

        with th.no_grad():
            '''
            endmember diffusion, img_W shape is (R, 1, L).
            Output:
            '''
            out_W = diffusion_W.p_sample(
                    model_W,
                    img_W,
                    tw,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,  # denoising function only for this 
                    model_kwargs=model_kwargs,
            )
            '''
            abundance diffusion, img_H shape is (B, R, H, W) - batch size, endmember nums, height, width.
            '''
            # split img_H 
            out_H_list = list()
            slice = [[0,1,2],[3,4,5]]

            for s in slice:
                out_H = diffusion_H.p_sample(
                    model_H,
                    img_H[:, s,:,:],  # 形状为(Bb, Cc , Hh, Ww ), B是batch_size
                    t,
                    t_next,
                    clip_denoised=True,
                    denoised_fn=None,  
                )
                out_H_list.append(out_H)
            out_H["xhat"] = th.cat([out_H_list[0]["xhat"], out_H_list[1]["xhat"]], dim=1)
            out_H["sample"] = th.cat([out_H_list[0]["sample"], out_H_list[1]["sample"]], dim=1)
            # Calulation gradient, g1 for endmember, g2 for abundance, the shape should be identical
            if measurement is not None:
                g1, g2, _ = measurement(out_W["pred_xstart"], 
                                        out_H["xhat"], 
                                        model_kwargs["ref_img"],
                                        diffusion_W.alphas_cumprod[i], 
                                        1-diffusion_W.betas[i], 
                                        noise_var, 
                                        i,
                                        mask=None,
                                        )
            else:
                g1, g2 = 0,0

            out_W["sample"] += g1
            out_H["sample"] += g2
        
        # for the next step, sample = xt_next, pred_xtart = x0, xhat is x0 after clip.
        yield  out_W, out_H
        img_W = out_W["sample"]  # out_W has other keys, but we don't them.
        img_H = out_H["sample"]  
            
  