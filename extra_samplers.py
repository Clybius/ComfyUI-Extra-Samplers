import math

from scipy import integrate
import torch
from torch import nn
import torchsde
from tqdm.auto import trange, tqdm

import comfy.sample

import k_diffusion.sampling
from k_diffusion.sampling import BrownianTreeNoiseSampler, PIDStepSizeController, get_ancestral_step, to_d, default_noise_sampler
import random

# The following function adds the samplers during initialization, in __init__.py
def add_samplers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    for sampler in extra_samplers: #getattr(self, "sample_{}".format(extra_samplers))
        if sampler not in KSampler.SAMPLERS:
            try:
                idx = KSampler.SAMPLERS.index("uni_pc_bh2") # Last item in the samplers list
                KSampler.SAMPLERS.insert(idx+1, sampler) # Add our custom samplers
                setattr(k_diffusion_sampling, "sample_{}".format(sampler), extra_samplers[sampler])
                import importlib
                importlib.reload(k_diffusion_sampling)
            except ValueError as err:
                pass

# The following function adds the samplers during initialization, in __init__.py
def add_schedulers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    for scheduler in extra_schedulers: #getattr(self, "sample_{}".format(extra_samplers))
        if scheduler not in KSampler.SCHEDULERS:
            try:
                idx = KSampler.SCHEDULERS.index("ddim_uniform") # Last item in the samplers list
                KSampler.SCHEDULERS.insert(idx+1, scheduler) # Add our custom samplers
                setattr(k_diffusion_sampling, "get_sigmas_{}".format(scheduler), extra_schedulers[scheduler])
                import importlib
                importlib.reload(k_diffusion_sampling)
            except ValueError as err:
                pass

# Noise samplers
from torch import Generator, Tensor, lerp
from torch.nn.functional import unfold
from typing import Callable, Tuple
from math import pi

def get_positions(block_shape: Tuple[int, int]) -> Tensor:
    """
    Generate position tensor.

    Arguments:
        block_shape -- (height, width) of position tensor

    Returns:
        position vector shaped (1, height, width, 1, 1, 2)
    """
    bh, bw = block_shape
    positions = torch.stack(
        torch.meshgrid(
            [(torch.arange(b) + 0.5) / b for b in (bw, bh)],
            indexing="xy",
        ),
        -1,
    ).view(1, bh, bw, 1, 1, 2)
    return positions


def unfold_grid(vectors: Tensor) -> Tensor:
    """
    Unfold vector grid to batched vectors.

    Arguments:
        vectors -- grid vectors

    Returns:
        batched grid vectors
    """
    batch_size, _, gpy, gpx = vectors.shape
    return (
        unfold(vectors, (2, 2))
        .view(batch_size, 2, 4, -1)
        .permute(0, 2, 3, 1)
        .view(batch_size, 4, gpy - 1, gpx - 1, 2)
    )


def smooth_step(t: Tensor) -> Tensor:
    """
    Smooth step function [0, 1] -> [0, 1].

    Arguments:
        t -- input values (any shape)

    Returns:
        output values (same shape as input values)
    """
    return t * t * (3.0 - 2.0 * t)


def perlin_noise_tensor(
    vectors: Tensor, positions: Tensor, step: Callable = None
) -> Tensor:
    """
    Generate perlin noise from batched vectors and positions.

    Arguments:
        vectors -- batched grid vectors shaped (batch_size, 4, grid_height, grid_width, 2)
        positions -- batched grid positions shaped (batch_size or 1, block_height, block_width, grid_height or 1, grid_width or 1, 2)

    Keyword Arguments:
        step -- smooth step function [0, 1] -> [0, 1] (default: `smooth_step`)

    Raises:
        Exception: if position and vector shapes do not match

    Returns:
        (batch_size, block_height * grid_height, block_width * grid_width)
    """
    if step is None:
        step = smooth_step

    batch_size = vectors.shape[0]
    # grid height, grid width
    gh, gw = vectors.shape[2:4]
    # block height, block width
    bh, bw = positions.shape[1:3]

    for i in range(2):
        if positions.shape[i + 3] not in (1, vectors.shape[i + 2]):
            raise Exception(
                f"Blocks shapes do not match: vectors ({vectors.shape[1]}, {vectors.shape[2]}), positions {gh}, {gw})"
            )

    if positions.shape[0] not in (1, batch_size):
        raise Exception(
            f"Batch sizes do not match: vectors ({vectors.shape[0]}), positions ({positions.shape[0]})"
        )

    vectors = vectors.view(batch_size, 4, 1, gh * gw, 2)
    positions = positions.view(positions.shape[0], bh * bw, -1, 2)

    step_x = step(positions[..., 0])
    step_y = step(positions[..., 1])

    row0 = lerp(
        (vectors[:, 0] * positions).sum(dim=-1),
        (vectors[:, 1] * (positions - positions.new_tensor((1, 0)))).sum(dim=-1),
        step_x,
    )
    row1 = lerp(
        (vectors[:, 2] * (positions - positions.new_tensor((0, 1)))).sum(dim=-1),
        (vectors[:, 3] * (positions - positions.new_tensor((1, 1)))).sum(dim=-1),
        step_x,
    )
    noise = lerp(row0, row1, step_y)
    return (
        noise.view(
            batch_size,
            bh,
            bw,
            gh,
            gw,
        )
        .permute(0, 3, 1, 4, 2)
        .reshape(batch_size, gh * bh, gw * bw)
    )


def perlin_noise(
    grid_shape: Tuple[int, int],
    out_shape: Tuple[int, int],
    batch_size: int = 1,
    generator: Generator = None,
    *args,
    **kwargs,
) -> Tensor:
    """
    Generate perlin noise with given shape. `*args` and `**kwargs` are forwarded to `Tensor` creation.

    Arguments:
        grid_shape -- Shape of grid (height, width).
        out_shape -- Shape of output noise image (height, width).

    Keyword Arguments:
        batch_size -- (default: {1})
        generator -- random generator used for grid vectors (default: {None})

    Raises:
        Exception: if grid and out shapes do not match

    Returns:
        Noise image shaped (batch_size, height, width)
    """
    # grid height and width
    gh, gw = grid_shape
    # output height and width
    oh, ow = out_shape
    # block height and width
    bh, bw = oh // gh, ow // gw

    if oh != bh * gh:
        raise Exception(f"Output height {oh} must be divisible by grid height {gh}")
    if ow != bw * gw != 0:
        raise Exception(f"Output width {ow} must be divisible by grid width {gw}")

    angle = torch.empty(
        [batch_size] + [s + 1 for s in grid_shape], *args, **kwargs
    ).uniform_(to=2.0 * pi, generator=generator)
    # random vectors on grid points
    vectors = unfold_grid(torch.stack((torch.cos(angle), torch.sin(angle)), dim=1))
    # positions inside grid cells [0, 1)
    positions = get_positions((bh, bw)).to(vectors)
    return perlin_noise_tensor(vectors, positions).squeeze(0)

def rand_perlin_like(x):
    noise = torch.randn_like(x) / 2.0
    noise_size_H = noise.size(dim=2)
    noise_size_W = noise.size(dim=3)
    perlin = None
    for i in range(2):
        noise += perlin_noise((noise_size_H, noise_size_W), (noise_size_H, noise_size_W), batch_size=4).to(x.device)
    #noise += perlin
    #print(noise)
    return noise / noise.std()

def uniform_noise_sampler(x): # Even distribution, seemingly produces more information in non-subject areas than the normal (gaussian) noise sampler
    return lambda sigma, sigma_next: (torch.rand_like(x) - 0.5) * 2 * 1.73

from torch.distributions import StudentT
def studentt_noise_sampler(x): # Produces more subject-focused outputs due to distribution, unsure if this works
    noise = StudentT(loc=0, scale=0.2, df=1).rsample(x.size())
    #noise *= 2 / (torch.max(torch.abs(noise)) + 1e-8)
    s: FloatTensor = torch.quantile(
        noise.flatten(start_dim=1).abs(),
        0.75,
        dim = -1
    )
    #s.clamp_(min = 1.)
    s = s.reshape(*s.shape, 1, 1, 1)
    noise = noise.clamp(-s, s)
    noise = torch.copysign(torch.pow(torch.abs(noise), 0.5), noise)
    print(s)
    return lambda sigma, sigma_next: noise.to(x.device) / (7/3)

def highres_pyramid_noise_like(x, discount=0.7):
    b, c, h, w = x.shape # EDIT: w and h get over-written, rename for a different variant!
    orig_h = h
    orig_w = w
    u = torch.nn.Upsample(size=(orig_h, orig_w), mode='bilinear')
    noise = (torch.rand_like(x) - 0.5) * 2 * 1.73 # Start with scaled uniform noise
    for i in range(4):
        r = random.random()*2+2 # Rather than always going 2x, 
        h, w = min(orig_h*15, int(h*(r**i))), min(orig_w*15, int(w*(r**i)))
        noise += u(torch.randn(b, c, h, w).to(x)) * discount**i
        if h>=orig_h*15 or w>=orig_w*15: break # Lowest resolution is 1x1
    return noise/noise.std() # Scaled back to roughly unit variance

def green_noise_like(x):
    noise = torch.randn_like(x)
    width = noise.size(dim=2)
    height = noise.size(dim=3)
    scale = 1.0 / (width * height)
    fy = torch.fft.fftfreq(width, device=x.device)[:, None] ** 2
    fx = torch.fft.fftfreq(height, device=x.device) ** 2
    f = fy + fx
    power = torch.sqrt(f)
    power[0, 0] = 1
    noise = torch.fft.ifft2(torch.fft.fft2(noise) / torch.sqrt(power))
    noise *= scale / noise.std()
    noise = torch.real(noise).to(x.device)
    return noise / noise.std()

def green_noise_sampler(x): # This doesn't work properly right now
    width = x.size(dim=2)
    height = x.size(dim=3)
    noise = torch.randn(width, height)
    #scale = 1.0 / (width * height)
    fy = torch.fft.fftfreq(width)[:, None] ** 2
    fx = torch.fft.fftfreq(height) ** 2
    f = fy + fx
    power = torch.sqrt(f)
    power[0, 0] = 1
    noise = torch.fft.ifft2(torch.fft.fft2(noise) / torch.sqrt(power))
    #noise *= scale / noise.std()
    noise = torch.real(noise).to(x.device)
    mean = torch.mean(noise)
    std = torch.std(noise)

    noise.sub_(mean).div_(std)
    print(noise)
    return lambda sigma, sigma_next: noise

def power_noise_sampler(tensor, alpha=2, k=1): # This doesn't work properly right now
    """Generate 1/f noise for a given tensor.

    Args:
        tensor: The tensor to add noise to.
        alpha: The parameter that determines the slope of the spectrum.
        k: A constant.

    Returns:
        A tensor with the same shape as `tensor` containing 1/f noise.
    """
    tensor = torch.randn_like(tensor)
    fft = torch.fft.fft2(tensor)
    freq = torch.arange(1, len(fft) + 1, dtype=torch.float)
    spectral_density = k / freq**alpha
    noise = torch.rand(tensor.shape) * spectral_density
    mean = torch.mean(noise, dim=(-2, -1), keepdim=True).to(tensor.device)
    std = torch.std(noise, dim=(-2, -1), keepdim=True).to(tensor.device)
    noise = noise.to(tensor.device).sub_(mean).div_(std)
    variance = torch.var(noise, dim=(-2, -1), keepdim=True)
    print(variance)
    return lambda sigma, sigma_next: noise / 3

# Below this point are extra samplers
@torch.no_grad()
def sample_clyb_4m_sde_momentumized(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1., noise_sampler=None, momentum=0.0):
    """DPM-Solver++(3M) SDE, modified with an extra SDE, and momentumized in both the SDE and ODE(?). 'its a first' - Clybius 2023
    The expression for d1 is derived from the extrapolation formula given in the paper “Diffusion Monte Carlo with stochastic Hamiltonians” by M. Foulkes, L. Mitas, R. Needs, and G. Rajagopal. The formula is given as follows:
    d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
    (if this is an incorrect citing, we blame Google's Bard and OpenAI's ChatGPT for this and NOT me :^) )
     
    where d1_0, d1_1, and d1_2 are defined as follows:
    d1_0 = (denoised - denoised_1) / r2
    d1_1 = (denoised_1 - denoised_2) / r1
    d1_2 = (denoised_2 - denoised_3) / r0

    The variables r0, r1, and r2 are defined as follows:
    r0 = h_3 / h_2
    r1 = h_2 / h
    r2 = h / h_1
    """

    def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
        return momentum_vel

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = rand_perlin_like(x) if noise_sampler is None else noise_sampler

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2, denoised_3 = None, None, None
    h_1, h_2, h_3 = None, None, None
    vel, vel_sde = None, None
    for i in trange(len(sigmas) - 1, disable=disable):
        time = sigmas[i] / sigma_max
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)
            x_diff = momentum_func((-h_eta).expm1().neg() * denoised, vel, time)
            vel = x_diff
            x = torch.exp(-h_eta) * x + vel

            if h_3 is not None:
                r0 = h_3 / h_2
                r1 = h_2 / h
                r2 = h / h_1
                d1_0 = (denoised - denoised_1) / r2
                d1_1 = (denoised_1 - denoised_2) / r1
                d1_2 = (denoised_2 - denoised_3) / r0
                d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
                d2 = (d1_0 - d1_1) / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r2 + r1) * (r0 + r1))
                phi_3 = h_eta.neg().expm1() / h_eta + 1
                phi_4 = phi_3 / h_eta - 0.5
                sde_diff = momentum_func(phi_3 * d1 - phi_4 * d2, vel_sde, time)
                vel_sde = sde_diff
                x = x + vel_sde
            elif h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                sde_diff = momentum_func(phi_2 * d1 - phi_3 * d2, vel_sde, time)
                vel_sde = sde_diff
                x = x + vel_sde
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                sde_diff = momentum_func(phi_2 * d, vel_sde, time)
                vel_sde = sde_diff
                x = x + vel_sde

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

            denoised_1, denoised_2, denoised_3 = denoised, denoised_1, denoised_2 
            h_1, h_2, h_3 = h, h_1, h_2
    return x

# Kat's Truncated Taylor Method sampler, by Katherine Crowson
def sample_ttm_jvp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Second order truncated Taylor method (torch.func.jvp() version)."""

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    model_fn = lambda x, sigma: model(x, sigma * s_in, **extra_args)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model_fn(x, sigmas[i])

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # 2nd order truncated Taylor method
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            eps = to_d(x, sigmas[i], denoised)
            _, denoised_prime = torch.func.jvp(model_fn, (x, sigmas[i]), (eps * -sigmas[i], -sigmas[i]))

            phi_1 = -torch.expm1(-h_eta)
            #phi_2 = torch.expm1(-h_eta) + h_eta
            phi_2 = torch.expm1(-h) + h # seems to work better with eta > 0
            x = torch.exp(-h_eta) * x + phi_1 * denoised + phi_2 * denoised_prime

            if eta:
                phi_1_noise = torch.sqrt(-torch.expm1(-2 * h * eta))
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * phi_1_noise * s_noise

    return x

# Many thanks to Kat + Birch-San for this wonderful sampler implementation! https://github.com/Birch-san/sdxl-play/commits/res/
from .other_samplers.refined_exp_solver import sample_refined_exp_s
def sample_res_solver(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, denoise_to_zero=True, simple_phi_calc=False, c2=0.5, ita=torch.Tensor((0.25,)), momentum=0.0):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    match noise_sampler:
        case "brownian":
            noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=False)
        case "gaussian":
            noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
        case "uniform":
            noise_sampler = lambda sigma, sigma_next: (torch.rand_like(x) - 0.5) * 2 * 1.73
        case "highres-pyramid":
            noise_sampler = lambda sigma, sigma_next: highres_pyramid_noise_like(x)
        case "perlin":
            noise_sampler = lambda sigma, sigma_next: rand_perlin_like(x)
        case _:
            noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
    return sample_refined_exp_s(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, noise_sampler=noise_sampler, denoise_to_zero=denoise_to_zero, simple_phi_calc=simple_phi_calc, c2=c2, ita=ita, momentum=momentum)

@torch.no_grad()
def sample_dpmpp_dualsde_momentum(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1/2, momentum=0.0):
    """DPM-Solver++ (Stochastic with Momentum). Personal modified sampler by Clybius"""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = rand_perlin_like(x) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    denoisedsde_1, denoisedsde_2, denoisedsde_3 = None, None, None # new line
    h_1, h_2, h_3 = None, None, None # new line

    def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
        return momentum_vel

    vel = None
    vel_2 = None
    vel_sde = None
    for i in trange(len(sigmas) - 1, disable=disable):
        time = sigmas[i] / sigma_max
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            h_eta = h * (eta + 1)
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            diff_2 = momentum_func((t - s_).expm1() * denoised, vel_2, time)
            vel_2 = diff_2
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - diff_2
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            diff = momentum_func((t - t_next_).expm1() * denoised_d, vel, time)
            vel = diff
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - diff

            if h_3 is not None:
                r0 = h_3 / h_2
                r1 = h_2 / h
                r2 = h / h_1
                d1_0 = (denoised_d - denoisedsde_1) / r2
                d1_1 = (denoisedsde_1 - denoisedsde_2) / r1
                d1_2 = (denoisedsde_2 - denoisedsde_3) / r0
                d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
                d2 = (d1_0 - d1_1) / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r2 + r1) * (r0 + r1))
                phi_3 = h_eta.neg().expm1() / h_eta + 1
                phi_4 = phi_3 / h_eta - 0.5
                diff = momentum_func(phi_3 * d1 - phi_4 * d2, vel_sde, time)
                vel_sde = diff
                x = x + diff
            elif h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised_d - denoisedsde_1) / r0
                d1_1 = (denoisedsde_1 - denoisedsde_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                diff = momentum_func(phi_2 * d1 - phi_3 * d2, vel_sde, time)
                vel_sde = diff
                x = x + diff
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised_d - denoisedsde_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                diff = momentum_func(phi_2 * d, vel_sde, time)
                vel_sde = diff
                x = x + diff

            if eta:
                x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
            #if 'denoised_d' in locals():
            denoisedsde_1, denoisedsde_2, denoisedsde_3 = denoised_d, denoisedsde_1, denoisedsde_2 # new line
            #if 'h' in locals():
            h_1, h_2, h_3 = h, h_1, h_2
    return x

def sample_dpmpp_dualsdemomentum(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1/2, momentum=0.0):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    match noise_sampler:
        case "brownian":
            noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=False)
        case "gaussian":
            noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
        case "uniform":
            noise_sampler = lambda sigma, sigma_next: (torch.rand_like(x) - 0.5) * 2 * 1.73
        case "perlin":
            noise_sampler = lambda sigma, sigma_next: rand_perlin_like(x)
        case _:
            noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
    return sample_dpmpp_dualsde_momentum(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, r=r, momentum=momentum)

from .other_samplers.sample_ttm import sample_ttm_jvp
def sample_ttmcustom(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    match noise_sampler:
        case "gaussian":
            noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
        case "uniform":
            noise_sampler = lambda sigma, sigma_next: (torch.rand_like(x) - 0.5) * 2 * 1.73
        case "brownian":
            noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=False)
        case _:
            noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
    return sample_ttm_jvp(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler)

from k_diffusion.sampling import sample_lcm
def sample_lcmcustom(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    match noise_sampler:
        case "gaussian":
            noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
        case "uniform":
            noise_sampler = lambda sigma, sigma_next: (torch.rand_like(x) - 0.5) * 2 * 1.73
        case "brownian":
            noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=False)
        case _:
            noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
    return sample_lcm(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, noise_sampler=noise_sampler)

def sample_clyb_4m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler="brownian", momentum=0.0):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    match noise_sampler:
        case "brownian":
            noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=False)
        case "gaussian":
            noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
        case "uniform":
            noise_sampler = lambda sigma, sigma_next: (torch.rand_like(x) - 0.5) * 2 * 1.73
        case "highres-pyramid":
            noise_sampler = lambda sigma, sigma_next: highres_pyramid_noise_like(x)
        case "perlin":
            noise_sampler = lambda sigma, sigma_next: rand_perlin_like(x)
        case _:
            noise_sampler = lambda sigma, sigma_next: (torch.rand_like(x) - 0.5) * 2 * 1.73
    return sample_clyb_4m_sde_momentumized(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, momentum=momentum)

# Add your personal samplers below here, just for formatting purposes ;3

# Add any extra samplers to the following dictionary
extra_samplers = {
    "res_momentumized": sample_res_solver,
    "dpmpp_dualsde_momentumized": sample_dpmpp_dualsdemomentum,
    "clyb_4m_sde_momentumized": sample_clyb_4m_sde,
    "ttm": sample_ttmcustom,
    "lcm_custom_noise": sample_lcmcustom,
}