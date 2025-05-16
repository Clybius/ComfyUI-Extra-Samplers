import math

import torch
from torch import nn, FloatTensor
import torchsde
import kornia
from tqdm.auto import trange, tqdm
import numpy as np

import comfy.sample
import comfy.model_patcher

from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler, PIDStepSizeController, get_ancestral_step, to_d, default_noise_sampler, DPMSolver

# The following function adds the samplers during initialization, in __init__.py
def add_samplers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    if hasattr(KSampler, "DISCARD_PENULTIMATE_SIGMA_SAMPLERS"):
        KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS |= discard_penultimate_sigma_samplers
    added = 0
    for sampler in extra_samplers: #getattr(self, "sample_{}".format(extra_samplers))
        if sampler not in KSampler.SAMPLERS:
            try:
                idx = KSampler.SAMPLERS.index("uni_pc_bh2") # Last item in the samplers list
                KSampler.SAMPLERS.insert(idx+1, sampler) # Add our custom samplers
                setattr(k_diffusion_sampling, "sample_{}".format(sampler), extra_samplers[sampler])
                added += 1
            except ValueError as _err:
                pass
    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)

# The following function adds the samplers during initialization, in __init__.py
def add_schedulers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    added = 0
    for scheduler in extra_schedulers: #getattr(self, "sample_{}".format(extra_samplers))
        if scheduler not in KSampler.SCHEDULERS:
            try:
                idx = KSampler.SCHEDULERS.index("ddim_uniform") # Last item in the samplers list
                KSampler.SCHEDULERS.insert(idx+1, scheduler) # Add our custom samplers
                setattr(k_diffusion_sampling, "get_sigmas_{}".format(scheduler), extra_schedulers[scheduler])
                added += 1
            except ValueError as err:
                pass
    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)


# Noise samplers
IMMISCIBLE_NOISE_NAMES=("gaussian", "perlin")
NOISE_SAMPLER_NAMES=("gaussian", "uniform", "brownian", "highres-pyramid", "pyramid", "perlin", "laplacian", "immiscible_gaussian", "immiscible_gaussian_maximize", "immiscible_perlin", "immiscible_perlin_maximize")

def get_noise_sampler_names(default=None):
    if not default:
        return NOISE_SAMPLER_NAMES
    return (default,) + tuple(n for n in NOISE_SAMPLER_NAMES if n != default)

def get_immiscible_noise_sampler_names(default=None):
    if not default:
        return IMMISCIBLE_NOISE_NAMES
    return (default,) + tuple(n for n in IMMISCIBLE_NOISE_NAMES if n != default)

def mk_noise_sampler(x, fun):
    return lambda _sigma, _sigma_next: fun(x)

def get_noise_sampler(x, sigmas, noise_sampler_type="brownian", extra_args=None, cpu=False):
    if noise_sampler_type == "brownian":
        seed = extra_args.get("seed", None) if extra_args else None
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        return BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=cpu)
    return mk_noise_sampler(x, NOISE_SAMPLER_HANDLERS.get(noise_sampler_type, uniform_noise_like))

from torch import Generator, Tensor, lerp
from torch.nn.functional import unfold
from typing import Callable, Tuple
from math import pi

def uniform_noise_like(x):
    return (torch.rand_like(x) - 0.5) * 2 * 1.73

from scipy.optimize import linear_sum_assignment
def check_set_immiscible(x, noise_sampler_type, extra_args):
    if noise_sampler_type.startswith("immiscible"):
        match noise_sampler_type:
            case "immiscible_gaussian":
                immiscibility = make_immiscible("gaussian") # FINISH THE REST
                extra_args = immiscibility.set_immiscible_extra_args(extra_args)
                noise_sampler = lambda _sigma, _sigma_next: immiscibility(x)
                return noise_sampler, extra_args
            case "immiscible_gaussian_maximize":
                immiscibility = make_immiscible("gaussian", maximize=True) # FINISH THE REST
                extra_args = immiscibility.set_immiscible_extra_args(extra_args)
                noise_sampler = lambda _sigma, _sigma_next: immiscibility(x)
                return noise_sampler, extra_args
            case "immiscible_perlin":
                immiscibility = make_immiscible("perlin") # FINISH THE REST
                extra_args = immiscibility.set_immiscible_extra_args(extra_args)
                noise_sampler = lambda _sigma, _sigma_next: immiscibility(x)
                return noise_sampler, extra_args
            case "immiscible_perlin_maximize":
                immiscibility = make_immiscible("perlin", maximize=True) # FINISH THE REST
                extra_args = immiscibility.set_immiscible_extra_args(extra_args)
                noise_sampler = lambda _sigma, _sigma_next: immiscibility(x)
                return noise_sampler, extra_args
    return None, extra_args

class make_immiscible:
    def __init__(self, noise_func="gaussian", immiscible_latents=1024, maximize=False):
        self.noise_func = noise_func
        self.n_latents = immiscible_latents
        self.maximize = maximize
        self.updated_latent = None

    """
    def __call__(self, latents):
        # "Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment" (2024) Li et al. arxiv.org/abs/2406.12303
        # Minimize latent-noise pairs over a batch
        # Code from https://github.com/kohya-ss/sd-scripts/pull/1395
        reference_latent = latents
        if self.updated_latent != None:
            reference_latent = self.updated_latent
        reference_latent = self.batch(reference_latent)
        n = self.n_latents # arg is an integer for how many noise tensors to generate
        noise = None
        match self.noise_func:
            case "gaussian_1024":
                #n = 1024
                size = [n] + list(reference_latent.shape[1:])
                noise = torch.randn(size, dtype=reference_latent.dtype, layout=reference_latent.layout, device=reference_latent.device)
            case "perlin":
                #n = n//32
                size = [n] + list(reference_latent.shape[1:])
                noise = torch.randn(size, dtype=reference_latent.dtype, layout=reference_latent.layout, device=reference_latent.device)
                for i in range(n):
                    for j in range(reference_latent.size(dim=1)):
                        noise_values = rand_perlin_2d_octaves((reference_latent.size(dim=-2), reference_latent.size(dim=-1)), (1,1), 1, 1).to(reference_latent.device)
                        result = (1+0/10)*torch.erfinv(2 * noise_values - 1) * (2 ** 0.5)
                        result = torch.where(torch.abs(result) > 5, noise[i, j, :, :], result)
                        noise[i, j, :, :] = result
        latents_expanded = reference_latent.half().unsqueeze(1).expand(-1, n, *reference_latent.shape[1:])
        noise_expanded = noise.half().unsqueeze(0).expand(reference_latent.shape[0], *noise.shape)
        dist = (latents_expanded - noise_expanded)**2
        dist = dist.mean(list(range(2, dist.dim()))).cpu()
        assign_mat = linear_sum_assignment(dist, maximize=self.maximize)
        noise = noise[assign_mat[1]]
        return self.unbatch(noise, latents)

    def batch(self, ref):
        if self.batching == "batch":
            return ref
        rsz = ref.shape
        if len(rsz) != 4:
            raise ValueError("Reference must be four-dimensional")
        if self.batching == "channel":
            ref = ref.view(rsz[0] * rsz[1], *rsz[2:])
            return ref
        if self.batching == "row":
            ref = ref.view(rsz[0] * rsz[1] * rsz[2], rsz[3])
            return ref
        if self.batching == "column":
            ref = ref.permute(0, 1, 3, 2).reshape(rsz[0] * rsz[1] * rsz[3], rsz[2])
            return ref
        raise ValueError("Bad Immmiscible noise batching type")
    
    def unbatch(self, noise, x_ref):
        xsz = x_ref.shape
        if self.batching == "column":
            return noise.view(*xsz[:2], xsz[3], xsz[2]).permute(0, 1, 3, 2)
        return noise.view(*xsz)

    """

    def __call__(self, latents):
        reference_latent = latents
        if self.updated_latent != None:
            reference_latent = self.updated_latent

        batch_size = latents.shape[0] if self.n_latents is None else self.n_latents
        size = [batch_size] + list(latents.shape[1:])
        #noise = torch.randn_like(latents)  # [B, C, H, W]

        match self.noise_func:
            case "gaussian":
                noise = torch.randn(size, dtype=latents.dtype, layout=latents.layout, device=latents.device)
            case "perlin":
                noise = create_noisy_latents_perlin(torch.randn(size, dtype=latents.dtype, layout=latents.layout, device=latents.device))

        # Distance calculation (simplified for single process)
        distance = torch.linalg.vector_norm(
            0.10 * latents.to(torch.float16).flatten(start_dim=1).unsqueeze(1) -
            0.10 * noise.to(torch.float16).flatten(start_dim=1).unsqueeze(0),
            dim=2
        )  # [B, B]

        # Noise Assignment (simplified for single process)
        _, col_ind = linear_sum_assignment(distance.cpu().numpy(), maximize=self.maximize)
        noise = noise[col_ind].to(latents.device)  # Assign the permuted noise

        return noise

    def set_immiscible_extra_args(self, extra_args):
        def immiscible_post_cfg_function(args):
            self.updated_latent = args["cond_denoised"]
            return args["denoised"]
        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, immiscible_post_cfg_function, disable_cfg1_optimization=True)
        return extra_args

# From https://github.com/Extraltodeus/noise_latent_perlinpinpin/blob/main/latent_noisy_perlin.py
# which was found at https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
# which was ported from https://github.com/pvigier/perlin-numpy/blob/master/perlin2d.py
def rand_perlin_2d(shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    
    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
    angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
    
    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
    
    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    noise = torch.remainder(torch.abs(noise)*1000000,11)/11
    # noise = (torch.sin(torch.remainder(noise*1000000,83))+1)/2
    return noise

def create_noisy_latents_perlin(x, detail_level=0):
    batch_size = x.size(dim=0)
    noise = torch.randn((batch_size, x.size(dim=1), x.size(dim=2), x.size(dim=3)), dtype=x.dtype, layout=x.layout, device=x.device)
    for i in range(batch_size):
        for j in range(x.size(dim=1)):
            noise_values = rand_perlin_2d_octaves((x.size(dim=2), x.size(dim=3)), (1,1), 1, 1).to(x.device)
            result = (1+detail_level/10)*torch.erfinv(2 * noise_values - 1) * (2 ** 0.5)
            result = torch.where(torch.abs(result) > 3, noise[i, j, :, :], result)
            noise[i, j, :, :] = result
    return noise

def rand_perlin_like(x): # Even distribution, seemingly produces more information in non-subject areas than the normal (gaussian) noise sampler
    return create_noisy_latents_perlin(x)

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

from torch.distributions import Laplace
def rand_laplacian_like(x):
    noise = torch.zeros_like(x)#.div_(4.0)
    noise += Laplace(loc=0, scale=2 ** 0.5).rsample(x.size()).to(noise.device)
    return noise / noise.std()

def highres_pyramid_noise_like(x, discount=0.7):
    b, c, h, w = x.shape # EDIT: w and h get over-written, rename for a different variant!
    orig_h = h
    orig_w = w
    u = torch.nn.Upsample(size=(orig_h, orig_w), mode='bilinear')
    noise = (torch.rand_like(x) - 0.5) * 2 * 1.73 # Start with scaled uniform noise
    for i in range(4):
        r = torch.rand(1).item() * 2 + 2 # Rather than always going 2x,
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

# I'm not sure how this differs from the other implementation but it doesn't seem to be used at present.
def power_noise_sampler_2(tensor, alpha=2, k=1): # This doesn't work properly right now
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

def pyramid_noise_like(size, dtype, layout, generator, device="cpu", discount=0.8):
    b, c, h, w = size
    orig_h = h
    orig_w = w
    noise = torch.zeros(size=size, dtype=dtype, layout=layout, device=device)
    r = 1
    for i in range(5):
        r *= 2 # Rather than always going 2x,
        #w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
        noise += torch.nn.functional.interpolate((torch.normal(mean=0, std=0.5 ** i, size=(b, c, h * r, w * r), dtype=dtype, layout=layout, generator=generator, device=device)), size=(orig_h, orig_w), mode='nearest-exact') * discount**i
        #if w>=orig_w*16 or h>=orig_h*16: break
    return noise

def power_noise_sampler(size, dtype, layout, generator, device="cpu", alpha=2, k=1): # This doesn't work properly right now
    """Generate 1/f noise for a given tensor.

    Args:
        tensor: The tensor to add noise to.
        alpha: The parameter that determines the slope of the spectrum.
        k: A constant.

    Returns:
        A tensor with the same shape as `tensor` containing 1/f noise.
    """
    tensor = torch.randn(size=size, dtype=dtype, layout=layout, generator=generator, device=device)
    fft = torch.fft.fft2(tensor)
    freq = torch.arange(1, len(fft) + 1, dtype=torch.float)
    spectral_density = k / freq**alpha
    noise = torch.rand(size=size, dtype=dtype, layout=layout, generator=generator, device=device) * spectral_density
    mean = torch.mean(noise, dim=(-2, -1), keepdim=True).to(tensor.device)
    std = torch.std(noise, dim=(-2, -1), keepdim=True).to(tensor.device)
    noise = noise.to(tensor.device).sub_(mean).div_(std)
    return noise

def prepare_noise(latent_image, seed, noise_type, noise_inds=None): # From `sample.py`
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    generator = torch.manual_seed(seed)
    match noise_type:
        case "gaussian":
            noise_func = torch.randn
        case "uniform":
            def uniform_rand(*size, **kwargs):
                return (torch.rand(*size, **kwargs) - 0.5) * 2 * 1.73
            noise_func = uniform_rand
        case "pyramid":
            noise_func = pyramid_noise_like
        case "power":
            noise_func = power_noise_sampler
        case _:
            noise_func = torch.randn
    if noise_inds is None:
        return noise_func(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = noise_func([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises

NOISE_SAMPLER_HANDLERS={
    # Brownian is special-cased.
    "gaussian": torch.randn_like,
    "highres-pyramid": highres_pyramid_noise_like,
    "pyramid": lambda x: pyramid_noise_like(x.size(), x.dtype, x.layout, None, device=x.device),
    "perlin": rand_perlin_like,
    "laplacian": rand_laplacian_like,
    "uniform": uniform_noise_like,
}


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
def sample_res_solver(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian", noise_sampler=None, denoise_to_zero=True, simple_phi_calc=False, c2=0.5, ita=torch.Tensor((0.25,)), momentum=0.0):
    if len(sigmas) <= 1:
        return x
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sample_refined_exp_s(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args), denoise_to_zero=denoise_to_zero, simple_phi_calc=simple_phi_calc, c2=c2, ita=ita, momentum=momentum)

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

def sample_dpmpp_dualsdemomentum(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler_type="gaussian", noise_sampler=None, r=1/2, momentum=0.0):
    if len(sigmas) <= 1:
        return x
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sample_dpmpp_dualsde_momentum(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args), r=r, momentum=momentum)

from .other_samplers.sample_ttm import sample_ttm_jvp
def sample_ttmcustom(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler_type="gaussian",noise_sampler=None):
    if len(sigmas) <= 1:
        return x
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sample_ttm_jvp(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args))

from comfy.k_diffusion.sampling import sample_lcm
def sample_lcmcustom(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian", noise_sampler=None):
    if len(sigmas) <= 1:
        return x
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sample_lcm(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args))

def sample_clyb_4m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler_type="brownian", noise_sampler=None, momentum=0.0):
    if len(sigmas) <= 1:
        return x
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sample_clyb_4m_sde_momentumized(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args), momentum=momentum)


# This code works, but I'm currently experimenting with different methods
@torch.no_grad()
def sampler_euler_ancestral_dancing(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, leap=2, eta_dance=1.0):
    #Ancestral sampling with Euler method steps, dancing steps.
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    unsample_noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if i < len(sigmas) - leap:
            is_danceable = sigmas[i + leap] > 0
        else:
            is_danceable = False
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + leap] if is_danceable else sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            if is_danceable:
                x = x + noise_sampler(sigmas[i], sigmas[i + leap]) * s_noise * sigma_up
                #x = x + noise_sampler(sigmas[i + 2], sigmas[i + 1]) * s_noise * sigma_up
                #denoised2 = model(x, sigmas[i + 2] * s_in, **extra_args)
                sigma_down2, sigma_up2 = get_ancestral_step(sigmas[i + leap], sigmas[i + 1], eta=eta_dance)
                d_2 = to_d(x, sigmas[i + leap], denoised)
                dt_2 = sigma_down2 - sigmas[i + leap]
                x = x + d_2 * dt_2
                x = x + noise_sampler(sigmas[i + leap], sigmas[i + 1]) * s_noise * sigma_up2

                #sigma_down3, sigma_up3 = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
                #x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up3

                #denoised2 = model(x, sigmas[i] * s_in, **extra_args)
                #d_3 = to_d(x, sigmas[i], denoised2)
                #dt_3 = sigma_down3 - sigmas[i]
                #x = x + d_3 * dt_3 + d_2 * dt_2
                #print(dt_3, dt_2)
                #x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up3
                #x = x + d * dt
            else:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

    return x

def sample_euler_ancestral_dancing(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler_type="gaussian", noise_sampler=None, leap=2, eta_dance=1.0):
    return sampler_euler_ancestral_dancing(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args), leap=leap, eta_dance=eta_dance)


@torch.no_grad()
def sampler_dpmpp_3m_sde_dynamic_eta(model, x, sigmas, extra_args=None, callback=None, disable=None, eta_max=1.0, eta_min=0.0, s_noise=1., noise_sampler=None):
    """DPM-Solver++(3M) SDE with dynamic eta."""
    def eta_schedule_cosine_annealing(i, n, eta_max=eta_max, eta_min=eta_min):
        """Cosine annealing schedule for eta."""
        progress = i / (n - 1)
        eta = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * progress))
        return eta

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(3M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t

            # Dynamic eta
            eta = eta_schedule_cosine_annealing(i, len(sigmas))
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

def sample_dpmpp_3m_sde_dynamic_eta(model, x, sigmas, extra_args=None, callback=None, disable=None, eta_max=1.0, eta_min=0.0, s_noise=1., noise_sampler_type="brownian", noise_sampler=None):
    if len(sigmas) <= 1:
        return x
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sampler_dpmpp_3m_sde_dynamic_eta(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta_max=eta_max, eta_min=eta_min, s_noise=s_noise, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args))


from .other_samplers.refined_exp_solver import _de_second_order

# Default is 2, so only methods with other values are included here.
SUPREME_ORDER = { "euler": 1, "dpm_1s": 1, "dpm_3s": 3, "rk4": 4, "reversible_heun_1s": 1, "rkf45": 6, "bogacki_shampine": 3, }

@torch.no_grad()
def sampler_supreme(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., noise_sampler=None, eta=1.0, step_method="euler", substep_method="euler", warmup_method="euler", centralization=0.00, normalization=0.00, edge_enhancement=0.00, perphist=0.25, substeps=2, noise_modulation="none", modulation_strength=2., modulation_dims=3, reversible_eta=1.0, dyneta=True, reversible_dyneta=True, enable_free_reverse=True, free_reverse_eta=0.0, free_reverse_dyneta=True):
    """
    Supreme Sampler, Euler steps. Based on no paper, purely interesting thoughts.

    Args:
        model: Denoising model call.
        x: The initial noisy sample.
        sigmas: The noise schedule.
        extra_args: Additional arguments for the model.
        callback: A callback function for monitoring the sampling process.
        disable: Whether to disable the progress bar.
        s_noise: The noise scale factor.
        noise_sampler: A custom noise sampler function.
        eta: Ancestral-ness.
        centralization: Subtracts mean from the denoised latent.
        normalization: Divides the denoised latent by the standard deviation.
        edge_enhancement: Multiplies the edges by the mean using a laplacian kernel
        perphist: Adds previous denoised variable to the current denoised using perpendicular vector projection
        substeps: Amount of times to iterate over each step and average the results
        noise_modulation: Method of changing the noise based on situations within the sampler
        modulation_strength: Strength of the modulation using a weighted sum between the modulation and noise sampler's noise.
        modulation_dims: Choose between (channel) modulation, (height, width) modulation, or (channels, height, width) modulation
        reversible_eta: Ancestralness in the reversible component of reversible samplers.
        dyneta: Enable a dynamic eta based on sigma. Higher sigmas have a lower eta, while lower sigmas have a higher eta, max clamped to user-chosen eta.
        reversible_dyneta: Enable a dynamic reversible eta based on sigma. Higher sigmas have a lower eta, while lower sigmas have a higher eta, max clamped to user-chosen eta. Good for stability.
    """

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    # Centralization
    def centralize(denoised_sample, centralization, iteration):
        for b in range(len(denoised_sample)):
            for c in range(len(denoised_sample[b])):
                channel = denoised_sample[b][c]
                denoised_sample[b][c] -= channel.mean() * centralization * (sigmas[iteration] ** 0.5)
        return denoised_sample

    # Normalization
    def normalize(denoised_sample, normalization, iteration):
        for b in range(len(denoised_sample)):
            for c in range(len(denoised_sample[b])):
                channel = denoised_sample[b][c]
                denoised_sample[b][c] += ((denoised_sample[b][c] / channel.std()) - denoised_sample[b][c]) * normalization * (sigmas[iteration] ** 0.5)
        return denoised_sample

    # Perp-hist
    def perpadd(denoised_tensor, old_denoised_tensor, x, alpha):
        a_diff = x - (denoised_tensor - x)
        b_diff = x - (old_denoised_tensor - x)
        a_ortho = a_diff * (a_diff / torch.linalg.norm(a_diff) * (b_diff / torch.linalg.norm(a_diff))).sum()
        b_perp = b_diff - a_ortho
        res = denoised_tensor + alpha * b_perp
        return res

    # DynETA
    orig_eta = eta
    orig_reversible_eta = reversible_eta
    orig_free_reverse_eta = free_reverse_eta
    def dyneta_fn(original_eta, sigma, sigma_next):
        return torch.clamp(1 / (sigma**2 - sigma_next**2)**0.5, min=0.0, max=original_eta)

    order, sub_order = SUPREME_ORDER.get(step_method, 2), SUPREME_ORDER.get(substep_method, 2)
    steps_per_sigma = order + sub_order * (substeps - 1)

    def apply_enhancements(x, i, model, sigma_s_in, old_denoised):
        args = extra_args
        denoised = model(x, sigma_s_in, **args)

        if edge_enhancement != 0:
            blur = (kornia.filters.joint_bilateral_blur(x, denoised, (3, 3), 0.1, (1.5, 1.5)) - x) # Blurs non-edges
            denoised += (kornia.filters.unsharp_mask(denoised, (3, 3), (1.5, 1.5)) - denoised) * (sigmas[i] - sigmas[i + 1]) * edge_enhancement # Sharpens everything
            denoised += blur * (sigmas[i] - sigmas[i + 1]) * edge_enhancement # Apply blur to non-edges, thus leaving edges sharpened

        if centralization != 0:
            denoised = centralize(denoised, centralization, i)

        if normalization != 0:
            denoised = normalize(denoised, normalization, i)

        if old_denoised != None and perphist != 0:
            denoised = perpadd(denoised, old_denoised, x, perphist)

        return denoised

    # Dynamic sampling
    dynamic_order_samplers = {
        1: "euler",
        2: "trapezoidal",
        3: "bogacki_shampine",
        4: "rk4",
        6: "rkf45",
    }
    # Adaptive RK order sampling
    adaptive_rk_weights = {
        1: [1],
        2: [0.5, 0.5],
        3: [1/6, 2/3, 1/6],
        4: [1/8, 3/8, 3/8, 1/8],
    }

    def dynamic_step_method(step_method, model, prev_x, denoised, prev_denoised, iteration, substep_iter):
        """
        Step method function, applies cond-error modification, and dynamic step selection if chosen.
        """
        sampler = step_method
        order = 1
        error = 0
        if iteration == 0 or prev_denoised == None: # Warmup with the chosen warmup step, else use substep method for substeps
            if warmup_method == "none":
                return step_method
            if substep_iter > 0:
                return substep_method, 1, error
            order = 2 # Chosen for simplicity
            return warmup_method, order, error

        d = to_d(prev_x, sigmas[iteration - 1], prev_denoised)
        x_pred = prev_x + d * (sigmas[iteration] - sigmas[iteration - 1])

        d_pred = to_d(x_pred, sigmas[iteration], denoised)

        error = torch.linalg.norm(d_pred - d) / torch.linalg.norm(d)

        if substep_iter > 0:
            return substep_method, 1, error
        if step_method != "dynamic" and step_method != "adaptive_rk": # If we're not a dynamic sampler, return the step unmodified step method
            return step_method, order, error

        if (error < 1e-2):
            order = 6
        elif (error < 3.75e-2):
            order = 4
        elif (error < 7.5e-2):
            order = 3
        elif (error < 1.5e-1):
            order = 2
        else:
            order = 1

        if step_method == "adaptive_rk":
            return step_method, min(order, 4), error
        return dynamic_order_samplers[order], order, error

    renoise_weights = torch.ones(substeps, device=x.device) / substeps
    def intensity_based_multiplicative_noise_fn(x, noise, s_noise, sigma_up, intensity, dims):
        """
        Scales noise based on the intensities of the input tensor.
        """
        std = torch.std(x - x.mean(), dim=dims, keepdim=True)  # Average across channels to get intensity
        scaling = (1 / (std * abs(intensity) + 1.0)) # Scale std by intensity, as not doing this leads to more noise being left over, leading to crusty/preceivably extremely oversharpened images
        additive_noise = noise * s_noise * sigma_up
        scaled_noise = noise * s_noise * sigma_up * scaling + additive_noise

        noise_norm = torch.norm(additive_noise)
        scaled_noise_norm = torch.norm(scaled_noise)
        scaled_noise *= noise_norm / scaled_noise_norm # Scale to normal noise strength
        scaled_noise = scaled_noise * intensity + additive_noise * (1 - intensity)
        return scaled_noise

    def frequency_based_noise(z_k, noise, s_noise, sigma_up, intensity, channels):
        """
        Scales the high-frequency components of the noise based on the given intensity.
        """
        additive_noise = noise * s_noise * sigma_up

        std = torch.std(z_k - z_k.mean(), dim=channels, keepdim=True)  # Average across channels to get intensity
        scaling = (1 / (std * abs(intensity) + 1.0))
        # Perform Fast Fourier Transform (FFT)
        z_k_freq = torch.fft.fft2(scaling * additive_noise + additive_noise)

        # Get the magnitudes of the frequency components
        magnitudes = torch.abs(z_k_freq)

        # Create a high-pass filter (emphasize high frequencies)
        h, w = z_k.shape[-2:]
        b = abs(intensity)  # Controls the emphasis of the high pass (higher frequencies are boosted)
        high_pass_filter = 1 - torch.exp(-((torch.arange(h)[:, None] / h)**2 + (torch.arange(w)[None, :] / w)**2) * b**2)
        high_pass_filter = high_pass_filter.to(z_k.device)

        # Apply the filter to the magnitudes
        magnitudes_scaled = magnitudes * (1 + high_pass_filter)

        # Reconstruct the complex tensor with scaled magnitudes
        z_k_freq_scaled = magnitudes_scaled * torch.exp(1j * torch.angle(z_k_freq))

        # Perform Inverse Fast Fourier Transform (IFFT)
        z_k_scaled = torch.fft.ifft2(z_k_freq_scaled)

        # Return the real part of the result
        z_k_scaled = torch.real(z_k_scaled)

        noise_norm = torch.norm(additive_noise)
        scaled_noise_norm = torch.norm(z_k_scaled)

        z_k_scaled *= (noise_norm / scaled_noise_norm) # Scale to normal noise strength

        scaled_noise = z_k_scaled * intensity + additive_noise * (1 - intensity)

        return scaled_noise
    
    def spectral_modulate_noise(z_k, noise, s_noise, sigma_up, intensity, channels, spectral_mod_percentile=5.0): # Modified for soft quantile adjustment using a novel:tm::c::r: method titled linalg.
        additive_noise = noise * s_noise * sigma_up
        # Convert image to Fourier domain
        fourier = torch.fft.fftn(additive_noise, dim=channels)  # Apply FFT along Height and Width dimensions
    
        log_amp = torch.log(torch.sqrt(fourier.real ** 2 + fourier.imag ** 2))

        quantile_low = torch.quantile(
            log_amp.abs().flatten(1),
            spectral_mod_percentile * 0.01,
            dim = 1
        ).unsqueeze(-1).unsqueeze(-1).expand(log_amp.shape)
        
        quantile_high = torch.quantile(
            log_amp.abs().flatten(1),
            1 - (spectral_mod_percentile * 0.01),
            dim = 1
        ).unsqueeze(-1).unsqueeze(-1).expand(log_amp.shape)

        quantile_max = torch.quantile(
            log_amp.abs().flatten(1),
            1,
            dim = 1
        ).unsqueeze(-1).unsqueeze(-1).expand(log_amp.shape)

        # Decrease high-frequency components
        mask_high = log_amp > quantile_high # If we're larger than 95th percentile

        additive_mult_high = torch.where(
            mask_high,
            1 - ((log_amp - quantile_high) / (quantile_max - quantile_high)).clamp_(max=0.5), # (1) - (0-1), where 0 is 95th %ile and 1 is 100%ile
            torch.tensor(1.0)
        )
        

        # Increase low-frequency components
        mask_low = log_amp < quantile_low
        additive_mult_low = torch.where(
            mask_low,
            1 + (1 - (log_amp / quantile_low)).clamp_(max=0.5), # (1) + (0-1), where 0 is 5th %ile and 1 is 0%ile
            torch.tensor(1.0)
        )
        
        mask_mult = ((additive_mult_low * additive_mult_high) ** intensity)
        #print(mask_mult)
        filtered_fourier = fourier * mask_mult
        
        # Inverse transform back to spatial domain
        inverse_transformed = torch.fft.ifftn(filtered_fourier, dim=channels)  # Apply IFFT along Height and Width dimensions
        
        scaled_noise = inverse_transformed.real.to(additive_noise.device)

        #noise_norm = torch.norm(additive_noise)
        #scaled_noise_norm = torch.norm(scaled_noise)

        return scaled_noise# * (noise_norm / scaled_noise_norm)

    dims = (-3, -2, -1)
    match modulation_dims:
        case 1:
            dims = (-3)
        case 2:
            dims = (-2, -1)
        case 3:
            dims = (-3, -2, -1)

    orig_model = model
    old_denoised = None
    prev_denoised = None
    prev_x = x
    for i in trange(len(sigmas) - 1, disable=disable):
        def model(x, sigma_s_in, **extra_args): # Model wrapper to apply enhancements at every call
            nonlocal old_denoised
            denoised = apply_enhancements(x, i, orig_model, sigma_s_in, old_denoised)
            old_denoised = denoised
            if callback is not None:
                callback({'x': z_k, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            return denoised

        dpm_solver = DPMSolver(model, extra_args)

        # DynETA
        if dyneta: eta = dyneta_fn(orig_eta, sigmas[i], sigmas[i + 1])
        if reversible_dyneta: reversible_eta = dyneta_fn(orig_reversible_eta, sigmas[i], sigmas[i + 1])

        # Renoising iterations
        z_avg = torch.zeros_like(x)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        sigma_down_reversible, _ = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=reversible_eta)
        for k in range(substeps):
            z_k = x
            orig_zk = z_k
            eps_cache = {}

            denoised = model(z_k, sigmas[i] * s_in, **extra_args)


            eps = (z_k - denoised) / sigmas[i]
            eps_cache = {'eps': eps}

            step_method_dyn, order, error = dynamic_step_method(step_method, model, prev_x, denoised, prev_denoised, i, k) #step_method, model, prev_x, denoised, prev_denoised, i, k

            match step_method_dyn if sigmas[i + 1] != 0 else "euler":
                case "euler": # 1 model call
                    d = to_d(z_k, sigmas[i], denoised)
                    dt = sigma_down - sigmas[i]

                    z_k = z_k + d * dt
                case "dpm_1s": # DPM Family, 1 model call
                    if callback is not None:
                        dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
                    z_k, eps_cache = dpm_solver.dpm_solver_1_step(z_k, dpm_solver.t(sigmas[i]), dpm_solver.t(sigma_down), eps_cache=eps_cache)
                case "dpm_2s": # 2 model calls
                    if callback is not None:
                        dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
                    z_k, eps_cache = dpm_solver.dpm_solver_2_step(z_k, dpm_solver.t(sigmas[i]), dpm_solver.t(sigma_down), eps_cache=eps_cache)
                case "dpm_3s": # 3 model calls
                    if callback is not None:
                        dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
                    z_k, eps_cache = dpm_solver.dpm_solver_3_step(z_k, dpm_solver.t(sigmas[i]), dpm_solver.t(sigma_down), eps_cache=eps_cache)
                case "rk4": # Fourth-order Runge-Kutta method, 4 model calls
                    # Calculate the derivative using the model
                    d = to_d(z_k, sigmas[i], denoised)
                    dt = sigma_down - sigmas[i]

                    # Runge-Kutta steps
                    k1 = d * dt
                    k2 = to_d(z_k + k1 / 2, sigmas[i] + dt / 2, model(z_k + k1 / 2, (sigmas[i] + dt / 2) * s_in, **extra_args)) * dt
                    k3 = to_d(z_k + k2 / 2, sigmas[i] + dt / 2, model(z_k + k2 / 2, (sigmas[i] + dt / 2) * s_in, **extra_args)) * dt
                    k4 = to_d(z_k + k3, sigmas[i] + dt, model(z_k + k3, (sigmas[i] + dt) * s_in, **extra_args)) * dt

                    # Update the sample
                    z_k = z_k + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                case "reversible_heun": # 2 model calls
                    sigma_i, sigma_i_plus_1 = sigmas[i], sigma_down
                    dt = sigma_i_plus_1 - sigma_i
                    dt_reversible = sigma_down_reversible - sigma_i

                    # Calculate the derivative using the model
                    d_i = to_d(z_k, sigma_i, denoised)

                    # Predict the sample at the next sigma using Euler step
                    x_pred = z_k + d_i * dt

                    # Denoised sample at the next sigma
                    denoised_i_plus_1 = model(x_pred, sigma_i_plus_1 * s_in, **extra_args)

                    # Calculate the derivative at the next sigma
                    d_i_plus_1 = to_d(x_pred, sigma_i_plus_1, denoised_i_plus_1)

                    # Update the sample using the Reversible Heun formula
                    z_k = z_k + dt * (d_i + d_i_plus_1) / 2 - dt_reversible**2 * (d_i_plus_1 - d_i) / 4
                case "reversible_heun_1s": # Experimental 1 model call variant, utilizing previous denoised variables to speed up diffusion.
                    # Reversible Heun-inspired update (first-order)
                    sigma_i, sigma_i_plus_1 = sigmas[i], sigma_down
                    dt = sigma_i_plus_1 - sigma_i
                    dt_reversible = sigma_down_reversible - sigma_i

                    # Calculate the derivative using the model
                    d_i_old = to_d(z_k, sigma_i, prev_denoised) if prev_denoised is not None else to_d(z_k, sigma_i, model(z_k, sigma_i * s_in, **extra_args))

                    # Predict the sample at the next sigma using Euler step
                    x_pred = z_k + d_i_old * dt

                    # Calculate the derivative at the next sigma
                    d_i_plus_1 = to_d(x_pred, sigma_i_plus_1, denoised)

                    # Update the sample using the Reversible Heun formula
                    z_k = z_k + dt * (d_i_old + d_i_plus_1) / 2 - dt_reversible**2 * (d_i_plus_1 - d_i_old) / 4
                case "rkf45": # 6 model calls (expensive)
                    sigma_i, sigma_i_plus_1 = sigmas[i], sigma_down
                    dt = sigma_i_plus_1 - sigma_i
                    # Calculate the derivative using the model
                    d_i = to_d(z_k, sigmas[i], denoised)
                    # RKF45 steps
                    k1 = d_i * dt
                    k2 = to_d(z_k + k1 / 4, sigmas[i] + dt / 4, model(z_k + k1 / 4, (sigmas[i] + dt / 4) * s_in, **extra_args)) * dt
                    k3 = to_d(z_k + 3 * k1 / 32 + 9 * k2 / 32, sigmas[i] + 3 * dt / 8, model(z_k + 3 * k1 / 32 + 9 * k2 / 32, (sigmas[i] + 3 * dt / 8) * s_in, **extra_args)) * dt
                    k4 = to_d(z_k + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197, sigmas[i] + 12 * dt / 13, model(z_k + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197, (sigmas[i] + 12 * dt / 13) * s_in, **extra_args)) * dt
                    k5 = to_d(z_k + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104, sigmas[i] + dt, model(z_k + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104, (sigmas[i] + dt) * s_in, **extra_args)) * dt

                    # Update the sample
                    z_k = z_k + 25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5
                case "adaptive_rk":
                    sigma_i, sigma_i_plus_1 = sigmas[i], sigma_down
                    dt = sigma_i_plus_1 - sigma_i

                    # Calculate the derivative using the model
                    d_i = to_d(z_k, sigma_i, denoised)

                    # Adaptive order Runge-Kutta steps
                    k_values = [d_i * dt]  # Initialize with k1
                    for j in range(1, order):
                        # Calculate intermediate k values based on the current order
                        k_sum = sum(adaptive_rk_weights[order][l] * k_values[l] for l in range(j))
                        k_values.append(to_d(z_k + k_sum, sigma_i + dt * sum(adaptive_rk_weights[order][:j]), model(z_k + k_sum, (sigma_i + dt * sum(adaptive_rk_weights[order][:j])) * s_in, **extra_args)) * dt)

                    # Update the sample using the weighted sum of k values
                    z_k = z_k + sum(adaptive_rk_weights[order][j] * k_values[j] for j in range(order))
                case "bogacki_shampine":
                    sigma_i, sigma_i_plus_1 = sigmas[i], sigma_down
                    dt = sigma_i_plus_1 - sigma_i

                    # Calculate the derivative using the model
                    d_i = to_d(z_k, sigma_i, denoised)

                    # Bogacki-Shampine steps
                    k1 = d_i * dt
                    k2 = to_d(z_k + k1 / 2, sigma_i + dt / 2, model(z_k + k1 / 2, (sigma_i + dt / 2) * s_in, **extra_args)) * dt
                    k3 = to_d(z_k + 3 * k1 / 4 + k2 / 4, sigma_i + 3 * dt / 4, model(z_k + 3 * k1 / 4 + k2 / 4, (sigma_i + 3 * dt / 4) * s_in, **extra_args)) * dt

                    # Update the sample
                    z_k = z_k + 2 * k1 / 9 + k2 / 3 + 4 * k3 / 9
                case "reversible_bogacki_shampine":
                    sigma_i, sigma_i_plus_1 = sigmas[i], sigma_down
                    dt = sigma_i_plus_1 - sigma_i
                    dt_reversible = sigma_down_reversible - sigma_i

                    # Calculate the derivative using the model
                    d_i = to_d(z_k, sigma_i, denoised)

                    # Bogacki-Shampine steps
                    k1 = d_i * dt
                    k2 = to_d(z_k + k1 / 2, sigma_i + dt / 2, model(z_k + k1 / 2, (sigma_i + dt / 2) * s_in, **extra_args)) * dt
                    k3 = to_d(z_k + 3 * k1 / 4 + k2 / 4, sigma_i + 3 * dt / 4, model(z_k + 3 * k1 / 4 + k2 / 4, (sigma_i + 3 * dt / 4) * s_in, **extra_args)) * dt

                    # Reversible correction term (inspired by Reversible Heun)
                    correction = dt_reversible**2 * (k3 - k2) / 6

                    # Update the sample
                    z_k = z_k + 2 * k1 / 9 + k2 / 3 + 4 * k3 / 9 - correction
                case "trapezoidal": # 2 model calls
                    if sigmas[i + 1] > 0:
                        dt = sigmas[i + 1] - sigmas[i]

                        # Calculate the derivative using the model
                        d_i = to_d(z_k, sigmas[i], denoised)

                        # Predict the sample at the next sigma using Euler step
                        x_pred = z_k + d_i * dt

                        # Denoised sample at the next sigma
                        denoised_i_plus_1 = model(x_pred, sigmas[i + 1] * s_in, **extra_args)

                        # Calculate the derivative at the next sigma
                        d_i_plus_1 = to_d(x_pred, sigmas[i + 1], denoised_i_plus_1)

                        dt_2 = sigma_down - sigmas[i]
                        # Update the sample using the Trapezoidal rule
                        z_k = z_k + dt_2 * (d_i + d_i_plus_1) / 2
                    else:
                        z_k = denoised
                case "RES":
                    if sigmas[i + 1] > 0:
                        lam_next = sigma_down.log().neg() if eta != 0 else sigmas[i + 1].log().neg()
                        lam = sigmas[i].log().neg()

                        h = lam_next - lam
                        a2_1, b1, b2 = _de_second_order(h=h, c2=0.5, simple_phi_calc=False)

                        c2_h = 0.5*h

                        x_2 = math.exp(-c2_h)*z_k + a2_1*h*denoised
                        lam_2 = lam + c2_h
                        sigma_2 = lam_2.neg().exp()

                        denoised2 = model(x_2, sigma_2 * s_in, **extra_args)

                        z_k = math.exp(-h)*z_k + h*(b1*denoised + b2*denoised2)
                    else:
                        z_k = denoised
            
            # Free Reverse
            if enable_free_reverse:
                if free_reverse_dyneta: free_reverse_eta = dyneta_fn(orig_free_reverse_eta, sigmas[i], sigmas[i + 1])
                sigma_down_freereversible, _ = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=free_reverse_eta)
                
                d_i = to_d(orig_zk, sigmas[i], denoised)

                dt_reversible = sigma_down_freereversible - sigmas[i]

                d_i_old = to_d(prev_x, sigmas[i], prev_denoised) if prev_denoised is not None else to_d(prev_x, sigmas[i], model(prev_x, sigmas[i] * s_in, **extra_args))

                z_k = z_k + (d_i - d_i_old) / 2 * dt - dt_reversible**2 * (d_i_old - d_i) / 2

            z_avg += renoise_weights[k] * z_k
            if sigmas[i + 1] > 0: # Random noise for variance on ancestral samplers
                noise_mod = noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
                match noise_modulation:
                    case "none":
                        noise_mod = noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
                    case "intensity":
                        noise = noise_sampler(sigmas[i], sigmas[i + 1])
                        noise_mod = intensity_based_multiplicative_noise_fn(z_k, noise, s_noise, sigma_up, modulation_strength, dims)
                    case "frequency":
                        noise = noise_sampler(sigmas[i], sigmas[i + 1])
                        noise_mod = frequency_based_noise(z_k, noise, s_noise, sigma_up, modulation_strength, dims)
                    case "spectral_signum":
                        noise = noise_sampler(sigmas[i], sigmas[i + 1])
                        noise_mod = spectral_modulate_noise(x, noise, s_noise, sigma_up, modulation_strength, dims)
                z_k = z_k + noise_mod

        x = z_avg
        if sigmas[i + 1] > 0:
            noise_mod = noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
            match noise_modulation:
                case "none":
                    noise_mod = noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
                case "intensity":
                    noise = noise_sampler(sigmas[i], sigmas[i + 1])
                    noise_mod = intensity_based_multiplicative_noise_fn(x, noise, s_noise, sigma_up, modulation_strength, dims)
                case "frequency":
                    noise = noise_sampler(sigmas[i], sigmas[i + 1])
                    noise_mod = frequency_based_noise(x, noise, s_noise, sigma_up, modulation_strength, dims)
                case "spectral_signum":
                    noise = noise_sampler(sigmas[i], sigmas[i + 1])
                    noise_mod = spectral_modulate_noise(x, noise, s_noise, sigma_up, modulation_strength, dims)

            x = x + noise_mod

        prev_x = x
        prev_denoised = denoised

    return x

def sample_supreme(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., noise_sampler_type="gaussian", noise_sampler=None, eta=1.0, step_method="RES", substep_method="euler", warmup_method="euler", centralization=0.00, normalization=0.00, edge_enhancement=0.00, perphist=0.25, substeps=2, noise_modulation="none", modulation_strength=2., modulation_dims=3, reversible_eta=1.0, dyneta=True, reversible_dyneta=True, enable_free_reverse=True, free_reverse_eta=0.0, free_reverse_dyneta=True):
    if len(sigmas) <= 1:
        return x
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sampler_supreme(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args), eta=eta, step_method=step_method, substep_method=substep_method, warmup_method=warmup_method, centralization=centralization, normalization=normalization, edge_enhancement=edge_enhancement, perphist=perphist, substeps=substeps, noise_modulation=noise_modulation, modulation_strength=modulation_strength, modulation_dims=modulation_dims, reversible_eta=reversible_eta, dyneta=dyneta, reversible_dyneta=reversible_dyneta, enable_free_reverse=enable_free_reverse, free_reverse_eta=free_reverse_eta, free_reverse_dyneta=free_reverse_dyneta)

@torch.no_grad()
def sampler_sens(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., rsde_eta=1., tsde_eta=1., s_noise=1., noise_sampler=None, flow=False):
    """SDE-Endowed Nimble Sampler. Based off of DPM-Solver++(2M) SDE and DPM-Solver++(3M) SDE. R-SDE for reversible SDE, T-SDE for tertiary SDE."""
    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised, old_denoised_2 = None, None
    h_last, h_last_2 = None, None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h
            rsde_eta_h = rsde_eta * h
            tsde_eta_h = tsde_eta * h

            # If/for flow model
            downstep_ratio = 1 + (sigmas[i+1]/sigmas[i] - 1) * eta
            sigma_down = sigmas[i+1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i+1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i+1]**2 - sigma_down**2*alpha_ip1**2/alpha_down**2)**0.5

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised) / 2 - ((-h - rsde_eta_h).expm1().neg() / (-h - rsde_eta_h) + 1)**2 * (1 / r) * (old_denoised - denoised) / 2

            # DPM-Solver++(3M) SDE
            if h_last_2 is not None and tsde_eta:
                r = h_last_2 / h
                d = (old_denoised - old_denoised_2) / r
                d_2 = (old_denoised - denoised) / r
                
                d_rev = (denoised - old_denoised) / r
                d_2_rev = (old_denoised_2 - old_denoised) / r
                
                #phi = eta_h.neg().expm1() / eta_h + 1
                rphi = tsde_eta_h.neg().expm1() / tsde_eta_h + 1
                x = x + rphi * (d + d_2) / 2 - rphi**2 * (d_rev + d_2_rev) / 2

            if eta and not flow:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise
            elif eta and flow:
                x = (alpha_ip1/alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff

        old_denoised, old_denoised_2 = denoised, old_denoised
        h_last, h_last_2 = h, h_last
    return x

@torch.no_grad()
def sample_sens(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., rsde_eta=1., tsde_eta=1., s_noise=1., noise_sampler_type="brownian", noise_sampler=None):
    if len(sigmas) <= 1:
        return x
    flow = False
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        flow = True
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sampler_sens(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, rsde_eta=rsde_eta, tsde_eta=tsde_eta, s_noise=s_noise, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args), flow=flow)

#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
#under Apache 2 license
def sampler_ipndm_vapp(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=4, eta=1., s_noise=1., noise_sampler=None, pp_guidance=1.0):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    temp_uncond = [0]
    temp_cond = [0]
    def post_cfg_function(args):
        temp_uncond[0] = args["uncond_denoised"]
        temp_cond[0] = args["cond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])

    x_next = x
    t_steps = sigmas

    buffer_model = []
    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        sigma_down, sigma_up = get_ancestral_step(t_cur, t_next, eta=eta)

        x_cur = x_next

        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        faux_d_cur = (x_cur - temp_uncond[0]) / t_cur # CFG++
        #d_cur = ((x_cur - temp_cond[0]) - (denoised - temp_uncond[0])) / t_cur # 2x CFG
        d_cur = -temp_cond[0] / t_cur * pp_guidance + (x_cur - denoised) / t_cur + temp_uncond[0] / t_cur * pp_guidance
        # I've found that chhanging x_cur to `denoised` results in over-denoised samples, so we're sticking with this alt method

        order = min(max_order, i+1)
        if order == 1:      # First Euler step.
            x_next = x_cur + (sigma_down - t_cur) * d_cur # Modified t_next to sigma_down for ancestral capability.
        elif order == 2:    # Use one history point.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            coeff1 = (2 + (h_n / h_n_1)) / 2
            coeff2 = -(h_n / h_n_1) / 2
            x_next = x_cur + (sigma_down - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1])
        elif order == 3:    # Use two history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            h_n_2 = (t_steps[i-1] - t_steps[i-2])
            temp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp
            coeff3 = temp * h_n_1 / h_n_2
            x_next = x_cur + (sigma_down - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2])
        elif order == 4:    # Use three history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            h_n_2 = (t_steps[i-1] - t_steps[i-2])
            h_n_3 = (t_steps[i-2] - t_steps[i-3])
            temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
                   * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp1 + temp2
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))) * temp2
            coeff3 = temp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1 + h_n_2 / h_n_3)) * temp2
            coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2
            x_next = x_cur + (sigma_down - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2] + coeff4 * buffer_model[-3])
        
        if eta and sigmas[i + 1] > 0:
            x_next = x_next + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = faux_d_cur.detach() # Utilize CFG++ as history points
        else:
            buffer_model.append(faux_d_cur.detach())

    return x_next

@torch.no_grad()
def sample_ipndm_vapp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., max_order=4, noise_sampler_type="gaussian", noise_sampler=None, pp_guidance=1.0):
    if len(sigmas) <= 1:
        return x
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sampler_ipndm_vapp(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, max_order=max_order, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args), pp_guidance=pp_guidance)


import functools
import operator
@torch.no_grad()
def sampler_SHIDS(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, order=16, eta_order=1., solver_method="weighted_projection", flow=False):
    """Full ancestral sampling with SHIDS (Stochastic, Historical, Improvised Sampling) steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    temp_uncond = [0]
    temp_cond = [0]
    def post_cfg_function(args):
        temp_uncond[0] = args["uncond_denoised"]
        temp_cond[0] = args["cond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    old_uncond, old_uncond_2 = None, None
    old_cond, old_cond_2 = None, None
    old_dt, old_dt_2 = None, None

    buffer_model_cond = []
    #buffer_model_uncond = []
    #buffer_model_dt = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        _, sigma_up_order = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta_order)

        # If/for flow model
        downstep_ratio = None
        sigma_down_rf = None
        alpha_ip1 = None
        alpha_down = None
        renoise_coeff = None
        if flow:
            downstep_ratio = 1 + (sigmas[i+1]/sigmas[i] - 1) * eta
            sigma_down_rf = sigmas[i+1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i+1]
            alpha_down = 1 - sigma_down_rf
            renoise_coeff = (sigmas[i+1]**2 - sigma_down_rf**2*alpha_ip1**2/alpha_down**2)**0.5

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_full = to_d(x, sigmas[i], denoised)
        d = to_d(x, sigmas[i], temp_uncond[0])
        #d_2 = to_d(x, sigmas[i], temp_cond[0])
        # Euler method
        dt = sigma_down - sigmas[i] # Time Difference between now and next step (negative)
        x_full = denoised + d_full * sigma_down
        x_step = denoised + d * sigma_down

        # Project denoised onto a line between (primarily) x_step (cfgpp), and x_full (normal cfg)
        match solver_method:
            case "weighted_projection":
                ba = x_step - denoised
                ca = x_full - denoised
                alpha = (ba * ca) / (ba ** 2 + 1e-8)
                x = (1 - alpha)*denoised + alpha*x_step
            case "qr_decomposition":
                original_shape = x_step.shape
                if not original_shape:
                    shape_2d = (1, 1)
                elif len(original_shape) == 4:
                    shape_2d = (-1, functools.reduce(operator.mul, original_shape[1:]))
                else:
                    shape_2d = (-1, original_shape[-1])

                A = x_step.reshape(shape_2d)
                B = x_full.reshape(shape_2d)
                C = denoised.reshape(shape_2d)
                Q, _ = torch.qr(A - C)
                # Compute the mapping matrix
                mapping_matrix = torch.mm(Q.t(), B - C)
                mapped_tensor = torch.mm(Q, mapping_matrix)
                x = (C + mapped_tensor).reshape(original_shape)
            case "svd_lowrank":
                original_shape = x_step.shape
                if not original_shape:
                    shape_2d = (1, 1)
                elif len(original_shape) == 4:
                    shape_2d = (-1, functools.reduce(operator.mul, original_shape[1:]))
                else:
                    shape_2d = (-1, original_shape[-1])

                A = x_step.reshape(shape_2d)
                B = x_full.reshape(shape_2d)
                C = denoised.reshape(shape_2d)
                Ua, Sa, Va = torch.svd_lowrank(A - C, q=6, niter=2)
                
                A_lowrank = torch.mm(Ua, torch.mm(torch.diag(Sa), Va.t()))
                A_diff = (A - C) - A_lowrank

                Qb, _ = torch.qr(B - C)

                A_diff_projected = torch.mm(Qb, torch.mm(Qb.t(), A_diff))

                x = (B + A_diff_projected).reshape(original_shape)
            case "svd":
                original_shape = x_step.shape
                if not original_shape:
                    shape_2d = (1, 1)
                elif len(original_shape) == 4:
                    shape_2d = (-1, functools.reduce(operator.mul, original_shape[1:]))
                else:
                    shape_2d = (-1, original_shape[-1])

                A = x_step.reshape(shape_2d)
                B = x_full.reshape(shape_2d)
                C = denoised.reshape(shape_2d)
                Ua, Sa, Va = torch.linalg.svd(A - C, full_matrices=False, driver="gesvd")
                Ub, Sb, Vb = torch.linalg.svd(B - C, full_matrices=False, driver="gesvd")#Sb = torch.linalg.svdvals(B - C, driver="gesvd")#
                
                A_lowrank = torch.mm(Ub, torch.mm(torch.diag_embed(Sa), Vb))
                A_diff = (A - C) - A_lowrank

                #Qb, _ = torch.qr(B - C)

                #A_diff_projected = torch.mm(Ua, torch.mm(Ua.t(), A_diff))
                    
                x = (B + A_diff).reshape(original_shape)

        # Create a list of order multipliers
        multipliers = [i for i in range(1, len(buffer_model_cond))]
        # Normalize so that they're summed up to a total of 1
        total = sum(multipliers)
        normalized_multipliers = [m / total for m in multipliers]

        for iteration in range(len(buffer_model_cond) - 1):
            if not flow:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up_order * normalized_multipliers[iteration]
            elif flow and eta_order:
                downstep_ratio = 1 + (sigmas[i+1]/sigmas[i] - 1) * eta_order
                sigma_down_rf = sigmas[i+1] * downstep_ratio
                alpha_ip1 = 1 - sigmas[i+1]
                alpha_down = 1 - sigma_down_rf
                renoise_coeff = (sigmas[i+1]**2 - sigma_down_rf**2*alpha_ip1**2/alpha_down**2)**0.5
                x = (alpha_ip1/alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
            match solver_method:
                case "weighted_projection":
                    ba = x - buffer_model_cond[iteration]
                    ca = x_step - buffer_model_cond[iteration]
                    alpha = (ba * ca) / (ba ** 2 + 1e-8)
                    x = (1 - alpha)*buffer_model_cond[iteration] + alpha*x
                case "qr_decomposition":
                    original_shape = x_step.shape
                    if not original_shape:
                        shape_2d = (1, 1)
                    elif len(original_shape) == 4:
                        shape_2d = (-1, functools.reduce(operator.mul, original_shape[1:]))
                    else:
                        shape_2d = (-1, original_shape[-1])

                    A = x.reshape(shape_2d)
                    B = x_step.reshape(shape_2d)
                    C = buffer_model_cond[iteration].reshape(shape_2d)
                    Q, _ = torch.qr(A - C)
                    # Compute the mapping matrix
                    mapping_matrix = torch.mm(Q.t(), B - C)
                    mapped_tensor = torch.mm(Q, mapping_matrix)
                    x = (C + mapped_tensor).reshape(original_shape)
                case "svd_lowrank":
                    original_shape = x_step.shape
                    if not original_shape:
                        shape_2d = (1, 1)
                    elif len(original_shape) == 4:
                        shape_2d = (-1, functools.reduce(operator.mul, original_shape[1:]))
                    else:
                        shape_2d = (-1, original_shape[-1])

                    A = x.reshape(shape_2d)
                    B = x_step.reshape(shape_2d)
                    C = buffer_model_cond[iteration].reshape(shape_2d)
                    Ua, Sa, Va = torch.svd_lowrank(A - C, q=6, niter=2)
                    
                    A_lowrank = torch.mm(Ua, torch.mm(torch.diag(Sa), Va.t()))
                    A_diff = (A - C) - A_lowrank

                    #Qb, _ = torch.qr(B - C)

                    #A_diff_projected = torch.mm(Qb, torch.mm(Qb.t(), A_diff))
                    
                    x = (B + A_diff).reshape(original_shape)
                case "svd":
                    original_shape = x.shape
                    if not original_shape:
                        shape_2d = (1, 1)
                    elif len(original_shape) == 4:
                        shape_2d = (-1, functools.reduce(operator.mul, original_shape[1:]))
                    else:
                        shape_2d = (-1, original_shape[-1])

                    A = x.reshape(shape_2d)
                    B = x_step.reshape(shape_2d)
                    C = buffer_model_cond[iteration].reshape(shape_2d)
                    Ua, Sa, Va = torch.linalg.svd(A - C, full_matrices=False, driver="gesvd")
                    Ub, Sb, Vb = torch.linalg.svd(B - C, full_matrices=False, driver="gesvd")#Sb = torch.linalg.svdvals(B - C, driver="gesvd")#
                    
                    A_lowrank = torch.mm(Ub, torch.mm(torch.diag_embed(Sa), Vb))
                    A_diff = (A - C) - A_lowrank

                    #Qb, _ = torch.qr(B - C)

                    #A_diff_projected = torch.mm(Ua, torch.mm(Ua.t(), A_diff))
                    
                    x = (B + A_diff).reshape(original_shape)

        if len(buffer_model_cond) == max(order - 1, 1):
            for k in range(order - 2):
                buffer_model_cond[k] = buffer_model_cond[k+1]
                #buffer_model_uncond[k] = buffer_model_uncond[k+1]
                #buffer_model_dt[k] = buffer_model_dt[k+1]
            buffer_model_cond[-1] = denoised.detach()
            #buffer_model_uncond[-1] = temp_uncond[0].detach()
            #buffer_model_dt[-1] = dt.detach()
        else:
            buffer_model_cond.append(denoised.detach())
            #buffer_model_uncond.append(temp_uncond[0].detach())
            #buffer_model_dt.append(dt.detach())
        #if old_uncond is not None and old_cond is not None and order >= 2:
        #    x = x + (old_cond - old_uncond) / (old_dt / dt)
        #if old_uncond_2 is not None and old_cond_2 is not None and order >= 3:
        #    x = x + (old_cond_2 - old_uncond_2) / (old_dt_2 / old_dt) / (old_dt / dt)
        if sigmas[i + 1] > 0 and not flow:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        elif sigmas[i + 1] > 0 and flow:
            x = (alpha_ip1/alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
        #old_uncond, old_uncond_2 = temp[0], old_uncond
        #old_cond, old_cond_2 = temp_cond[0], old_cond
        #old_dt, old_dt_2 = dt, old_dt
    return x

@torch.no_grad()
def sample_SHIDS(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler_type="gaussian", noise_sampler=None, order=16, eta_order=1., solver_method="weighted_projection"):
    if len(sigmas) <= 1:
        return x
    flow = False
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        flow = True
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sampler_SHIDS(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args), order=order, eta_order=eta_order, solver_method=solver_method, flow=flow)

@torch.no_grad()
def sampler_dpmpp_2m_sde_ema(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, amp_fac=2., beta1=0.8, beta2=0.95, weight_decay=0.1, centralization=1.0, normalization=1.0, flow=False):
    """DPM-Solver++(2M) SDE, with EMA uncond."""
    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None

    ema = torch.zeros_like(x)
    ema_squared = torch.zeros_like(x)

    grad = None
    temp_cond = [0]
    temp_uncond = [0]
    #alpha = torch.linspace(1.0, 0.0, steps=len(sigmas)) ** amp_fac
    alpha = [0]
    def ema_retrieve_uncond_alpha(args):
        temp_cond[0] = args["cond_denoised"]
        temp_uncond[0] = args["uncond_denoised"]
        alpha[0] = model.inner_model.inner_model.model_sampling.timestep(args["sigma"]) / 999.0
        #alpha[0] = args["sigma"]
        return args["denoised"]
    
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, ema_retrieve_uncond_alpha, disable_cfg1_optimization=True)

    ema = torch.zeros_like(x)
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            # If/for flow model
            downstep_ratio = 1 + (sigmas[i+1]/sigmas[i] - 1) * eta
            sigma_down = sigmas[i+1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i+1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i+1]**2 - sigma_down**2*alpha_ip1**2/alpha_down**2)**0.5

            grad = denoised

            # Centralization
            if centralization != 0:
                grad.sub_(
                    grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                )
            # Lerp EMA
            ema.lerp_(grad, 1. - beta1)
            # Normalization
            ema.lerp_(ema.div(ema.std(dim=tuple(range(1, grad.dim())), keepdim=True)), weight=normalization)
            # Apply EMA onto grad (denoised)
            grad.lerp_(ema, beta2)

            if weight_decay != 0:
                # Perform stepweight decay
                wd_mult = 1 / (1 + weight_decay * (sigmas[i] - sigmas[i + 1]))
                grad.mul_(wd_mult)

            ema += (x - temp_uncond[0]) / sigmas[i] * amp_fac * (sigmas[i] - sigmas[i + 1])
            ema -= (x - temp_cond[0]) / sigmas[i] * amp_fac * (sigmas[i] - sigmas[i + 1])

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * grad

            if old_denoised is not None:
                r = h_last / h
                x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (grad - old_denoised)
            
            if eta and not flow:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise
            elif eta and flow:
                x = (alpha_ip1/alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff

        old_denoised = denoised
        h_last = h
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde_ema(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler_type="brownian", noise_sampler=None, amp_fac=2., beta1=0.8, beta2=0.95, weight_decay=0.1, centralization=1.0, normalization=1.0):
    if len(sigmas) <= 1:
        return x
    flow = False
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        flow = True
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sampler_dpmpp_2m_sde_ema(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args), amp_fac=amp_fac, beta1=beta1, beta2=beta2, weight_decay=weight_decay, centralization=centralization, normalization=normalization, flow=flow)

@torch.no_grad()
def sampler_biscope(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, amp_fac=2.0, local_smoothing_fac=4, smoothing_fac=0.75, ema_fac=0.9, flow=False):
    """Solving for the Compass model's noise problem using the Compass-like training procedure as an inference sampler."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    local_smoothing = []
    smoothing = None
    smoothing_diff = None
    ema = None
    prev_denoised = None
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        grad = denoised

        if len(local_smoothing) == max(local_smoothing_fac, 1):
            for k in range(local_smoothing_fac - 1):
                local_smoothing[k] = local_smoothing[k+1]
            local_smoothing[-1] = grad.detach()
        else:
            local_smoothing.append(grad.detach())
        #print(local_smoothing)
        local_grad = torch.mean(torch.stack(local_smoothing), dim=0)# if len(local_smoothing) > 1 else grad

        if smoothing is None:
            smoothing = local_grad

        smoothing.mul_(smoothing_fac).add_(local_grad, alpha=1 - smoothing_fac)

        diff_grad = local_grad - smoothing

        if smoothing_diff is None:
            smoothing_diff = diff_grad
        smoothing_diff.mul_(smoothing_fac).add_(diff_grad, alpha=1 - smoothing_fac)

        local_grad.add_(smoothing_diff, alpha=amp_fac)

        if ema is None:
            ema = local_grad
        ema.mul_(ema_fac).add_(local_grad, alpha=1 - ema_fac)

        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        # Flow
        downstep_ratio = None
        alpha_ip1 = None
        alpha_down = None
        renoise_coeff = None
        if flow:
            # If/for flow model
            downstep_ratio = 1 + (sigmas[i+1]/sigmas[i] - 1) * eta
            sigma_down = sigmas[i+1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i+1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i+1]**2 - sigma_down**2*alpha_ip1**2/alpha_down**2)**0.5
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': ema})
        d = to_d(x, sigmas[i], ema)
        
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0 and not flow:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        elif sigmas[i + 1] > 0 and flow:
            x = (alpha_ip1/alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff

        prev_denoised = denoised
    return x

@torch.no_grad()
def sample_biscope(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler_type="gaussian", noise_sampler=None, amp_fac=2.0, local_smoothing_fac=4, smoothing_fac=0.5, ema_fac=0.5):
    if len(sigmas) <= 1:
        return x
    flow = False
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        flow = True
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sampler_biscope(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args), amp_fac=amp_fac, local_smoothing_fac=local_smoothing_fac, smoothing_fac=smoothing_fac, ema_fac=ema_fac, flow=flow)

def gaussian_kernel_2d(kernel_size, sigma):
    """Generates a 2D Gaussian kernel."""
    k = kernel_size // 2
    x, y = torch.meshgrid(torch.arange(-k, k + 1, dtype=torch.float32), torch.arange(-k, k + 1, dtype=torch.float32))
    gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return gaussian / gaussian.sum()


@torch.no_grad()
def sampler_euler_g(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, g_eta=1.0, sigma=5.0, order=2, flow=False):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    neighborhood_size = min(x.shape[-2], x.shape[-1]) * 2 + 1
    padding = neighborhood_size // 2
    kernel = gaussian_kernel_2d(neighborhood_size, sigma).unsqueeze(0).unsqueeze(0).repeat(x.shape[1], 1, 1, 1).to(x.device)

    x_buffer = []
    denoised_buffer = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if sigmas[i + 1] == 0:
            return denoised
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        # Flow
        downstep_ratio = None
        alpha_ip1 = None
        alpha_down = None
        renoise_coeff = None
        if flow:
            # If/for flow model
            downstep_ratio = 1 + (sigmas[i+1]/sigmas[i] - 1) * eta
            sigma_down = sigmas[i+1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i+1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i+1]**2 - sigma_down**2*alpha_ip1**2/alpha_down**2)**0.5
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt

        """
        for curr_order in range(1, order):
            if sigmas[i + 1] > 0 and not flow:
                faux_x = torch.nn.functional.conv2d(x, kernel, padding=padding, groups=x.shape[1]) + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (sigma_up)
            elif sigmas[i + 1] > 0 and flow:
                #x = (alpha_ip1/alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
                #faux_x = (alpha_ip1/(1 - sigmas[i])) * torch.nn.functional.conv2d(x, kernel, padding=padding, groups=x.shape[1]) + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (sigmas[i+1]**2 - sigmas[i]**2*alpha_ip1**2/(1 - sigmas[i])**2)**0.5
                faux_x = (alpha_ip1/alpha_down) * torch.nn.functional.conv2d(x, kernel, padding=padding, groups=x.shape[1]) + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
            #gauss_x = F.conv2d(faux_x, kernel, padding=padding, groups=x.shape[1])
            faux_denoised = model(faux_x, sigmas[i + 1] * s_in, **extra_args)
            faux_d = to_d(faux_x, sigmas[i + 1], faux_denoised)
            x = x - faux_d * (sigmas[i + 1] - sigmas[i]) * g_eta / (order - 1)
        """

        if sigmas[i + 1] > 0:
            # Create a list of order multipliers
            multipliers = [i for i in range(1, len(x_buffer))]
            # Normalize so that they're summed up to a total of 1
            total = sum(multipliers)
            normalized_multipliers = [m / total for m in multipliers]

            for iteration in range(len(x_buffer) - 1):
                if not flow:
                    faux_x = torch.nn.functional.conv2d(x_buffer[iteration], kernel, padding=padding, groups=x.shape[1]) + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
                else:
                    #x = (alpha_ip1/alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
                    #faux_x = (alpha_ip1/(1 - sigmas[i])) * torch.nn.functional.conv2d(x, kernel, padding=padding, groups=x.shape[1]) + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (sigmas[i+1]**2 - sigmas[i]**2*alpha_ip1**2/(1 - sigmas[i])**2)**0.5
                    faux_x = (alpha_ip1/alpha_down) * torch.nn.functional.conv2d(x_buffer[iteration], kernel, padding=padding, groups=x.shape[1]) + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
                #gauss_x = F.conv2d(faux_x, kernel, padding=padding, groups=x.shape[1])
                faux_denoised = model(faux_x, sigmas[i + 1] * s_in, **extra_args)
                faux_d = to_d(faux_x, sigmas[i + 1], faux_denoised)
                x = x - faux_d * (sigmas[i + 1] - sigmas[i]) * g_eta * normalized_multipliers[iteration]

            if len(x_buffer) == max(order - 1, 1):
                for k in range(order - 2):
                    x_buffer[k] = x_buffer[k+1]
                    denoised_buffer[k] = denoised_buffer[k+1]
                x_buffer[-1] = x.detach()
                denoised_buffer[-1] = denoised.detach()
            else:
                x_buffer.append(x.detach())
                denoised_buffer.append(denoised.detach())

            if flow:
                x = (alpha_ip1/alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
            else:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_euler_g(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler_type="gaussian", noise_sampler=None, g_eta=1.0, sigma=5.0, order=2):
    if len(sigmas) <= 1:
        return x
    flow = False
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        flow = True
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sampler_euler_g(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args), g_eta=g_eta, sigma=sigma, order=order, flow=flow)

@torch.no_grad()
def sampler_leaping_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, leap=1, eta=1., s_noise=1., noise_sampler=None, flow=False):
    #if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
    #    return sample_euler_ancestral_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler)
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        do_dance = i < (len(sigmas) - (2 + leap))
        if not do_dance:
            leap -= 1
            do_dance = True

        sigma_next = sigmas[i + (1 + leap)] if do_dance else sigmas[i + 1]
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        # Flow
        downstep_ratio = None
        alpha_ip1 = None
        alpha_down = None
        renoise_coeff = None
        if flow:
            # If/for flow model
            downstep_ratio = 1 + (sigmas[i+1]/sigmas[i] - 1) * eta
            sigma_down = sigmas[i+1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i+1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i+1]**2 - sigma_down**2*alpha_ip1**2/alpha_down**2)**0.5

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_next - sigmas[i]
        x_2 = x + d * dt
        
        if do_dance:
            reverse_denoised = model(x_2, sigma_next * s_in, **extra_args)
            _, r_sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            r_d = to_d(x_2, sigma_next, reverse_denoised)
            r_dt = sigma_down - sigma_next
            x_2 = x + d * dt + r_d * r_dt
            if sigmas[i + 1] > 0 and not flow:
                x_2 = x_2 + noise_sampler(sigma_next, sigmas[i+1]) * s_noise * sigma_up
            elif flow:
                x_2 = (alpha_ip1/alpha_down) * x_2 + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
        x = x_2

    return x

@torch.no_grad()
def sample_leaping_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, leap=1, eta=1., s_noise=1., noise_sampler_type="gaussian", noise_sampler=None):
    if len(sigmas) <= 1:
        return x
    flow = False
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        flow = True
    noise_sampler, extra_args = check_set_immiscible(x, noise_sampler_type, extra_args)
    return sampler_leaping_euler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, leap=leap, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler if noise_sampler is not None else get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args), flow=flow)

# Add your personal samplers below here, just for formatting purposes ;3

# Add any extra samplers to the following dictionary
extra_samplers = {
    "res_momentumized": sample_res_solver,
    "dpmpp_dualsde_momentumized": sample_dpmpp_dualsdemomentum,
    "clyb_4m_sde_momentumized": sample_clyb_4m_sde,
    "ttm": sample_ttmcustom,
    "lcm_custom_noise": sample_lcmcustom,
    "euler_ancestral_dancing": sample_euler_ancestral_dancing,
    "dpmpp_3m_sde_dynamic_eta": sample_dpmpp_3m_sde_dynamic_eta,
    "supreme": sample_supreme,
    "sens": sample_sens,
    "ipndm_vapp": sample_ipndm_vapp,
    "SHIDS": sample_SHIDS,
    "dpmpp_2m_sde_ema": sample_dpmpp_2m_sde_ema,
    "biscope": sample_biscope,
    "euler_g": sample_euler_g,
    "leaping_euler": sample_leaping_euler,
}

discard_penultimate_sigma_samplers = set((
    "dpmpp_dualsde_momentumized",
    "clyb_4m_sde_momentumized"
))

def get_sigmas_simple_exponential(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    sigs = torch.FloatTensor(sigs)
    exp = torch.exp(torch.log(torch.linspace(1, 0, steps + 1)))
    return sigs * exp

def get_sigmas_kl_optimal(model_sampling, steps):
    s = model_sampling
    sigs = []
    alpha_min = torch.arctan(s.sigma_min).item()
    alpha_max = torch.arctan(s.sigma_max).item()
    for x in range(steps+1):
        sigs += [torch.tan(torch.tensor(((x/steps) * alpha_min + (1.0-x/steps) * alpha_max)))]
    return torch.FloatTensor(sigs)

def get_sigmas_simple_kl_optimal(model_sampling, steps):
    s = model_sampling
    sigs = []
    idx_list = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        step = (x/steps) * math.atan(len(s.sigmas) / steps) + (x/steps) * math.atan(1 / steps)
        idx = int(len(s.sigmas) * (1.0 - math.atan(step))) - 1
        idx_list += [idx]
        sigs += [float(s.sigmas[idx])]
    #print(idx_list)
    sigs += [0.0]
    return torch.FloatTensor(sigs)

extra_schedulers = {
    "simple_exponential": get_sigmas_simple_exponential,
    "kl_optimal": get_sigmas_kl_optimal,
    "simple_kl_optimal": get_sigmas_simple_kl_optimal,
}
