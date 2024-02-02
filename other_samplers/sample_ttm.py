from k_diffusion.sampling import default_noise_sampler, to_d
from tqdm import trange
import torch
from torch import enable_grad

# by Katherine Crowson
@enable_grad()
def sample_ttm(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    import torch.autograd.forward_ad as fwAD

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
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

            with fwAD.dual_level():
                eps = to_d(x, sigmas[i], denoised)
                dual_x = fwAD.make_dual(x, eps * -sigmas[i])
                dual_sigma = fwAD.make_dual(sigmas[i] * s_in, -sigmas[i] * s_in)
                dual_denoised = model(dual_x, dual_sigma, **extra_args)
                denoised_prime = fwAD.unpack_dual(dual_denoised).tangent

        phi_1 = -torch.expm1(-h_eta)
        phi_2 = torch.expm1(-h_eta) + h_eta
        x = torch.exp(-h_eta) * x + phi_1 * denoised + phi_2 * denoised_prime

        if eta:
            phi_1_noise = torch.sqrt(-torch.expm1(-2 * h * eta))
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * phi_1_noise * s_noise

    return x

# by Katherine Crowson
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
            phi_2 = torch.expm1(-h_eta) + h_eta
            # phi_2 = torch.expm1(-h) + h # seems to work better with eta > 0
            x = torch.exp(-h_eta) * x + phi_1 * denoised + phi_2 * denoised_prime

            if eta:
                phi_1_noise = torch.sqrt(-torch.expm1(-2 * h * eta))
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * phi_1_noise * s_noise

    return x

@torch.no_grad()
def sample_lcm_ttm_jvp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, eta=0.0):
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

            x = denoised
            x_2 = x + sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])

            eps = to_d(x_2, sigmas[i + 1], denoised)
            _, denoised_prime = torch.func.jvp(model_fn, (x_2, sigmas[i + 1]), (eps * -sigmas[i + 1], -sigmas[i + 1]))

            phi_1 = -torch.expm1(-h_eta)
            phi_2 = torch.expm1(-h_eta) + h_eta
            # phi_2 = torch.expm1(-h) + h # seems to work better with eta > 0
            x = torch.exp(-h_eta) * x + phi_1 * denoised + phi_2 * denoised_prime

        if sigmas[i + 1] > 0:
            x = x + sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])

    return x