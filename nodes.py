from .other_samplers.refined_exp_solver import sample_refined_exp_s
from .extra_samplers import get_noise_sampler_names, prepare_noise

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
from comfy.k_diffusion import sampling as k_diffusion_sampling
import node_helpers

import latent_preview
import torch
import math
from tqdm.auto import trange

import kornia

class SamplerRES_MOMENTUMIZED:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise_sampler_type": (get_noise_sampler_names(), ),
                     "momentum": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step":0.01}),
                     "denoise_to_zero": ("BOOLEAN", {"default": True}),
                     "simple_phi_calc": ("BOOLEAN", {"default": False}),
                     "ita": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "c2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, noise_sampler_type, momentum, denoise_to_zero, simple_phi_calc, ita, c2):
        sampler = comfy.samplers.ksampler("res_momentumized", {"noise_sampler_type": noise_sampler_type, "denoise_to_zero": denoise_to_zero, "simple_phi_calc": simple_phi_calc, "c2": c2, "ita": torch.Tensor((ita,)), "momentum": momentum})
        return (sampler, )

class SamplerDPMPP_DUALSDE_MOMENTUMIZED:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise_sampler_type": (get_noise_sampler_names(), ),
                     "momentum": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step":0.01}),
                     "eta": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step":0.01}),
                     "s_noise": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step":0.01}),
                     "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, noise_sampler_type, momentum, eta, s_noise, r,):
        sampler = comfy.samplers.ksampler("dpmpp_dualsde_momentumized", {"noise_sampler_type": noise_sampler_type, "eta": eta, "s_noise": s_noise, "r": r, "momentum": momentum})
        return (sampler, )

class SamplerTTM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise_sampler_type": (get_noise_sampler_names(), ),
                     "eta": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step":0.01}),
                     "s_noise": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step":0.01}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, noise_sampler_type, eta, s_noise):
        sampler = comfy.samplers.ksampler("ttm", {"noise_sampler_type": noise_sampler_type, "eta": eta, "s_noise": s_noise})
        return (sampler, )


class SamplerLCMCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise_sampler_type": (get_noise_sampler_names(), ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, noise_sampler_type):
        sampler = comfy.samplers.ksampler("lcm_custom_noise", {"noise_sampler_type": noise_sampler_type})
        return (sampler, )

class SamplerCLYB_4M_SDE_MOMENTUMIZED:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise_sampler_type": (get_noise_sampler_names(default="brownian"), ),
                     "momentum": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step":0.01}),
                     "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01}),
                     "s_noise": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step":0.01}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, noise_sampler_type, eta, s_noise, momentum):
        sampler = comfy.samplers.ksampler("clyb_4m_sde_momentumized", {"noise_sampler_type": noise_sampler_type, "eta": eta, "s_noise": s_noise, "momentum": momentum})
        return (sampler, )

class SamplerEULER_ANCESTRAL_DANCING:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise_sampler_type": (get_noise_sampler_names(), ),
                     "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01}),
                     "eta_dance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01}),
                     "s_noise": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step":0.01}),
                     "leap": ("INT", {"default": 2, "min": 1, "max": 16, "step":1}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, noise_sampler_type, eta, s_noise, leap, eta_dance):
        sampler = comfy.samplers.ksampler("euler_ancestral_dancing", {"noise_sampler_type": noise_sampler_type, "eta": eta, "s_noise": s_noise, "leap": leap, "eta_dance": eta_dance})
        return (sampler, )

class SamplerDPMPP_3M_SDE_DYN_ETA:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise_sampler_type": (get_noise_sampler_names(default="brownian"), ),
                     "eta_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01}),
                     "eta_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step":0.01}),
                     "s_noise": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step":0.01}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, noise_sampler_type, eta_max, eta_min, s_noise):
        sampler = comfy.samplers.ksampler("dpmpp_3m_sde_dynamic_eta", {"noise_sampler_type": noise_sampler_type, "eta_max": eta_max, "eta_min": eta_min, "s_noise": s_noise})
        return (sampler, )

class SamplerSUPREME:
    @classmethod
    def INPUT_TYPES(s):
        SUBSTEP_METHODS=["euler", "dpm_1s", "dpm_2s", "dpm_3s", "bogacki_shampine", "rk4", "rkf45", "reversible_heun", "reversible_heun_1s", "reversible_bogacki_shampine", "trapezoidal", "RES"]
        STEP_METHODS=SUBSTEP_METHODS+["dynamic", "adaptive_rk"]
        NOISE_MODULATION_TYPES=["none", "intensity", "frequency", "spectral_signum"]
        return {"required":
                    {"noise_sampler_type": (get_noise_sampler_names(),),
                     "step_method": (STEP_METHODS, ),
                     "substep_method": (SUBSTEP_METHODS, ),
                     "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01}),
                     "centralization": ("FLOAT", {"default": 0.02, "min": -1.0, "max": 1.0, "step":0.01}),
                     "normalization": ("FLOAT", {"default": 0.01, "min": -1.0, "max": 1.0, "step":0.01}),
                     "edge_enhancement": ("FLOAT", {"default": 0.05, "min": -100.0, "max": 100.0, "step":0.01}),
                     "perphist": ("FLOAT", {"default": 0, "min": -5.0, "max": 5.0, "step":0.01}),
                     "substeps": ("INT", {"default": 2, "min": 1, "max": 100, "step":1}),
                     "s_noise": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step":0.01}),
                     "noise_modulation": (NOISE_MODULATION_TYPES, {"default": "intensity"}),
                     "modulation_strength": ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step":0.01}),
                     "modulation_dims": ("INT", {"default": 3, "min": 1, "max": 3, "step":1}),
                     "reversible_eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, noise_sampler_type, step_method, substep_method, eta, centralization, normalization, edge_enhancement, perphist, substeps, noise_modulation, modulation_strength, modulation_dims, reversible_eta, s_noise):
        sampler = comfy.samplers.ksampler("supreme", {"noise_sampler_type": noise_sampler_type, "step_method": step_method, "eta": eta, "centralization": centralization, "normalization": normalization, "edge_enhancement": edge_enhancement, "perphist": perphist, "substeps": substeps, "substep_method": substep_method, "noise_modulation": noise_modulation, "modulation_strength": modulation_strength, "modulation_dims": modulation_dims, "reversible_eta": reversible_eta, "s_noise": s_noise})
        return (sampler, )

### Schedulers
from .extra_samplers import get_sigmas_simple_exponential
class SimpleExponentialScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "clybNodes/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            total_steps = int(steps/denoise)

        sigmas = get_sigmas_simple_exponential(model.model, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]
        return (sigmas, )

### KSampler Nodes

from comfy import model_management
import comfy.utils
import comfy.conds
from comfy.sampler_helpers import prepare_sampling, cleanup_additional_models, get_models_from_cond

def mixture_sample(model, model2, noise, positive, positive2, negative, negative2, cfg, cfg2, device, device2, sampler, sampler2, sigmas, sigmas2, model_options={}, model_options2={}, latent_image=None, denoise_mask=None, denoise_mask2=None, callback=None, callback2=None, disable_pbar=False, seed=None):
    positive = positive[:]
    negative = negative[:]
    positive2 = positive2[:]
    negative2 = negative2[:]

    comfy.samplers.resolve_areas_and_cond_masks(positive, noise.shape[2], noise.shape[3], device)
    comfy.samplers.resolve_areas_and_cond_masks(negative, noise.shape[2], noise.shape[3], device)
    comfy.samplers.resolve_areas_and_cond_masks(positive2, noise.shape[2], noise.shape[3], device2)
    comfy.samplers.resolve_areas_and_cond_masks(negative2, noise.shape[2], noise.shape[3], device2)

    model_wrap = comfy.samplers.wrap_model(model)
    model_wrap2 = comfy.samplers.wrap_model(model2)

    comfy.samplers.calculate_start_end_timesteps(model, negative)
    comfy.samplers.calculate_start_end_timesteps(model, positive)
    comfy.samplers.calculate_start_end_timesteps(model2, negative2)
    comfy.samplers.calculate_start_end_timesteps(model2, positive2)

    if latent_image is not None:
        latent_image = model.process_latent_in(latent_image)

    if hasattr(model, 'extra_conds'):
        positive = comfy.samplers.encode_model_conds(model.extra_conds, positive, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)
        negative = comfy.samplers.encode_model_conds(model.extra_conds, negative, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)
    if hasattr(model2, 'extra_conds'):
        positive = comfy.samplers.encode_model_conds(model2.extra_conds, positive2, noise, device2, "positive", latent_image=latent_image, denoise_mask=denoise_mask2, seed=seed)
        negative = comfy.samplers.encode_model_conds(model2.extra_conds, negative2, noise, device2, "negative", latent_image=latent_image, denoise_mask=denoise_mask2, seed=seed)

    #make sure each cond area has an opposite one with the same area
    for c in positive:
        comfy.samplers.create_cond_with_same_area_if_none(negative, c)
    for c in negative:
        comfy.samplers.create_cond_with_same_area_if_none(positive, c)
    for c in positive2:
        comfy.samplers.create_cond_with_same_area_if_none(negative2, c)
    for c in negative2:
        comfy.samplers.create_cond_with_same_area_if_none(positive2, c)

    comfy.samplers.pre_run_control(model, negative + positive)
    comfy.samplers.pre_run_control(model2, negative2 + positive2)

    comfy.samplers.apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), negative, 'control', lambda cond_cnets, x: cond_cnets[x])
    comfy.samplers.apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

    comfy.samplers.apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive2)), negative2, 'control', lambda cond_cnets, x: cond_cnets[x])
    comfy.samplers.apply_empty_x_to_equal_area(positive2, negative2, 'gligen', lambda cond_cnets, x: cond_cnets[x])

    extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg, "model_options": model_options, "seed":seed}
    extra_args2 = {"cond":positive2, "uncond":negative2, "cond_scale": cfg, "model_options": model_options2, "seed":seed}
    samples = None
    temp_sigmas = sigmas
    temp_sigmas2 = sigmas2
    #samples = sampler.sample(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, True)
    for i in trange(len(sigmas) - 1, disable=disable_pbar):
        last_step = i + 1
        start_step = i
        if last_step is not None and last_step < (len(sigmas) - 1):
            temp_sigmas = sigmas[:last_step + 1]
            temp_sigmas2 = sigmas2[:last_step + 1]
        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                temp_sigmas = temp_sigmas[start_step:]
                temp_sigmas2 = temp_sigmas2[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)
        if len(temp_sigmas) != 2:
            temp_sigmas = sigmas[-2:]
            temp_sigmas2 = sigmas2[-2:]
        if (i % 2) == 0:
            #print(temp_sigmas)
            samples = sampler.sample(model_wrap, temp_sigmas, extra_args, callback, noise.to(device) if i == 0 else torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=device), samples if samples is not None else latent_image, denoise_mask, True)
        else:
            #print(temp_sigmas)
            samples = sampler2.sample(model_wrap2, temp_sigmas2, extra_args2, callback2, noise.to(device2) if i == 0 else torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=device2), samples if samples is not None else latent_image, denoise_mask2, True)
    return model.process_latent_out(samples.to(torch.float32))

def sample_mixture(model, model2, noise, cfg, cfg2, sampler, sampler2, sigmas, sigmas2, positive, negative, latent_image, noise_mask=None, callback=None, callback2=None, disable_pbar=False, seed=None):
    real_model, positive_copy, negative_copy, noise_mask, models = prepare_sampling(model, noise.shape, positive, negative, noise_mask)
    real_model2, positive_copy2, negative_copy2, noise_mask2, models2 = prepare_sampling(model2, noise.shape, positive, negative, noise_mask)
    noise = noise.to(model.load_device)
    latent_image = latent_image.to(model.load_device)
    sigmas = sigmas.to(model.load_device)
    sigmas2 = sigmas2.to(model.load_device)

    samples = mixture_sample(real_model, real_model2, noise, positive_copy, positive_copy2, negative_copy, negative_copy2, cfg, cfg2, model.load_device, model2.load_device, sampler, sampler2, sigmas, sigmas2, model_options=model.model_options, model_options2=model2.model_options, latent_image=latent_image, denoise_mask=noise_mask, denoise_mask2=noise_mask2, callback=callback, callback2=callback2, disable_pbar=disable_pbar, seed=seed)

    samples = samples.to(comfy.model_management.intermediate_device())
    cleanup_additional_models(models)
    cleanup_additional_models(models2)
    cleanup_additional_models(set(get_models_from_cond(positive_copy, "control") + get_models_from_cond(negative_copy, "control")))
    cleanup_additional_models(set(get_models_from_cond(positive_copy2, "control") + get_models_from_cond(negative_copy2, "control")))
    return samples

class SamplerCustomNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "noise_is_latent": ("BOOLEAN", {"default": False}),
                    "noise_type": (["gaussian", "uniform", "pyramid", "power"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, add_noise, noise_is_latent, noise_type, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        if not add_noise:
            torch.manual_seed(noise_seed)
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = prepare_noise(latent_image, noise_seed, noise_type, batch_inds)

        if noise_is_latent:
            noise += latent_image.cpu()# * noise.std()
            noise.sub_(noise.mean()).div_(noise.std())

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}

        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = False
        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)

class SamplerCustomNoiseDuo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "add_noise_pass2": ("BOOLEAN", {"default": True}),
                    "return_noisy_pass1": ("BOOLEAN", {"default": False}),
                    "noise_type": (["gaussian", "uniform", "pyramid", "power"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "cfg2": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "sampler2": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "sigmas2": ("SIGMAS", ),
                    "hr_upscale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 9.0, "step":0.1, "round": 0.01}),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, add_noise, add_noise_pass2, return_noisy_pass1, noise_type, noise_seed, cfg, cfg2, positive, negative, sampler, sampler2, sigmas, sigmas2, hr_upscale, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        if not add_noise:
            torch.manual_seed(noise_seed)
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = prepare_noise(latent_image, noise_seed, noise_type, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}

        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = False

        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

        if not return_noisy_pass1:
            out_denoised = latent.copy()
            samples = model.model.process_latent_out(x0_output["x0"].cpu())
        if hr_upscale > 1.0:
            if "noise_mask" in latent:
                noise_mask = comfy.utils.common_upscale(noise_mask, (int)(noise_mask.shape[-1] * hr_upscale), (int)(noise_mask.shape[-2] * hr_upscale), "bislerp", "disabled")
            samples = comfy.utils.common_upscale(samples, (int)(samples.shape[-1] * hr_upscale), (int)(samples.shape[-2] * hr_upscale), "bislerp", "disabled")
            noise = prepare_noise(samples, noise_seed, noise_type, batch_inds)

        samples = comfy.sample.sample_custom(model, noise if add_noise_pass2 else torch.zeros(samples.size(), dtype=samples.dtype, layout=samples.layout, device="cpu"), cfg2, sampler2, sigmas2, positive, negative, samples, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)

class SamplerCustomModelMixtureDuo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "model2": ("MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "add_noise_pass2": ("BOOLEAN", {"default": True}),
                    "return_noisy_pass1": ("BOOLEAN", {"default": False}),
                    "noise_type": (["gaussian", "uniform", "pyramid", "power"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "cfg2": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "sampler2": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "sigmas2": ("SIGMAS", ),
                    "hr_upscale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 9.0, "step":0.1, "round": 0.01}),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, model2, add_noise, add_noise_pass2, return_noisy_pass1, noise_type, noise_seed, cfg, cfg2, positive, negative, sampler, sampler2, sigmas, sigmas2, hr_upscale, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        if not add_noise:
            torch.manual_seed(noise_seed)
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = prepare_noise(latent_image, noise_seed, noise_type, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}

        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
        callback2 = latent_preview.prepare_callback(model2, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = False

        samples = sample_mixture(model, model2, noise, cfg, cfg2, sampler, sampler2, sigmas, sigmas2, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, callback2=callback2, disable_pbar=disable_pbar, seed=noise_seed)

        #if not return_noisy_pass1:
        #    out_denoised = latent.copy()
        #    samples = model.model.process_latent_out(x0_output["x0"].cpu())
        #if hr_upscale > 1.0:
        #    if "noise_mask" in latent:
        #        noise_mask = comfy.utils.common_upscale(noise_mask, (int)(noise_mask.shape[-1] * hr_upscale), (int)(noise_mask.shape[-2] * hr_upscale), "bislerp", "disabled")
        #    samples = comfy.utils.common_upscale(samples, (int)(samples.shape[-1] * hr_upscale), (int)(samples.shape[-2] * hr_upscale), "bislerp", "disabled")
        #    noise = prepare_noise(samples, noise_seed, noise_type, batch_inds)

        #samples = sample_mixture(model, model2, noise if add_noise_pass2 else torch.zeros(samples.size(), dtype=samples.dtype, layout=samples.layout, device="cpu"), cfg2, sampler2, sigmas2, positive, negative, samples, noise_mask=noise_mask, callback=callback, callback2=callback2, disable_pbar=disable_pbar, seed=noise_seed)

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)

class Guider_GeometricCFG(comfy.samplers.CFGGuider):
    def set_cfg(self, cfg1, geometric_alpha):
        self.cfg1 = cfg1
        self.alpha = geometric_alpha

    def set_conds(self, positive, positive2, negative):
        self.inner_set_conds({"positive": positive, "positive2": positive2, "negative": negative})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        negative_cond = self.conds.get("negative", None)
        positive_cond = self.conds.get("positive", None)
        positive2_cond = self.conds.get("positive2", None)

        out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, positive2_cond, positive_cond], x, timestep, model_options) # negative, positive2, positive

        a = torch.complex(out[2], torch.zeros_like(out[2]))
        b = torch.complex(out[1], torch.zeros_like(out[1]))
        res = a ** (1 - self.alpha) * b ** self.alpha
        res = res.real

        return comfy.samplers.cfg_function(self.inner_model, res, out[0], self.cfg1, x, timestep, model_options=model_options, cond=positive_cond, uncond=negative_cond)

class GeometricCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "cond1": ("CONDITIONING", ),
                    "cond2": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "geometric_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, cond1, cond2, negative, cfg, geometric_alpha):
        guider = Guider_GeometricCFG(model)
        guider.set_conds(cond1, cond2, negative) # Conds
        guider.set_cfg(cfg, geometric_alpha) # Strengths
        return (guider,)

class Guider_ImageGuidedCFG(comfy.samplers.CFGGuider):
    def set_cfg(self, model, cfg1, image_cfg, latent_img, img_weighting, weight_scaling):
        self.cfg1 = cfg1
        self.icfg = image_cfg
        self.img = latent_img
        self.img_weighting = img_weighting
        self.weight_scaling = weight_scaling
        self.model = model

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        negative_cond = self.conds.get("negative", None)
        positive_cond = self.conds.get("positive", None)

        out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, positive_cond], x, timestep, model_options)

        img = self.img["samples"].to(out[1].device)

        norm_out1 = torch.linalg.norm(out[1]) # Get norm of positive cond

        res = img - out[1] * (out[1] / norm_out1 * (img / norm_out1)).sum() # Project positive cond onto image
        res *= torch.linalg.norm(out[1]) / torch.linalg.norm(res) # Normalize to cond
        res = self.model.model.model_sampling.calculate_denoised(timestep, res, out[1])

        weight = 1.0 # Flat by default
        match self.img_weighting:
            case "flat":
                weight = 1.0
            case "linear down":
                weight = (timestep / self.model.model.model_sampling.sigma_max)[:, None, None, None].clone()
            case "cosine down":
                weight = ((-torch.cos(timestep / self.model.model.model_sampling.sigma_max * math.pi) / 2) + 0.5)[:, None, None, None].clone()

        cfg = comfy.samplers.cfg_function(self.inner_model, out[1], out[0], self.cfg1, x, timestep, model_options=model_options, cond=positive_cond, uncond=negative_cond)

        return cfg + (cfg - res) * self.icfg * (weight**self.weight_scaling) # Divide by 10 to mimic user-cfg. Do CFG - Res since the image is inverted the other way around.

class ImageGuidedCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "image_cfg": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": 0.001}),
                    "image_weighting": (["flat", "linear down", "cosine down"], ),
                    "weight_scaling": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step":0.01, "round": 0.001}),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg, image_cfg, image_weighting, weight_scaling, latent_image):
        guider = Guider_ImageGuidedCFG(model)
        guider.set_conds(positive, negative) # Conds
        guider.set_cfg(model, cfg, image_cfg, latent_image, image_weighting, weight_scaling) # Strengths
        return (guider,)

class Guider_ScaledCFG(comfy.samplers.CFGGuider):
    def set_cfg(self, cfg1, cond2_alpha):
        self.cfg1 = cfg1
        self.alpha = cond2_alpha

    def set_conds(self, positive, positive2, negative):
        self.inner_set_conds({"positive": positive, "positive2": positive2, "negative": negative})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        negative_cond = self.conds.get("negative", None)
        positive_cond = self.conds.get("positive", None)
        positive2_cond = self.conds.get("positive2", None)

        out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, positive2_cond, positive_cond], x, timestep, model_options) # negative, positive2, positive

        threshold = torch.maximum(torch.abs(out[2] - out[0]), torch.abs(out[1] - out[0]))
        dissimilarity = torch.clamp(torch.nan_to_num((out[0] - out[2]) * (out[1] - out[0]) / threshold**2, nan=0), 0)

        res = out[2] + (out[1] - out[0]) * self.alpha * dissimilarity

        cfg = comfy.samplers.cfg_function(self.inner_model, res, out[0], self.cfg1, x, timestep, model_options=model_options, cond=positive_cond, uncond=negative_cond)
        return cfg

class ScaledCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "cond1": ("CONDITIONING", ),
                    "cond2": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "cond2_alpha": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": 0.01}),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, cond1, cond2, negative, cfg, cond2_alpha):
        guider = Guider_ScaledCFG(model)
        guider.set_conds(cond1, cond2, negative) # Conds
        guider.set_cfg(cfg, cond2_alpha) # Strengths
        return (guider,)

class Guider_WarmupDecayCFG(comfy.samplers.CFGGuider):
    def set_cfg(self, model, cfg_max, cfg_min, warmup_percent):
        self.model = model
        self.cfg_max = cfg_max
        self.cfg_min = cfg_min
        self.warmup_percent = warmup_percent

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})


    def predict_noise(self, x, timestep, model_options={}, seed=None):
        negative_cond = self.conds.get("negative", None)
        positive_cond = self.conds.get("positive", None)

        out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, positive_cond], x, timestep, model_options) # negative, positive2, positive

        sigma_max = self.model.model.model_sampling.sigma_max # 120
        percent_sigma = self.model.model.model_sampling.percent_to_sigma(self.warmup_percent) # 30

        if timestep > percent_sigma:
            decay = (sigma_max - timestep) / (sigma_max - percent_sigma) # (1.0 - (120 - 110) / (120 - 90))
            cfg_scale = 1/2 * (self.cfg_max - self.cfg_min)
            cfg_cos = (1 + torch.cos((timestep / sigma_max) * math.pi))
            mod_cfg = cfg_scale * cfg_cos * decay + self.cfg_min
        else:
            cfg_scale = 1/2 * (self.cfg_max - self.cfg_min)
            cfg_cos = (1 + -torch.cos((timestep / percent_sigma) * math.pi))
            mod_cfg = cfg_scale * cfg_cos + self.cfg_min

        cfg = comfy.samplers.cfg_function(self.inner_model, out[1], out[0], mod_cfg, x, timestep, model_options=model_options, cond=positive_cond, uncond=negative_cond)
        return cfg

class WarmupDecayCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg_max": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "cfg_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "warmup_percent": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step":0.01, "round": 0.01}),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg_max, cfg_min, warmup_percent):
        guider = Guider_WarmupDecayCFG(model)
        guider.set_conds(positive, negative) # Conds
        guider.set_cfg(model, cfg_max, cfg_min, warmup_percent) # Strengths
        return (guider,)

class Guider_MegaCFG(comfy.samplers.CFGGuider):
    def set_cfg(self, model, cfg_max, cfg_min, warmup_percent, mean_cfg):
        self.model = model
        self.cfg_max = cfg_max
        self.cfg_min = cfg_min
        self.warmup_percent = warmup_percent
        self.mean_cfg = mean_cfg

        self.prev_cond = None
        self.prev_cfg = None

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_img_cfg(self, image_guidance, image_weighting, weight_scaling, latent_image):
        self.image_guidance = image_guidance
        self.image_weighting = image_weighting
        self.weight_scaling = weight_scaling
        self.latent_image = latent_image

    def post_cfg_reference_img(self, args):
        model = args["model"]
        cond_pred = args["cond_denoised"]
        cfg_result = args["denoised"]
        sigma = args["sigma"]

        ref = self.latent_image["samples"].to(cfg_result.device)

        if self.image_guidance == 0:
            return cfg_result

        norm_out1 = torch.linalg.norm(cond_pred) # Get norm of positive cond

        ref = ref - cond_pred * (cond_pred / norm_out1 * (ref / norm_out1)).sum() # Project positive cond onto image
        ref *= torch.linalg.norm(cond_pred) / torch.linalg.norm(ref) # Normalize to cond
        ref = self.model.model.model_sampling.calculate_denoised(sigma, ref, cond_pred)

        sigma_max = self.model.model.model_sampling.sigma_max

        weight = 1.0
        match self.image_weighting:
            case "linear down":
                weight = (sigma / sigma_max)[:, None, None, None].clone()
            case "cosine down":
                weight = ((-torch.cos((sigma / sigma_max) * math.pi) / 2) + 0.5)[:, None, None, None].clone()

        return cfg_result + (cond_pred - ref) * self.image_guidance * (weight**self.weight_scaling)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        negative_cond = self.conds.get("negative", None)
        positive_cond = self.conds.get("positive", None)

        out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, positive_cond], x, timestep, model_options) # negative, positive2, positive

        out0_mean = out[0].mean(dim=(1, 2, 3), keepdim=True)
        out1_mean = out[1].mean(dim=(1, 2, 3), keepdim=True)
        if self.mean_cfg != 0:
            out[0] -= out0_mean
            out[1] -= out1_mean

        sigma_max = self.model.model.model_sampling.sigma_max # 120
        percent_sigma = self.model.model.model_sampling.percent_to_sigma(self.warmup_percent) # 30

        if timestep > percent_sigma:
            decay = (sigma_max - timestep) / (sigma_max - percent_sigma) # (1.0 - (120 - 110) / (120 - 90))
            cfg_scale = 1/2 * (self.cfg_max - self.cfg_min)
            cfg_cos = (1 + torch.cos((timestep / sigma_max) * math.pi))
            mod_cfg = cfg_scale * cfg_cos * decay + self.cfg_min
        else:
            cfg_scale = 1/2 * (self.cfg_max - self.cfg_min)
            cfg_cos = (1 + -torch.cos((timestep / percent_sigma) * math.pi))
            mod_cfg = cfg_scale * cfg_cos + self.cfg_min

        cfg = comfy.samplers.cfg_function(self.inner_model, out[1], out[0], mod_cfg, x, timestep, model_options=model_options, cond=positive_cond, uncond=negative_cond)

        if self.mean_cfg != 0:
            cfg += out0_mean + (out1_mean - out0_mean) * self.mean_cfg

        self.prev_cfg = cfg
        self.prev_cond = out[1]

        return cfg

class MegaCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg_max": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "cfg_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "warmup_percent": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step":0.01, "round": 0.001}),
                    "mean_cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                     },
                "optional":
                    {
                    "image_guidance": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step":0.01, "round": 0.001}),
                    "image_weighting": (["linear down", "cosine down"], ),
                    "weight_scaling": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step":0.01, "round": 0.001}),
                    "latent_image": ("LATENT", ),
                    }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg_max, cfg_min, warmup_percent, mean_cfg,
                    image_guidance, image_weighting, weight_scaling, latent_image = None):
        m = model.clone()
        guider = Guider_MegaCFG(m)
        guider.set_conds(positive, negative) # Conds
        guider.set_cfg(m, cfg_max, cfg_min, warmup_percent, mean_cfg) # Strengths
        if latent_image != None:
            guider.set_img_cfg(image_guidance, image_weighting, weight_scaling, latent_image)
            m.set_model_sampler_post_cfg_function(guider.post_cfg_reference_img)
        return (guider,)
