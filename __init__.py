from . import extra_samplers
from . import nodes
extra_schedulers = extra_samplers.extra_schedulers

from comfy.samplers import SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES
extra_samplers.add_samplers()
#extra_samplers.add_schedulers()
for key, value in extra_schedulers.items():
    scheduler_name = f"{key}"
    if scheduler_name not in SCHEDULER_HANDLERS:
        scheduler_handler = SchedulerHandler(handler=value, use_ms=True)
        SCHEDULER_HANDLERS[scheduler_name] = scheduler_handler
        if scheduler_name not in SCHEDULER_NAMES:
            SCHEDULER_NAMES.append(scheduler_name)



NODE_CLASS_MAPPINGS = {
    ## K-Samplers
    "SamplerCustomNoise": nodes.SamplerCustomNoise,
    "SamplerCustomNoiseDuo": nodes.SamplerCustomNoiseDuo,
    "SamplerCustomModelMixtureDuo": nodes.SamplerCustomModelMixtureDuo,
    # Guiders
    "GeometricCFGGuider": nodes.GeometricCFGGuider,
    "ImageAssistedCFGGuider": nodes.ImageGuidedCFGGuider,
    "ScaledCFGGuider": nodes.ScaledCFGGuider,
    "WarmupDecayCFGGuider": nodes.WarmupDecayCFGGuider,
    "MegaCFGGuider": nodes.MegaCFGGuider,
    "APGGuider": nodes.APGGuider,
    ### Noise
    "ImmiscibleNoise": nodes.ImmiscibleNoise,
    ## Samplers
    "SamplerRES_Momentumized": nodes.SamplerRES_MOMENTUMIZED,
    "SamplerDPMPP_DualSDE_Momentumized": nodes.SamplerDPMPP_DUALSDE_MOMENTUMIZED,
    "SamplerCLYB_4M_SDE_Momentumized": nodes.SamplerCLYB_4M_SDE_MOMENTUMIZED,
    "SamplerTTM": nodes.SamplerTTM,
    "SamplerLCMCustom": nodes.SamplerLCMCustom,
    "SamplerEulerAncestralDancing_Experimental": nodes.SamplerEULER_ANCESTRAL_DANCING,
    "SamplerDPMPP_3M_SDE_DynETA": nodes.SamplerDPMPP_3M_SDE_DYN_ETA,
    "SamplerSupreme": nodes.SamplerSUPREME,
    "SamplerSENS": nodes.SamplerSENS,
    "SamplerIPNDM_VAPP": nodes.SamplerIPNDM_VAPP,
    "SamplerSHIDS": nodes.SamplerSHIDS,
    "SamplerDPMPP_2M_SDE_EMA": nodes.SamplerDPMPP_2M_SDE_EMA,
    "SamplerBiScope": nodes.SamplerBiScope,
    "SamplerEuler_G": nodes.SamplerEuler_G,
    "SamplerLeaping_Euler": nodes.SamplerLeaping_Euler,
    ### Schedulers
    "SimpleExponentialScheduler": nodes.SimpleExponentialScheduler,
    "KLOptimalScheduler": nodes.KLOptimalScheduler,
    "SimpleKLOptimalScheduler": nodes.SimpleKLOptimalScheduler,
}
__all__ = ['NODE_CLASS_MAPPINGS']
