from . import extra_samplers
from . import nodes

extra_samplers.add_samplers()
#extra_samplers.add_schedulers()

NODE_CLASS_MAPPINGS = {
    ## K-Samplers
    "SamplerCustomNoise": nodes.SamplerCustomNoise,
    "SamplerCustomNoiseDuo": nodes.SamplerCustomNoiseDuo,
    "SamplerCustomModelMixtureDuo": nodes.SamplerCustomModelMixtureDuo,
    # Guiders
    "GeometricCFGGuider": nodes.GeometricCFGGuider,
    "ImageAssistedCFGGuider": nodes.ImageGuidedCFGGuider,
    "ScaledCFGGuider": nodes.ScaledCFGGuider,
    ## Samplers
    "SamplerRES_Momentumized": nodes.SamplerRES_MOMENTUMIZED,
    "SamplerDPMPP_DualSDE_Momentumized": nodes.SamplerDPMPP_DUALSDE_MOMENTUMIZED,
    "SamplerCLYB_4M_SDE_Momentumized": nodes.SamplerCLYB_4M_SDE_MOMENTUMIZED,
    "SamplerTTM": nodes.SamplerTTM,
    "SamplerLCMCustom": nodes.SamplerLCMCustom,
    "SamplerEulerAncestralDancing_Experimental": nodes.SamplerEULER_ANCESTRAL_DANCING,
    "SamplerDPMPP_3M_SDE_DynETA": nodes.SamplerDPMPP_3M_SDE_DYN_ETA,
    "SamplerSupreme": nodes.SamplerSUPREME,
    ### Schedulers
    "SimpleExponentialScheduler": nodes.SimpleExponentialScheduler,
}
__all__ = ['NODE_CLASS_MAPPINGS']
