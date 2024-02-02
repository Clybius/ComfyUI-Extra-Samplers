from . import extra_samplers
from . import nodes

extra_samplers.add_samplers()
#extra_samplers.add_schedulers()

NODE_CLASS_MAPPINGS = {
    "SamplerCustomNoise": nodes.SamplerCustomNoise,
    "SamplerCustomNoiseDuo": nodes.SamplerCustomNoiseDuo,
    "SamplerCustomModelMixtureDuo": nodes.SamplerCustomModelMixtureDuo,
    "SamplerRES_Momentumized": nodes.SamplerRES_MOMENTUMIZED,
    "SamplerDPMPP_DualSDE_Momentumized": nodes.SamplerDPMPP_DUALSDE_MOMENTUMIZED,
    "SamplerCLYB_4M_SDE_Momentumized": nodes.SamplerCLYB_4M_SDE_MOMENTUMIZED,
    "SamplerTTM": nodes.SamplerTTM,
    "SamplerLCMCustom": nodes.SamplerLCMCustom,
}
__all__ = ['NODE_CLASS_MAPPINGS']
