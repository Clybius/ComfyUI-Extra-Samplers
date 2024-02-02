# ComfyUI Extra Samplers

### Currently included extra samplers: 
* RES (+ Somewhat naively momentumized and modified, thanks to Kat and Birch-San for the source implementation!)
* DPMPP Dual SDE (+ Somewhat naively momentumized, tis simply DPMPP SDE with an added SDE akin to how 3M samples)
* Clyb 4M SDE (A modified DPMPP 3M SDE, with an added SDE, egotisticalized by yours truly)
* TTM (Thanks to Kat and Birch-San for the source implementation!)
* LCM Custom Noise (Supports different types of noise other than generic gaussian)

### Currently included extra K-Sampling nodes:
* SamplerCustomNoise (Supports custom noises other than gaussian noise for init noise)
* SamplerCustomNoiseDuo (Same as above, but with an added High-res fix for simplicity.
* SamplerCustomModelMixtureDuo (Samples with custom noises, and switches between model1 and model2 every step. If you encounter vram errors, try adding/removing `--disable-smart-memory` when launching ComfyUI)
