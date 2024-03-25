# ComfyUI Extra Samplers

### Currently included extra samplers: 
* RES (+ Somewhat naively momentumized and modified, thanks to Kat and Birch-San for the source implementation!)
* DPMPP Dual SDE (+ Somewhat naively momentumized, tis simply DPMPP SDE with an added SDE akin to how 3M samples)
* Clyb 4M SDE (A modified DPMPP 3M SDE, with an added SDE, egotisticalized by yours truly)
* TTM (Thanks to Kat and Birch-San for the source implementation!)
* LCM Custom Noise (Supports different types of noise other than generic gaussian)
* DPMPP 3M SDE with Dynamic ETA (Anneals down towards a minimum eta via a cosine curve)
* Supreme (Many extra functionalities and step methods available)

### Currently included extra K-Sampling nodes:
* SamplerCustomNoise (Supports custom noises other than gaussian noise for init noise)
* SamplerCustomNoiseDuo (Same as above, but with an added High-res fix for simplicity.)
* SamplerCustomModelMixtureDuo (Samples with custom noises, and switches between model1 and model2 every step. If you encounter vram errors, try adding/removing `--disable-smart-memory` when launching ComfyUI)


#### Supreme Sampler features:
* centralization: Subtracts mean from the denoised latent. This can lead to perceptually sharper results, though may change the perceivable brightness of the image. Conservatively defaults to **(0.02)**.
* normalization: Divides the denoised latent by the standard deviation. Can increase contrast in the image, though may hurt fidelity and coherency at high strengths. Conservatively defaults to **(0.01)**.
* edge_enhancement: Sharpens the latent, and then applies a bilateral blur, leaving the edges sharpened. Conservatively defaults to **(0.05)**.
* perphist: Adds previous denoised variable to the current denoised using perpendicular vector projection. Default of **(0)**, where higher values add the old denoised variable, and negative values subtract the old denoised variable.
* substeps: Amount of times to iterate over each step and average the results. Can be useful for obtaining higher quality at a given step count. Inspired by [ReNoise](https://arxiv.org/pdf/2403.14602v1.pdf). Default of **(2)**.

Utilizing custom sampling within ComfyUI is encouraged for these samplers!