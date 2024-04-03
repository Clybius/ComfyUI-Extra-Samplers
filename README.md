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
* step_method: You have the ability to choose your own step method with this sampler! Optionally, there's a dynamic step method, which chooses the appropriate order based on the calculated error between steps, allowing you to obtain higher quality when it matters in the sampling process. Defaults to **(Euler)**.
* centralization: Subtracts mean from the denoised latent. This can lead to perceptually sharper results, though may change the perceivable brightness of the image. Conservatively defaults to **(0.05)**.
* normalization: Divides the denoised latent by the standard deviation. Can increase contrast in the image, though may hurt fidelity and coherency at high strengths. Conservatively defaults to **(0.05)**.
* edge_enhancement: Sharpens the latent, and then applies a bilateral blur, leaving the edges sharpened. Defaults to **(0.25)**.
* perphist: Adds previous denoised variable to the current denoised using perpendicular vector projection. Default of **(0.5)**, where higher values add the old denoised variable, and negative values subtract the old denoised variable.
* substeps: Amount of times to iterate over each step and average the results. Can be useful for obtaining higher quality at a given step count. Inspired by [ReNoise](https://arxiv.org/pdf/2403.14602v1.pdf). Default of **(2)**.
* noise_modulation: Modulates the noise based on the denoising step or other functionality. Current options with a default of **("none")**: ("none", "intensity")
* modulation_strength: Strength of the modulation, utilizing a weighted sum between the modulated noise and normal noise. Default of **(2.0)**. Only has an effect when != "none".

#### Supreme Sampler step methods:
* Euler: A simple 1st-order step method.
* DPM_1/2/3S: 1st, 2nd, and 3rd-order samplers of the DPM family of solvers.
* Bogacki Shampine: Denoise using the Bogacki-Shampine method of ODE solvers. 3rd-order sampler.
* RK4/RK45/Adaptive_RK: High-order Runge-Kutta samplers, where RK4 is a 4th-order sampler, RK45 is 6th-order, and Adaptive RK will choose between 4th/3rd/2nd/1st order based on error between previous and current denoised variables.
* Reversible_Heun: Utilizes an implementation of the reversible heun step method, similar to the one found in [torchsde](https://github.com/google-research/torchsde). 2nd-order sampler. 
* Trapezoidal: A [method to solve ODEs](https://en.wikipedia.org/wiki/Trapezoidal_rule_(differential_equations)) derived from the [Trapezoidal Rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) for computing integrals.

Utilizing custom sampling within ComfyUI is encouraged for these samplers!