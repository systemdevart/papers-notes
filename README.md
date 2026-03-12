## Introduction

Hi, these are highly opinionated notes I started taking after reading new papers, to answer questions that came up while reading them, plus random ML questions that popped into my head. I hope future me, or someone else, finds them useful.

### [Drax: Speech Recognition with Discrete Flow Matching](https://arxiv.org/pdf/2510.04162)

This paper itself is not that interesting. It mostly applies existing discrete flow matching (DFM) ideas to a different domain. But it was my first real encounter with DFM, so I wanted to write down a clean comparison of how flow matching changes in the discrete setting.

The bridge from flow matching (FM) to discrete flow matching (DFM) is:

In continuous FM, you have a Euclidean state $x_t \in \mathbb{R}^d$, so the velocity field is an actual vector in data space. For standard interpolation paths, you can regress that vector directly with an L2 loss. In the discrete case, there is no tangent vector attached to a token id like `7` or `cat`, so the model does not try to regress a raw velocity in token space. Instead, it moves from vector fields to probability fluxes, or jump intensities, between discrete states, modeled through a continuous-time discrete Markov chain (CTMC). That is the discrete analogue of the continuity equation: in continuous space, probability mass moves through a vector field; in discrete space, probability mass moves by jumps between states.

So what gets matched in the discrete case is not a geometric velocity vector but a probability path $p_t$ together with a probability velocity $u_t$. In DFM they define a conditional path $p_t(x \mid x_0, x_1)$, then show that if you have a generating conditional probability velocity, the marginal velocity is the posterior average:

```math
u_t(x, z) = \sum_{x_0, x_1} u_t(x, z \mid x_0, x_1)\, p_t(x_0, x_1 \mid z).
```

That is the exact discrete analogue of the continuous FM conditional-to-marginal construction.

The elegant step is the choice of path. For the standard two-endpoint discrete path, DFM uses a mixture of endpoint delta distributions, which is the direct discrete analogue of linear interpolation:

```math
p_t(x^i \mid x_0, x_1) = (1 - \kappa_t)\, \delta_{x_0}(x^i) + \kappa_t\, \delta_{x_1}(x^i).
```

For that path, they derive a closed-form marginal velocity:

```math
u_t^i(x^i, z) = \frac{\dot{\kappa}_t}{1 - \kappa_t}\left[p_{1 \mid t}(x^i \mid z) - \delta_z(x^i)\right].
```

There is a symmetric backward-time form with $p_{0 \mid t}$ as well.

That is also why the loss stops being L2. The model's unknown is now a categorical posterior such as $p_{1 \mid t}(x^i \mid z)$, not a Euclidean vector velocity. Once the velocity is written as an explicit function of this posterior, the natural training target is to predict that posterior. DFM shows that minimizing the training objective is exactly minimizing cross-entropy to the required posterior. So in discrete FM, velocity learning is indirect: learn the posterior with cross-entropy, then plug it into the closed-form probability-velocity formula.

And finally, a comparison table:

| Aspect | Flow Matching (continuous) | Discrete Flow Matching (DFM) |
| --- | --- | --- |
| **State space** | Continuous vector space, usually $x_t \in \mathbb{R}^d$ | Discrete space, for example token sequences $X_t \in \mathcal{V}^N$ |
| **What is modeled** | A continuous-time vector field $v_t(x)$ | A continuous-time probability velocity, or mass-transfer rate, $u_t(x, z)$ |
| **Path definition** | A probability path $p_t(x)$ between source and data, often via endpoint-conditioned interpolants $p_t(x \mid x_0, x_1)$ | A probability path $p_t(x)$ between source and data, often via endpoint-conditioned discrete bridges $p_t(x \mid x_0, x_1)$ |
| **Typical conditional path** | Linear interpolation, for example $x_t = (1 - t)x_0 + t x_1$ | Mixture bridge, $p_t(x^i \mid x_0, x_1) = (1 - \kappa_t)\delta_{x_0}(x^i) + \kappa_t \delta_{x_1}(x^i)$ |
| **Meaning of $t$** | Continuous time, usually $t \in [0, 1]$ | Same |
| **Velocity object** | $v_t(x)$, an actual vector in data space | $u_t^i(\cdot, z)$, a signed rate vector over vocabulary values at position $i$ |
| **Learned target** | Usually regress the conditional or marginal velocity directly | Usually predict the endpoint posterior $p_{1 \mid t}(\cdot \mid z)$, then convert it to velocity |
| **Training loss** | Often L2 or MSE on the velocity target | Usually cross-entropy on the endpoint token posterior |
| **Inference dynamics** | Solve or discretize an ODE, for example $\dot{x}_t = v_t(x_t)$ | Use a CTMC-style or Euler-like PMF update, then sample tokens |
| **What changes each step** | A continuous sample $x_t$ moves in $\mathbb{R}^d$ | Every token position is refined in parallel by resampling from an updated PMF |
| **Sampler state** | A continuous vector or tensor | A full discrete sequence $X_t$ |
| **Current-state representation** | Just the current point $x_t$ | A one-hot PMF at each position |
| **Best intuition** | Move a point through continuous space toward the data manifold | Move probability mass from the current token toward the predicted final-token distribution |

### [Attention Sinks in Diffusion Language Models](https://arxiv.org/pdf/2510.15731)

This paper is mostly a fun observation: masked Diffusion LMs also have attention sinks, but they do not behave like in autoregressive models. In ARMs, sinks are usually quite stable and a few tokens keep attracting a disproportionate amount of attention. In DLMs, sinks are much more dynamic: during denoising they can move, disappear, and reappear, often drifting across the sequence as more tokens get unmasked. The other neat part is that DLMs stay surprisingly robust when these sink tokens are masked - performance drops, but nowhere near the catastrophic failure usually seen in ARMs.

### [CosyEdit: Unlocking End-to-End Speech Editing Capability from Zero-Shot Text-to-Speech Models](https://arxiv.org/pdf/2601.05329)

The paper focuses on how to get an audio editing model without an explicitly labeled editing dataset. The main focus is on editing linguistic content in speech, for example adding or deleting words, while preserving all other paralinguistic information. The authors use a nice trick: instead of collecting a dedicated dataset, they leverage in-context learning. They first train a general voice-cloning autoregressive model conditioned on `[text, reference_clip]`, but at inference they pass `[text + new_text, reference_clip, original_speech]` and ask the model to continue the sequence for `new_text`. Simple to implement, and a good idea.

### [Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis](https://arxiv.org/pdf/2411.19509)

Recently I got interested in live avatar projects and decided to build a pet project with realtime talking-head "celeb avatars" from audio plus a reference image, and I ran into this paper. The main trick in Ditto is that they do not run diffusion in image or VAE space at all. Instead, they first extract a compact motion representation from a face reenactment model (LivePortrait), and train a Diffusion Transformer (DiT) to predict that motion from audio. The motion space is basically expression deformation plus head pose (`m = {delta, R, t}`), and the actual pixels are produced by a separate one-shot renderer: it takes appearance features from the reference image and warps or decodes them using the predicted motion to get the final frames.

The "synced with audio" part is very literal: during training they align audio features, frames, and conditions at 25 fps, so each slice of audio lines up with one motion step. That alignment is what makes streaming feel natural: you can chunk the incoming audio, generate motion in chunks (with overlaps + fusion), and keep rendering frames as you go instead of waiting for the whole utterance.
And because the diffusion output is an explicit motion vector (not mystery latents), they can add practical controls: emotion labels, eye state (blink/gaze), identity-adaptive keypoints.

The code is super clean, and it was easy to adapt for real-time WebRTC streaming - honestly, a great open project!

### [Controllable Video Generation: A Survey](https://arxiv.org/pdf/2507.16869)

Motivation: What tokenizers are used in video generation.

We discussed recently that the task of video generation, from a modeling perspective, might be closer to audio generation than the image domain. So, I looked at this survey of open video generation models and checked what they use for tokenizers.

All models run on VAE-3D, and the vast majority use non-autoregressive diffusion (which is expected). But even the autoregressive models are essentially LLMs with a diffusion head (there was a review in this chat of a similar TTS model, Ming-UniAudio).

Among modern discrete autoregressive models, I only found one, MAGVIT, which seems to be the exception.

### [Semantic-VAE: Semantic-Alignment Latent Representation for Better Speech Synthesis](https://arxiv.org/pdf/2509.22167)

Motivation: Non-autoregressive TTS models are trained on mel-spectrograms; let us train on a VAE, not a plain one, but one with semantically aligned latents, so TTS drops fewer phonemes.
Method: Train the VAE not only with the variational loss but also by minimizing cosine distance to HuBERT embeddings; then train a non-autoregressive TTS (e.g., F5-TTS) on top of these VAE latents.
Details: 64-dim latent (with TTS-VAEs it's unclear what size to use, but 64 is more or less standard, though the figures use much smaller), 40 latents/sec.
Outcome: WER decreases on generations, as expected.
Takeaway: The paper isn't very useful, but at least semantics has finally reached non-autoregressive TTS!

### [UltraEdit: Instruction-based Fine-Grained Image Editing at Scale](https://arxiv.org/pdf/2407.05282)

Motivation: How can we create a large dataset of image-edited pairs so a diffusion model learns to edit images in a free-form way?

I discussed this after "nano-banano" kicked off, and then this paper dropped. It was not clear at the beginning how such a dataset could be obtained. It turns out there is an interesting research method that adapts diffusion for this, and I wanted to share the gist of the paper.

Problem: there are not readily available pairs `(source image, editing text, target image)` that let a model generalize to prompts like "replace the dog on the right with a cat". Previously this was done via inpainting (masking plus diffusion), but getting a model to find the right object on its own was not supported. Inpainting is also fairly limited, so we need a more powerful tool.

Enter the Prompt-to-Prompt method: the idea is to run a standard text-to-image diffusion model twice. The first pass obtains latents for the source image we want to edit; the second pass injects a signal into those latents so that we perform an edit rather than simply regenerating the original image.

The process looks like this:

1. For the source image, generate a caption using any image-understanding model, for example "a dog on a white background" (`T_s`).
2. Add noise to the source image to obtain Z_s.
3. Run the diffusion model on `(T_s, Z_s)` and record all `K`, `V`, and `Q` from the cross-attention used to project text from `T_s` to `Z_s`.
4. Generate a new prompt, for example "a cat on a white background" (`T_t`).
5. Run the diffusion model on (T_t, Z_s), while preserving the K, V, Q corresponding to the parts of the text that didn't change ("on a white background").
6. Obtain the target image I_t, where only the dog is changed to a cat and the background remains the same. Then ask an LLM to craft an editing instruction (not "a cat on a white background," but "replace the dog with a cat") and get the editing prompt P_t.
7. Repeat this 100 times and filter by quality metrics.

As a result, we have mined a dataset `(I_s, P_t, I_t)` consisting of the source image, the editing prompt, and the target image, to train a model for free-form image editing.

### [Ming-UniAudio: Speech LLM for Joint Understanding, Generation and Editing with Unified Representation](https://xqacmer.github.io/Ming-Unitok-Audio.github.io/)

A pretty interesting model has been released that tackles a wide range of audio tasks. The model's generations aren't particularly strong, but what's most interesting is that it's an autoregressive LLM operating over latent representations. There's no paper yet, but the code and weights are open.

How did the authors get continuous representations into an LLM?
They use a hybrid approach. Text is tokenized in the standard way, while audio is embedded with an AudioVAE (no quantization) into continuous latents.
In this setup, the LLM doesn't autoregressively generate the audio latents themselves; instead, it generates conditioning for a Flow head. That head predicts the next patch of audio latents (e.g., 5 at a time) in the VAE space (64-dim in this case), which can then be decoded back to audio by the VAE decoder.
The generated continuous VAE latents then need to be projected (run back through part of the VAE so that, from the 64-dim bottleneck, they are mapped into a higher-level semantic space) to return to the LLM's embedding space for the next step.

Simplified generation process:

```
vae = AudioVAE()
linear_proj = LinearPooling()

input_embeds = prepare_initial_llm_inputs(speaker_prompt, source_text, target_text)
latent_history = init_latent_history() # the last generated VAE patch, shape [B, patch_size, vae_latent_dim] = [B, 5, 64]

while True:
    outputs = llm.model(inputs_embeds=input_embeds)
    llm_condition = outputs.hidden_states[-1][:, -1:, :]

    if stop_head(llm_condition).predict_stop():
        break

    sampled_vae_patch = flow.sample(
        c=llm_condition,
        latent_history=latent_history
    )  # shape = (B, 5, vae_latent_dim=64)

    # Project the VAE patch back into the LLM space
    high_level_latent = vae.encode_unified_emb_from_latent(sampled_vae_patch)
    input_embeds = linear_proj(high_level_latent)

    latent_history = sampled_vae_patch
```

How does the Flow head work and how is it conditioned?
The Flow head is implemented with a DiT (Diffusion Transformer) and is conditioned on the LLM output. As inputs, it takes embeddings of previously generated VAE latents (latent_history with shape [B, 5, 64]) and the current noisy patch x, along with the LLM condition c.

```
def dit_forward(self, x, t, c, latent_history):
    # c: LLM condition; x: current noisy VAE latent patch with shape [B, patch_size, vae_latent_dim] = [B, 5, 64]
    y = self.t_embedder(t) + self.c_embedder(c)

    x_history_emb = self.x_embedder(latent_history)
    x_now_emb = self.x_embedder(x)
    audio_sequence = torch.cat([x_history_emb, x_now_emb], dim=1)

    full_input = torch.cat([y, audio_sequence], dim=1)

    for block in self.blocks:
        full_input = block(full_input)

    # return the vector field prediction for the current patch only
    return full_input[:, -patch_size:, :]
```

Architecture and model sizes:

- VAE (note there's no quantizer): runs at 16 kHz, produces 50 latents per second, ~1B parameters.
- LLM: a standard LLaMA-like MoE model, ~16B parameters.
- Flow head (DiT/CFM): ~100M parameters.

Honestly, the model looks impressive, especially given how rare this kind of architecture is (LLM + continuous latents + CFM).

### [Qwen3-Omni Technical Report](https://arxiv.org/pdf/2509.17765)

The most interesting question is how the authors embedded audio and how they generated speech.

For audio embeddings, they pretrained their own Audio Transformer (AuT) on 20 million hours of data. This is a standard approach, essentially the same idea as feeding Whisper latents directly into an LLM.

In effect, they use a cascaded generation setup. Instead of the usual LLM → Text → TTS, they do LLM → LLM latents → TTS, which is a questionable choice given that the standard pipeline this year is to fuse audio and text into a single model. Concretely, for generation they introduce a second autoregressive model: from the GPT latents of the text model, when speech is needed, they autoregressively produce audio tokens of some RVQ quantizer, which are then decoded into audio by the quantizer's standard decoder.

Unfortunately, neither this paper nor the previous one provides any details about the quantizer, and, as often happens, they have bolted on generation as a side cascaded module rather than integrating it natively into the GPT text model.

### [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/pdf/2510.11690)

I've been wondering if it's possible to run diffusion on a standard autoencoder without the variational baggage, and then this paper dropped.

The authors introduce an approach called RAE (Representation Autoencoder). The main challenges they tackled were the discrete nature of standard AE latent spaces (which lack the continuity needed for diffusion) and the high dimensionality of latents from pretrained encoders.

Interestingly, the authors demonstrate that having a highly "semantic" space isn't a bug, it's a feature. It actually improves generation and accelerates convergence.

Here's the recipe they used to make it work:

1.  They took a pretrained encoder (e.g., DINOv2) and froze it.
2.  They trained a decoder on top of the frozen encoder, but injected Gaussian noise into the latents during training. This noise augmentation makes the decoder robust to the continuous, noisy distribution generated by the diffusion model, effectively side-stepping the need for the VAE formulation.
3.  They trained a DiT with flow matching on these latents. To handle the high-dimensional tokens, they had to:
    - Increase the model width significantly compared to VAE-based diffusion models.
    - Use a specialized noise scheduler that depends on the data dimensionality.

The authors claim a new SOTA (FID 1.13 on ImageNet 512x512) and massive convergence speedups (16-47x faster than VAE counterparts). However, the most interesting takeaway from a research perspective is the validation of the hypothesis that diffusion can be trained efficiently in high-dimensional, semantic spaces without relying on variational methods.
