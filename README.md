## Introduction

Hi, these are highly opinionated notes I recently started taking after reading some new papers — to answer questions that arose while reading each one, as well as random ML questions that popped into my head and the answers I found for them.
I hope future me - or someone else - finds it useful.

### [Ming-UniAudio: Speech LLM for Joint Understanding, Generation and Editing with Unified Representation](https://xqacmer.github.io/Ming-Unitok-Audio.github.io/)

A pretty interesting model has been released that tackles a wide range of audio tasks. The modelэs generations arenэt particularly strong, but what's most interesting is that it's an autoregressive LLM operating over latent representations. There's no paper yet, but the code and weights are open.

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
    # c — LLM condition # x — current noisy VAE latent patch of shape [B, patch_size, vae_latent_dim] = [B, 5, 64]
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

Honestly, the model looks impressive—especially given how rare this kind of architecture is (LLM + continuous latents + CFM).

### [Qwen3-Omni Technical Report](https://arxiv.org/pdf/2509.17765)

The most interesting question is how the authors embedded audio and how they generated speech.

For audio embeddings, they pretrained their own Audio Transformer (AuT) on 20 million hours of data—a standard approach, essentially the same idea as feeding Whisper latents directly into an LLM.

In effect, they use a cascaded generation setup. Instead of the usual LLM → Text → TTS, they do LLM → LLM latents → TTS, which is a questionable choice given that the standard pipeline this year is to fuse audio and text into a single model. Concretely, for generation they introduce a second autoregressive model: from the GPT latents of the text model, when speech is needed, they autoregressively produce audio tokens of some RVQ quantizer, which are then decoded into audio by the quantizer's standard decoder.

Unfortunately, neither this paper nor the previous one provides any details about the quantizer, and—as often happens—they've bolted on generation as a side cascaded module rather than integrating it natively into the GPT text model.

### [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/pdf/2510.11690)

I've been wondering if it's possible to run diffusion on a standard autoencoder without the variational baggage, and then this paper dropped: [https://arxiv.org/pdf/2510.11690](https://arxiv.org/pdf/2510.11690)

The authors introduce an approach called RAE (Representation Autoencoder). The main challenges they tackled were the discrete nature of standard AE latent spaces (which lack the continuity needed for diffusion) and the high dimensionality of latents from pretrained encoders.

Interestingly, the authors demonstrate that having a highly "semantic" space isn't a bug, it's a feature. It actually improves generation and accelerates convergence.

Here's the recipe they used to make it work:

1.  They took a pretrained encoder (e.g., DINOv2) and froze it.
2.  They trained a decoder on top of the frozen encoder, but injected Gaussian noise into the latents during training. This noise augmentation makes the decoder robust to the continuous, noisy distribution generated by the diffusion model, effectively side-stepping the need for the VAE formulation.
3.  They trained a DiT with flow matching on these latents. To handle the high-dimensional tokens, they had to:
    - Increase the model width significantly compared to VAE-based diffusion models.
    - Use a specialized noise scheduler that depends on the data dimensionality.

The authors claim a new SOTA (FID 1.13 on ImageNet 512x512) and massive convergence speedups (16-47x faster than VAE counterparts). However, the most interesting takeaway from a research perspective is the validation of the hypothesis that diffusion can be trained efficiently in high-dimensional, semantic spaces without relying on variational methods.
