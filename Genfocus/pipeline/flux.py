import torch
from typing import List, Union, Optional, Dict, Any, Callable, Type, Tuple

from diffusers.pipelines import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    FluxTransformer2DModel,
    calculate_shift,
    retrieve_timesteps,
    np,
)
from diffusers.models.attention_processor import Attention, F
from diffusers.models.embeddings import apply_rotary_emb
from transformers import pipeline

from peft.tuners.tuners_utils import BaseTunerLayer
from accelerate.utils import is_torch_version

from contextlib import contextmanager

import cv2

from PIL import Image, ImageFilter
import gc
import copy

def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)


def clip_hidden_states(hidden_states: torch.FloatTensor) -> torch.FloatTensor:
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states

def encode_images(pipeline: FluxPipeline, images: torch.Tensor,No_preprocess=False):
    """
    Encodes the images into tokens and ids for FLUX pipeline.
    """
    if not No_preprocess:
        images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids


depth_pipe = None


def convert_to_condition(
    condition_type: str,
    raw_img: Union[Image.Image, torch.Tensor],
) -> Union[Image.Image, torch.Tensor]:
    if condition_type == "depth":
        global depth_pipe
        depth_pipe = depth_pipe or pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device="cpu",  # Use "cpu" to enable parallel processing
        )
        source_image = raw_img.convert("RGB")
        condition_img = depth_pipe(source_image)["depth"].convert("RGB")
        return condition_img
    elif condition_type == "canny":
        img = np.array(raw_img)
        edges = cv2.Canny(img, 100, 200)
        edges = Image.fromarray(edges).convert("RGB")
        return edges
    elif condition_type == "coloring":
        return raw_img.convert("L").convert("RGB")
    elif condition_type == "deblurring":
        return raw_img.convert("RGB")
    else:
        print("Warning: Returning the raw image.")
        return raw_img.convert("RGB")


class Condition(object):
    def __init__(
        self,
        condition: Union[Image.Image, torch.Tensor],
        adapter_setting: Union[str, dict],
        position_delta=None,
        position_scale=1.0,
        latent_mask=None,
        is_complement=False,
        No_preprocess=False,
    ) -> None:
        self.condition = condition
        self.adapter = adapter_setting
        self.position_delta = position_delta
        self.position_scale = position_scale
        self.latent_mask = (
            latent_mask.T.reshape(-1) if latent_mask is not None else None
        )
        self.is_complement = is_complement
        self.No_preprocess=No_preprocess

    def encode(
        self, pipe: FluxPipeline, empty: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if isinstance(self.condition, Image.Image):
            condition_empty = Image.new("RGB", self.condition.size, (0, 0, 0))
        elif torch.is_tensor(self.condition):
            H, W = self.condition.shape[-2], self.condition.shape[-1]
            condition_empty = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8), "RGB")
        tokens, ids = encode_images(pipe, condition_empty if empty else self.condition,self.No_preprocess)

        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]

        if self.position_scale != 1.0:
            scale_bias = (self.position_scale - 1.0) / 2
            ids[:, 1:] *= self.position_scale
            ids[:, 1:] += scale_bias

        if self.latent_mask is not None:
            tokens = tokens[:, self.latent_mask]
            ids = ids[self.latent_mask]

        return tokens, ids

@contextmanager
def specify_lora(lora_modules: List[BaseTunerLayer], specified_lora):
    # Filter valid lora modules
    valid_lora_modules = [m for m in lora_modules if isinstance(m, BaseTunerLayer)]
    # Save original scales
    original_scales = [
        {
            adapter: module.scaling[adapter]
            for adapter in module.active_adapters
            if adapter in module.scaling
        }
        for module in valid_lora_modules
    ]
    # Enter context: adjust scaling
    for module in valid_lora_modules:
        for adapter in module.active_adapters:
            if adapter in module.scaling:
                module.scaling[adapter] = 1 if adapter == specified_lora else 0
    try:
        yield
    finally:
        # Exit context: restore original scales
        for module, scales in zip(valid_lora_modules, original_scales):
            for adapter in module.active_adapters:
                if adapter in module.scaling:
                    module.scaling[adapter] = scales[adapter]


def attn_forward(
    attn: Attention,
    hidden_states: List[torch.FloatTensor],
    adapters: List[str],
    hidden_states2: Optional[List[torch.FloatTensor]] = [],
    position_embs: Optional[List[torch.Tensor]] = None,
    group_mask: Optional[torch.Tensor] = None,
    cache_mode: Optional[str] = None,
    # to determine whether to cache the keys and values for this branch
    to_cache: Optional[List[torch.Tensor]] = None,
    cache_storage: Optional[List[torch.Tensor]] = None,
    **kwargs: dict,
) -> torch.FloatTensor:
    bs, _, _ = hidden_states[0].shape
    h2_n = len(hidden_states2)

    queries, keys, values = [], [], []

    # Prepare query, key, value for each encoder hidden state (text branch)
    for i, hidden_state in enumerate(hidden_states2):
        query = attn.add_q_proj(hidden_state)
        key = attn.add_k_proj(hidden_state)
        value = attn.add_v_proj(hidden_state)

        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)

        query, key, value = map(reshape_fn, (query, key, value))
        query, key = attn.norm_added_q(query), attn.norm_added_k(key)

        queries.append(query)
        keys.append(key)
        values.append(value)

    # Prepare query, key, value for each hidden state (image branch)
    for i, hidden_state in enumerate(hidden_states):
        with specify_lora((attn.to_q, attn.to_k, attn.to_v), adapters[i + h2_n]):
            query = attn.to_q(hidden_state)
            key = attn.to_k(hidden_state)
            value = attn.to_v(hidden_state)

        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)

        query, key, value = map(reshape_fn, (query, key, value))
        query, key = attn.norm_q(query), attn.norm_k(key)

        queries.append(query)
        keys.append(key)
        values.append(value)

    # Apply rotary embedding
    if position_embs is not None:
        queries = [apply_rotary_emb(q, position_embs[i]) for i, q in enumerate(queries)]
        keys = [apply_rotary_emb(k, position_embs[i]) for i, k in enumerate(keys)]

    if cache_mode == "write":
        for i, (k, v) in enumerate(zip(keys, values)):
            if to_cache[i]:
                cache_storage[attn.cache_idx][0].append(k)
                cache_storage[attn.cache_idx][1].append(v)

    attn_outputs = []
    for i, query in enumerate(queries):
        keys_, values_ = [], []
        # Add keys and values from other branches
        for j, (k, v) in enumerate(zip(keys, values)):
            if (group_mask is not None) and not (group_mask[i][j].item()):
                continue
            keys_.append(k)
            values_.append(v)
        if cache_mode == "read":
            keys_.extend(cache_storage[attn.cache_idx][0])
            values_.extend(cache_storage[attn.cache_idx][1])
        # Add keys and values from cache TODO
        # Attention computation
        attn_output = F.scaled_dot_product_attention(
            query, torch.cat(keys_, dim=2), torch.cat(values_, dim=2)
        ).to(query.dtype)
        attn_output = attn_output.transpose(1, 2).reshape(bs, -1, attn.heads * head_dim)
        attn_outputs.append(attn_output)

    # Reshape attention output to match the original hidden states
    h_out, h2_out = [], []

    for i, hidden_state in enumerate(hidden_states2):
        h2_out.append(attn.to_add_out(attn_outputs[i]))

    for i, hidden_state in enumerate(hidden_states):
        h = attn_outputs[i + h2_n]
        if getattr(attn, "to_out", None) is not None:
            with specify_lora((attn.to_out[0],), adapters[i + h2_n]):
                h = attn.to_out[0](h)
        h_out.append(h)
    return (h_out, h2_out) if h2_n else h_out


def block_forward(
    self,
    image_hidden_states: List[torch.FloatTensor],
    text_hidden_states: List[torch.FloatTensor],
    tembs: List[torch.FloatTensor],
    adapters: List[str],
    position_embs=None,
    attn_forward=attn_forward,
    **kwargs: dict,
):
    txt_n = len(text_hidden_states)

    img_variables, txt_variables = [], []

    for i, text_h in enumerate(text_hidden_states):
        txt_variables.append(self.norm1_context(text_h, emb=tembs[i]))

    for i, image_h in enumerate(image_hidden_states):
        with specify_lora((self.norm1.linear,), adapters[i + txt_n]):
            img_variables.append(self.norm1(image_h, emb=tembs[i + txt_n]))

    # Attention.
    img_attn_output, txt_attn_output = attn_forward(
        self.attn,
        hidden_states=[each[0] for each in img_variables],
        hidden_states2=[each[0] for each in txt_variables],
        position_embs=position_embs,
        adapters=adapters,
        **kwargs,
    )

    text_out = []
    for i in range(len(text_hidden_states)):
        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = txt_variables[i]
        text_h = text_hidden_states[i] + txt_attn_output[i] * gate_msa.unsqueeze(1)
        norm_h = (
            self.norm2_context(text_h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        text_h = self.ff_context(norm_h) * gate_mlp.unsqueeze(1) + text_h
        text_out.append(clip_hidden_states(text_h))

    image_out = []
    for i in range(len(image_hidden_states)):
        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = img_variables[i]
        image_h = (
            image_hidden_states[i] + img_attn_output[i] * gate_msa.unsqueeze(1)
        ).to(image_hidden_states[i].dtype)
        norm_h = self.norm2(image_h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        with specify_lora((self.ff.net[2],), adapters[i + txt_n]):
            image_h = image_h + self.ff(norm_h) * gate_mlp.unsqueeze(1)
        image_out.append(clip_hidden_states(image_h))
    return image_out, text_out


def single_block_forward(
    self,
    hidden_states: List[torch.FloatTensor],
    tembs: List[torch.FloatTensor],
    adapters: List[str],
    position_embs=None,
    attn_forward=attn_forward,
    **kwargs: dict,
):
    mlp_hidden_states, gates = [[None for _ in hidden_states] for _ in range(2)]

    hidden_state_norm = []
    for i, hidden_state in enumerate(hidden_states):
        # [NOTE]!: This function's output is slightly DIFFERENT from the original
        # FLUX version. In the original implementation, the gates were computed using
        # the combined hidden states from both the image and text branches. Here, each
        # branch computes its gate using only its own hidden state.
        with specify_lora((self.norm.linear, self.proj_mlp), adapters[i]):
            h_norm, gates[i] = self.norm(hidden_state, emb=tembs[i])
            mlp_hidden_states[i] = self.act_mlp(self.proj_mlp(h_norm))
        hidden_state_norm.append(h_norm)

    attn_outputs = attn_forward(
        self.attn, hidden_state_norm, adapters, position_embs=position_embs, **kwargs
    )

    h_out = []
    for i in range(len(hidden_states)):
        with specify_lora((self.proj_out,), adapters[i]):
            h = torch.cat([attn_outputs[i], mlp_hidden_states[i]], dim=2)
            h = gates[i].unsqueeze(1) * self.proj_out(h) + hidden_states[i]
            h_out.append(clip_hidden_states(h))

    return h_out


def transformer_forward(
    transformer: FluxTransformer2DModel,
    image_features: List[torch.Tensor],
    text_features: List[torch.Tensor] = None,
    img_ids: List[torch.Tensor] = None,
    txt_ids: List[torch.Tensor] = None,
    pooled_projections: List[torch.Tensor] = None,
    timesteps: List[torch.LongTensor] = None,
    guidances: List[torch.Tensor] = None,
    adapters: List[str] = None,
    # Assign the function to be used for the forward pass
    single_block_forward=single_block_forward,
    block_forward=block_forward,
    attn_forward=attn_forward,
    **kwargs: dict,
):
    self = transformer
    txt_n = len(text_features) if text_features is not None else 0

    adapters = adapters or [None] * (txt_n + len(image_features))
    assert len(adapters) == len(timesteps)

    # Preprocess the image_features
    image_hidden_states = []
    for i, image_feature in enumerate(image_features):
        with specify_lora((self.x_embedder,), adapters[i + txt_n]):
            image_hidden_states.append(self.x_embedder(image_feature))

    # Preprocess the text_features
    text_hidden_states = []
    for text_feature in text_features:
        text_hidden_states.append(self.context_embedder(text_feature))

    # Prepare embeddings of (timestep, guidance, pooled_projections)
    assert len(timesteps) == len(image_features) + len(text_features)

    def get_temb(timestep, guidance, pooled_projection):
        timestep = timestep.to(image_hidden_states[0].dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(image_hidden_states[0].dtype) * 1000
            return self.time_text_embed(timestep, guidance, pooled_projection)
        else:
            return self.time_text_embed(timestep, pooled_projection)

    tembs = [get_temb(*each) for each in zip(timesteps, guidances, pooled_projections)]

    # Prepare position embeddings for each token
    position_embs = [self.pos_embed(each) for each in (*txt_ids, *img_ids)]

    # Prepare the gradient checkpointing kwargs
    gckpt_kwargs: Dict[str, Any] = (
        {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
    )

    # dual branch blocks
    for block in self.transformer_blocks:
        block_kwargs = {
            "self": block,
            "image_hidden_states": image_hidden_states,
            "text_hidden_states": text_hidden_states,
            "tembs": tembs,
            "position_embs": position_embs,
            "adapters": adapters,
            "attn_forward": attn_forward,
            **kwargs,
        }
        if self.training and self.gradient_checkpointing:
            image_hidden_states, text_hidden_states = torch.utils.checkpoint.checkpoint(
                block_forward, **block_kwargs, **gckpt_kwargs
            )
        else:
            image_hidden_states, text_hidden_states = block_forward(**block_kwargs)

    # combine image and text hidden states then pass through the single transformer blocks
    all_hidden_states = [*text_hidden_states, *image_hidden_states]
    for block in self.single_transformer_blocks:
        block_kwargs = {
            "self": block,
            "hidden_states": all_hidden_states,
            "tembs": tembs,
            "position_embs": position_embs,
            "adapters": adapters,
            "attn_forward": attn_forward,
            **kwargs,
        }
        if self.training and self.gradient_checkpointing:
            all_hidden_states = torch.utils.checkpoint.checkpoint(
                single_block_forward, **block_kwargs, **gckpt_kwargs
            )
        else:
            all_hidden_states = single_block_forward(**block_kwargs)

    image_hidden_states = self.norm_out(all_hidden_states[txt_n], tembs[txt_n])
    output = self.proj_out(image_hidden_states)

    return (output,)


@torch.no_grad()
def generate(
    pipeline: FluxPipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    # Condition Parameters (Optional)
    main_adapter: Optional[List[str]] = None,
    conditions: List[Condition] = [],
    image_guidance_scale: float = 1.0,
    transformer_kwargs: Optional[Dict[str, Any]] = {},
    kv_cache=False,
    latent_mask=None,
    TILE_SIZE = 32,
    NO_TILED_DENOISE=False,
    **params: dict,
):
    self = pipeline

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs

    # Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # Prepare prompt embeddings
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    # Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    _, fixed_latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        512,
        512,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    if latent_mask is not None:
        latent_mask = latent_mask.T.reshape(-1)
        latents = latents[:, latent_mask]
        latent_image_ids = latent_image_ids[latent_mask]
# Determine if we need tiled denoising
    # latents shape: [batch_size, H*W, channels]
    # latent_image_ids shape: [H*W, 3] where 3 = [batch_idx, h_idx, w_idx]
    latent_h = int(latent_image_ids[:, 1].max().item() + 1)
    latent_w = int(latent_image_ids[:, 2].max().item() + 1)
    if latent_h * latent_w < TILE_SIZE**2:
        TILED_DENOISE = False
    else:
        TILED_DENOISE = True

    if NO_TILED_DENOISE:
        TILED_DENOISE = False

        
    print(f"Latent size: {latent_h}x{latent_w}, Tiled denoise: {TILED_DENOISE}")
    # Prepare tiling coordinates if needed
    if TILED_DENOISE:
        # Calculate tile positions with 12.5% overlap
        overlap = TILE_SIZE // 8
        stride = TILE_SIZE - overlap
        
        cropped_height_coord = np.arange(0, max(1, latent_h - TILE_SIZE + 1), stride, dtype=np.int32)
        cropped_width_coord = np.arange(0, max(1, latent_w - TILE_SIZE + 1), stride, dtype=np.int32)
        
        # Ensure we cover the full image
        if cropped_height_coord[-1] + TILE_SIZE < latent_h:
            cropped_height_coord = np.append(cropped_height_coord, latent_h - TILE_SIZE)
        if cropped_width_coord[-1] + TILE_SIZE < latent_w:
            cropped_width_coord = np.append(cropped_width_coord, latent_w - TILE_SIZE)
            
        cropped_height_coord, cropped_width_coord = np.meshgrid(cropped_height_coord, cropped_width_coord)
        cropped_height_coord = cropped_height_coord.ravel()
        cropped_width_coord = cropped_width_coord.ravel()
        
        print(f"Number of tiles: {len(cropped_height_coord)}")
        
        # Create Gaussian weight kernel for blending
        temp_d = torch.linspace(-2, 2, TILE_SIZE, dtype=latents.dtype, device=device)
        gaussian_2d = torch.exp(-(temp_d[:, None] ** 2 + temp_d[None] ** 2))
        # Flatten to match token format: [TILE_SIZE*TILE_SIZE]
        gaussian_weights = gaussian_2d.flatten()
    # Prepare conditions
    c_latents, uc_latents, c_ids, c_timesteps = ([], [], [], [])
    c_projections, c_guidances, c_adapters = ([], [], [])
    complement_cond = None
    for condition in conditions:
        tokens, ids = condition.encode(self)
        c_latents.append(tokens)  # [batch_size, token_n, token_dim]
        # Empty condition for unconditioned image
        if image_guidance_scale != 1.0:
            uc_latents.append(condition.encode(self, empty=True)[0])
        c_ids.append(ids)  # [token_n, id_dim(3)]
        c_timesteps.append(torch.zeros([1], device=device))
        c_projections.append(pooled_prompt_embeds)
        c_guidances.append(torch.ones([1], device=device))
        c_adapters.append(condition.adapter)
        # This complement_condition will be combined with the original image.
        # See the token integration of OminiControl2 [https://arxiv.org/abs/2503.08280]
        if condition.is_complement:
            complement_cond = (tokens, ids)
    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    if kv_cache:
        attn_counter = 0
        for module in self.transformer.modules():
            if isinstance(module, Attention):
                setattr(module, "cache_idx", attn_counter)
                attn_counter += 1
        kv_cond = [[[], []] for _ in range(attn_counter)]
        kv_uncond = [[[], []] for _ in range(attn_counter)]

        def clear_cache():
            for storage in [kv_cond, kv_uncond]:
                for kesy, values in storage:
                    kesy.clear()
                    values.clear()

    branch_n = len(conditions) + 2
    group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool)
    # Disable the attention cross different condition branches
    group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
    # # Disable the attention from condition branches to image branch and text branch
    # if kv_cache:
    #     group_mask[2:, :2] = False
    def create_mapping_from_latent_ids(latent_image_ids):
        h_coords = latent_image_ids[:, 1]
        w_coords = latent_image_ids[:, 2]
        latent_h = h_coords.max().item() + 1
        latent_w = w_coords.max().item() + 1
        #turn latent_h, latent_w to int
        latent_h = int(latent_h)
        latent_w = int(latent_w)
        h_coords = h_coords.int()
        w_coords = w_coords.int()
        
        
        coord_to_idx = torch.full((latent_h, latent_w), -1)
        coord_to_idx = coord_to_idx.to(dtype=torch.long, device=latent_image_ids.device)
        
        
        coord_to_idx[h_coords, w_coords] = torch.arange(
            len(latent_image_ids), 
            device=latent_image_ids.device
        )
        
        return coord_to_idx
    if TILED_DENOISE:
        coord_to_idx = create_mapping_from_latent_ids(latent_image_ids)


    
    #define disk kernel
    def make_disk_kernel(ksz: int, device, dtype):
        assert ksz >= 1
        r = (ksz - 1) / 2.0
        coords = torch.arange(ksz, device=device, dtype=torch.float32) - r
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        mask = (xx**2 + yy**2) <= (r + 1e-6)**2
        K = mask.float()
        K = K / K.sum().clamp(min=1.0)
        return K.view(1, 1, ksz, ksz).to(dtype=dtype)
    # Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance, c_guidances = None, [None for _ in c_guidances]

            if kv_cache:
                mode = "write" if i == 0 else "read"
                if mode == "write":
                    clear_cache()
            use_cond = not (kv_cache) or mode == "write"

            if TILED_DENOISE:
                # Initialize accumulators for tiled processing
                noise_pred_accumulated = torch.zeros_like(latents)
                weight_accumulated = torch.zeros(latents.shape[:2], device=device, dtype=latents.dtype)  # [batch, tokens]
                
                # Process each tile
                for tile_idx, (ch, cw) in enumerate(zip(cropped_height_coord, cropped_width_coord)):
                    h_end = ch + TILE_SIZE
                    w_end = cw + TILE_SIZE
                    # Extract tile
                    
                    tile_indices = coord_to_idx[ch:h_end, cw:w_end].flatten()
                    tile_latents = latents[:, tile_indices]
                    tile_c_latents_0 = c_latents[0][:, tile_indices] if use_cond else None
                    tile_c_latents_1 = c_latents[1][:, tile_indices] if use_cond else None
                    tile_c_latents=[tile_c_latents_0,tile_c_latents_1]
                    tile_latent_image_ids = latent_image_ids[tile_indices]
                    tile_c_ids_0 = c_ids[0][tile_indices] if use_cond else None
                    tile_c_ids_1 = c_ids[1][tile_indices] if use_cond else None
                    if tile_latents is None:
                        continue
                    


                    # Process tile through transformer
                    tile_noise_pred = transformer_forward(
                        self.transformer,
                        image_features=[tile_latents] + (tile_c_latents if use_cond else []),
                        text_features=[prompt_embeds],
                        img_ids=[fixed_latent_image_ids] + ([fixed_latent_image_ids,fixed_latent_image_ids] if use_cond else []),
                        txt_ids=[text_ids],
                        timesteps=[timestep, timestep] + (c_timesteps if use_cond else []),
                        pooled_projections=[pooled_prompt_embeds] * 2
                        + (c_projections if use_cond else []),
                        guidances=[guidance] * 2 + (c_guidances if use_cond else []),
                        return_dict=False,
                        adapters=[main_adapter] * 2 + (c_adapters if use_cond else []),
                        cache_mode=mode if kv_cache else None,
                        cache_storage=kv_cond if kv_cache else None,
                        to_cache=[False, False, *[True] * len(tile_c_latents)],
                        group_mask=group_mask,
                        **transformer_kwargs,
                    )[0]
                    
                    # Accumulate weighted predictions
                    #tile_weights = one_weights[:tile_latents.shape[1]]  # Match actual tile size
                    tile_weights = gaussian_weights[:tile_latents.shape[1]]
                    noise_pred_accumulated[:, tile_indices] += tile_noise_pred * tile_weights.unsqueeze(0).unsqueeze(-1)
                    weight_accumulated[:, tile_indices] += tile_weights
                    
                    

                # Normalize by accumulated weights
                noise_pred = noise_pred_accumulated / (weight_accumulated.unsqueeze(-1))
                del noise_pred_accumulated, weight_accumulated
                
            else:
                # Standard processing for small images
                noise_pred = transformer_forward(
                    self.transformer,
                    image_features=[latents] + (c_latents if use_cond else []),
                    text_features=[prompt_embeds],
                    img_ids=[latent_image_ids] + (c_ids if use_cond else []),
                    txt_ids=[text_ids],
                    timesteps=[timestep, timestep] + (c_timesteps if use_cond else []),
                    pooled_projections=[pooled_prompt_embeds] * 2
                    + (c_projections if use_cond else []),
                    guidances=[guidance] * 2 + (c_guidances if use_cond else []),
                    return_dict=False,
                    adapters=[main_adapter] * 2 + (c_adapters if use_cond else []),
                    cache_mode=mode if kv_cache else None,
                    cache_storage=kv_cond if kv_cache else None,
                    to_cache=[False, False, *[True] * len(c_latents)],
                    group_mask=group_mask,
                    **transformer_kwargs,
                )[0]


            if image_guidance_scale != 1.0:
                unc_pred = transformer_forward(
                    self.transformer,
                    image_features=[latents] + (uc_latents if use_cond else []),
                    text_features=[prompt_embeds],
                    img_ids=[latent_image_ids] + (c_ids if use_cond else []),
                    txt_ids=[text_ids],
                    timesteps=[timestep, timestep] + (c_timesteps if use_cond else []),
                    pooled_projections=[pooled_prompt_embeds] * 2
                    + (c_projections if use_cond else []),
                    guidances=[guidance] * 2 + (c_guidances if use_cond else []),
                    return_dict=False,
                    adapters=[main_adapter] * 2 + (c_adapters if use_cond else []),
                    cache_mode=mode if kv_cache else None,
                    cache_storage=kv_uncond if kv_cache else None,
                    to_cache=[False, False, *[True] * len(c_latents)],
                    **transformer_kwargs,
                )[0]

                noise_pred = unc_pred + image_guidance_scale * (noise_pred - unc_pred)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    if latent_mask is not None:
        # Combine the generated latents and the complement condition
        assert complement_cond is not None
        comp_latent, comp_ids = complement_cond
        all_ids = torch.cat([latent_image_ids, comp_ids], dim=0)  # (Ta+Tc,3)
        shape = (all_ids.max(dim=0).values + 1).to(torch.long)  # (3,)
        H, W = shape[1].item(), shape[2].item()
        B, _, C = latents.shape
        # Create a empty canvas
        canvas = latents.new_zeros(B, H * W, C)  # (B,H*W,C)

        # Stash the latents and the complement condition
        def _stash(canvas, tokens, ids, H, W) -> None:
            B, T, C = tokens.shape
            ids = ids.to(torch.long)
            flat_idx = (ids[:, 1] * W + ids[:, 2]).to(torch.long)
            canvas.view(B, -1, C).index_copy_(1, flat_idx, tokens)

        _stash(canvas, latents, latent_image_ids, H, W)
        _stash(canvas, comp_latent, comp_ids, H, W)
        latents = canvas.view(B, H * W, C)

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)