# -*- coding: utf-8 -*-
"""LLM smooth quantization module."""

import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from deepcompressor.calib.smooth import ActivationSmoother, smooth_attention, smooth_linear_modules
from deepcompressor.data.cache import IOTensorsCache
from deepcompressor.quantizer.processor import Quantizer
from deepcompressor.utils import tools

from ..nn.struct import LlmModelStruct, LlmTransformerBlockStruct
from .config import LlmQuantConfig
from .utils import get_needs_inputs_fn, get_needs_outputs_fn

from .bias_utils import bias_subtracted_tensors_cache  # 8.21 尝试减弱K_bias的影响

extract_K_bias = False


__all__ = ["smooth_llm"]


@torch.inference_mode()
def smooth_llm_layer(  # noqa: C901
    layer: LlmTransformerBlockStruct,
    config: LlmQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
) -> None:
    """Smooth a large language model layer.

    Args:
        layer (`LlmTransformerBlockStruct`):
            Large language model layer to smooth.
        config (`LlmQuantConfig`):
            Quantization configuration.
        smooth_cache (`dict[str, torch.Tensor]`):
            Smoothing scale caches.
        layer_caches (`dict[str, IOTensorsCache]` or `None`, *optional*, defaults to `None`):
            Activation caches of the layer.
        layer_kwargs (`dict[str, tp.Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for the layer.
    """
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    logger.debug("- Smoothing %s", layer.name)
    tools.logging.Formatter.indent_inc()
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}
    attn, ffn = layer.attn_struct, layer.ffn_struct
    # region attention qk
    needs_quant = config.enabled_opts
    needs_quant = needs_quant and (config.opts.is_enabled_for(attn.q_key) or config.opts.is_enabled_for(attn.k_key))
    if config.smooth.enabled_attn and needs_quant:
        logger.debug("- %s.%s", attn.name, attn.k_rkey)
        cache_key = f"{attn.name}.{attn.k_rkey}"
        
        # 8.21
        global extract_K_bias
        if extract_K_bias and layer_cache:
            print("cutting off bias")
            print("cutting off bias")
            print("cutting off bias")
            # 原始含 bias 的 K
            k_outputs_cache = layer_cache[attn.k_name].outputs
            # 构造去 bias 副本
            k_bias = attn.k_proj.bias
            k_outputs_cache_nobias = bias_subtracted_tensors_cache(k_outputs_cache, k_bias)
        else:
            k_outputs_cache_nobias = layer_cache[attn.k_name].outputs if layer_cache else None

        smooth_cache[cache_key] = smooth_attention(
            k_proj=attn.k_proj,
            q_proj=attn.q_proj,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.attn,
            query_quantizer=Quantizer(config.opts, channels_dim=-1, key=attn.q_key),
            key_quantizer=Quantizer(config.opts, channels_dim=-1, key=attn.k_key),
            queries=layer_cache[attn.q_name].outputs if layer_cache else None,
            keys=k_outputs_cache_nobias,
            attn_q=attn.q,
            attn_k=attn.k,
            eval_inputs=layer_cache[attn.name].inputs if layer_cache else None,
            eval_module=attn,
            eval_kwargs=attn.filter_kwargs(layer_kwargs),
            num_heads=attn.config.num_query_heads,
            num_head_repeats=attn.config.num_head_repeats,
            with_rope=attn.config.with_rope,
            develop_dtype=config.develop_dtype,
        )
        
        # smooth_cache[cache_key] = smooth_attention(
        #     k_proj=attn.k_proj,
        #     q_proj=attn.q_proj,
        #     scale=smooth_cache.get(cache_key, None),
        #     config=config.smooth.attn,
        #     query_quantizer=Quantizer(config.opts, channels_dim=-1, key=attn.q_key),
        #     key_quantizer=Quantizer(config.opts, channels_dim=-1, key=attn.k_key),
        #     queries=layer_cache[attn.q_name].outputs if layer_cache else None,
        #     keys=layer_cache[attn.k_name].outputs if layer_cache else None,
        #     attn_q=attn.q,
        #     attn_k=attn.k,
        #     eval_inputs=layer_cache[attn.name].inputs if layer_cache else None,
        #     eval_module=attn,
        #     eval_kwargs=attn.filter_kwargs(layer_kwargs),
        #     num_heads=attn.config.num_query_heads,
        #     num_head_repeats=attn.config.num_head_repeats,
        #     with_rope=attn.config.with_rope,
        #     develop_dtype=config.develop_dtype,
        # )

    # endregion
    # region qkv projection
    needs_quant = config.enabled_ipts and config.ipts.is_enabled_for(attn.qkv_proj_key)
    needs_quant = needs_quant or (config.enabled_wgts and config.wgts.is_enabled_for(attn.qkv_proj_key))
    if config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(attn.qkv_proj_key) and needs_quant:
        logger.debug("- %s.%s", attn.name, attn.qkv_proj_rkey)
        cache_key = attn.v_proj_name
        smooth_cache[cache_key] = smooth_linear_modules(
            attn.parent.pre_attn_norm,
            attn.qkv_proj,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=attn.qkv_proj_key),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=attn.qkv_proj_key),
            inputs=layer_cache[attn.q_proj_name].inputs if layer_cache else None,
            eval_inputs=layer_cache[attn.name].inputs if layer_cache else None,
            eval_module=attn,
            eval_kwargs=attn.filter_kwargs(layer_kwargs),
            develop_dtype=config.develop_dtype,
        )
        if not attn.parent.pre_attn_norm:
            ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(attn.qkv_proj)
    # endregion
    # region output projection
    needs_quant = config.enabled_ipts and config.ipts.is_enabled_for(attn.out_proj_key)
    needs_quant = needs_quant or (config.enabled_wgts and config.wgts.is_enabled_for(attn.out_proj_key))
    if config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(attn.out_proj_key) and needs_quant:
        logger.debug("- %s.%s", attn.name, attn.out_proj_rkey)
        cache_key = attn.o_proj_name
        smooth_cache[cache_key] = smooth_linear_modules(
            None if attn.config.linear_attn else attn.v_proj,
            attn.o_proj,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=attn.out_proj_key),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=attn.out_proj_key),
            inputs=layer_cache[cache_key].inputs if layer_cache else None,
            eval_inputs=layer_cache[cache_key].inputs if layer_cache else None,
            eval_module=attn.o_proj,
            num_heads=attn.config.num_query_heads,
            num_head_repeats=attn.config.num_head_repeats,
            develop_dtype=config.develop_dtype,
        )
        if attn.config.linear_attn:
            ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(attn.o_proj)
    # endregion
    num_experts = ffn.config.num_experts
    # region up projection
    needs_quant = config.enabled_ipts and config.ipts.is_enabled_for(ffn.up_proj_key)
    needs_quant = needs_quant or (config.enabled_wgts and config.wgts.is_enabled_for(ffn.up_proj_key))
    if config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(ffn.up_proj_key) and needs_quant:
        logger.debug("- %s.%s", ffn.name, ffn.up_proj_rkey)
        cache_key = ffn.name
        smooth_cache[cache_key] = smooth_linear_modules(
            ffn.parent.pre_ffn_norm,
            ffn.up_projs,
            scale=smooth_cache.get(cache_key, None),
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config.wgts, key=ffn.up_proj_key),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=ffn.up_proj_key),
            inputs=layer_cache[ffn.name].inputs if layer_cache else None,
            eval_inputs=layer_cache[ffn.name].inputs if layer_cache else None,
            eval_module=ffn,
            extra_modules=[ffn.moe_gate] if num_experts > 1 else None,
            develop_dtype=config.develop_dtype,
        )
        if not ffn.parent.pre_ffn_norm:
            hook = ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(ffn.up_projs)
            if num_experts > 1:
                hook.register(ffn.moe_gate)
    # endregion
    # region down projection
    needs_quant = config.enabled_ipts and config.ipts.is_enabled_for(ffn.down_proj_key)
    needs_quant = needs_quant or (config.enabled_wgts and config.wgts.is_enabled_for(ffn.down_proj_key))
    if config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(ffn.down_proj_key) and needs_quant:
        for expert_idx in range(num_experts):
            logger.debug("- %s.%s", ffn.expert_names[expert_idx], ffn.down_proj_rkey)
            cache_key = ffn.down_proj_names[expert_idx]
            smooth_cache[cache_key] = smooth_linear_modules(
                ffn.up_projs[expert_idx],
                ffn.down_projs[expert_idx],
                scale=smooth_cache.get(cache_key, None),
                config=config.smooth.proj,
                weight_quantizer=Quantizer(config.wgts, key=ffn.down_proj_key),
                input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=ffn.down_proj_key),
                inputs=layer_cache[ffn.down_proj_names[expert_idx]].inputs if layer_cache else None,
                eval_inputs=layer_cache[ffn.down_proj_names[expert_idx]].inputs if layer_cache else None,
                eval_module=ffn.down_projs[expert_idx],
                develop_dtype=config.develop_dtype,
            )
    # endregion
    tools.logging.Formatter.indent_dec()


@torch.inference_mode()
def smooth_llm(
    model: nn.Module | LlmModelStruct,
    /,
    config: LlmQuantConfig,
    tokenizer: PreTrainedTokenizer | None = None,
    smooth_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Smooth the large language model.

    Args:
        model (`nn.Module` or `LlmStruct`):
            Model to be smoothed.
        config (`LlmQuantConfig`):
            Quantization configuration.
        tokenizer (`PreTrainedTokenizer`, *optional*, defaults to `None`):
            Tokenizer.
        smooth_cache (`dict[str, torch.Tensor]`, *optional*, defaults to `None`):
            Smoothing scale caches.

    Returns:
        `dict[str, torch.Tensor]`:
            Dictionary mapping module names to smoothing scales.
    """
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.construct(model)
    assert isinstance(model, LlmModelStruct)
    smooth_cache = smooth_cache or {}
    if not smooth_cache:
        with tools.logging.redirect_tqdm():
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                config.calib.build_loader(tokenizer).iter_layer_activations(
                    model,
                    needs_inputs_fn=get_needs_inputs_fn(model=model, config=config),
                    needs_outputs_fn=get_needs_outputs_fn(model=model, config=config),
                ),
                desc="smoothing",
                leave=False,
                total=len(model.backbone_struct.layer_structs),
                dynamic_ncols=True,
            ):
                smooth_llm_layer(
                    layer=layer,
                    config=config,
                    smooth_cache=smooth_cache,
                    layer_cache=layer_cache,
                    layer_kwargs=layer_kwargs,
                )
    else:
        for layer in model.backbone_struct.layer_structs:
            smooth_llm_layer(layer=layer, config=config, smooth_cache=smooth_cache)  # 8.21 strucy.py这块相当于把标准的transformer库或者torch.nn库的一些标准结构抽象出来，供量化过程使用
    return smooth_cache
