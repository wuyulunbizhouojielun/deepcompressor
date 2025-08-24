# -*- coding: utf-8 -*-
"""Large Language Model Rotation module."""

import gc

import torch
from tqdm import tqdm
from transformers import PreTrainedModel

from deepcompressor.calib.config import QuantRotationConfig
from deepcompressor.calib.rotate import (
    get_rotation_matrix,
    hadamard_in_channels,
    rotate_in_channels,
    rotate_out_channels,
    transform_norm_and_linear,
)
from deepcompressor.utils import tools

from ..nn import LlmModelStruct

__all__ = ["rotate_llm"]


@torch.inference_mode()
def rotate_llm(  # noqa: C901
    model: PreTrainedModel | LlmModelStruct,
    /,
    config: QuantRotationConfig,
    rotation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate the weights of the large language model.

    Args:
        model (`PreTrainedModel` or `LlmStruct`):
            Model to be rotated.
        config (`QuantRotationConfig`):
            Rotation configuration.
        rotation (`torch.Tensor` or `None`, *optional*, defaults to `None`):
            Rotation matrix.

    Returns:
        `torch.Tensor`:
            The rotation matrix.
    """
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.construct(model)
    assert isinstance(model, LlmModelStruct)
    devices: list[torch.device] = []
    dtypes: list[torch.dtype] = []
    linears: list[torch.nn.Linear] = []
    size: float = 0

    # 遍历所有linear 参数规模大于30B则先转移到cpu
    for m in model.module.modules():
        if isinstance(m, torch.nn.Linear):
            devices.append(m.weight.device)
            dtypes.append(m.weight.dtype)
            linears.append(m)
            size += m.weight.numel() / 1e9
    for linear in linears:
        linear.to(dtype=torch.float32, device="cpu" if size > 30 else None)
    
    # 迭代每个transformer block查看是否有rotate不支持的结构
    for block in model.iter_transformer_block_structs():
        assert not block.post_attn_norms, "Rotation is only supported for models without post-attention norms."
        assert not block.post_ffn_norm, "Rotation is only supported for models without post-FFN norms."

    logger = tools.logging.getLogger(f"{__name__}.Rotate")
    backbone = model.backbone_struct
    layers = backbone.layer_structs
    # region transform norm and linear
    if backbone.norm_in is None:
        if backbone.proj_in is None:
            prev_modules = [backbone.embed_tokens]
            prev_out_channels_dims = 1
            if backbone.embed_positions is not None:
                prev_modules.append(backbone.embed_positions)
        elif backbone.embed_positions is None:
            prev_modules = [backbone.proj_in]
            prev_out_channels_dims = 0
        else:
            prev_modules = [backbone.proj_in, backbone.embed_positions]
            prev_out_channels_dims = [0, 1]
    else:
        prev_modules = [backbone.norm_in]
        prev_out_channels_dims = 0

    # 8.20 这里是按Quarot的做法 RMSnorm LayerNorm的α融入邻近的weight矩阵
    with tools.logging.redirect_tqdm():
        for layer in tqdm(layers, desc="Transforming norm and linear", dynamic_ncols=True):
            logger.debug(f"- Transforming norm and linear in {layer.name}")
            transform_norm_and_linear(
                parent=layer.module,
                norm_name=layer.pre_attn_norm_rname,
                next_modules=layer.attn_struct.qkv_proj,
                prev_modules=prev_modules,
                prev_out_channels_dims=prev_out_channels_dims,
            )
            prev_modules = [layer.attn_struct.out_proj]
            prev_out_channels_dims = 0
            transform_norm_and_linear(
                parent=layer.module,
                norm_name=layer.pre_ffn_norm_rname,
                next_modules=layer.ffn_struct.up_projs
                + ([layer.ffn_struct.moe_gate] if layer.ffn_struct.moe_gate is not None else []),
                prev_modules=prev_modules,
                prev_out_channels_dims=prev_out_channels_dims,
            )
            prev_modules = layer.ffn_struct.down_projs
            prev_out_channels_dims = 0
            gc.collect()
            torch.cuda.empty_cache()
    logger.debug(f"- Transforming {backbone.norm_out_name}")
    transform_norm_and_linear(
        parent=backbone.module,
        norm_name=backbone.norm_out_rname,
        next_modules=[model.head if backbone.proj_out is None else backbone.proj_out],
        prev_modules=prev_modules,
        prev_out_channels_dims=prev_out_channels_dims,
    )
    # endregion
    if rotation is None:
        rotation = get_rotation_matrix(backbone.config.num_channels, random=config.random)  # 直接参考Quarot 获取Hardmard矩阵
    # region rotate embeddings
    # 这里就直接左乘右乘得到的rotation matrix
    if backbone.proj_in is None:
        logger.debug(f"- Rotating {backbone.embed_tokens_name}")
        weight = backbone.embed_tokens.weight
        rotation = rotation.to(weight.device)
        rotate_in_channels(weight, rotation=rotation)
    else:
        logger.debug(f"- Rotating {backbone.proj_in_name} (out)")
        weight = backbone.proj_in.weight
        rotation = rotation.to(weight.device)
        rotate_out_channels(weight, rotation=rotation, bias=backbone.proj_in.bias)
    if backbone.embed_positions is not None:
        logger.debug(f"- Rotating {backbone.embed_positions_name}")
        weight = backbone.embed_positions.weight
        rotation = rotation.to(weight.device)
        rotate_in_channels(weight, rotation=rotation)
    # endregion
    down_proj = []
    # region rotate backbone layers
    head_rotation = get_rotation_matrix(model.config.num_head_channels, random=config.random)  # 针对注意力头维度，细粒度不同
    with tools.logging.redirect_tqdm():
        for layer in tqdm(layers, desc="Rotating backbone layers", dynamic_ncols=True):
            logger.debug(f"- Rotating {layer.name}")
            tools.logging.Formatter.indent_inc()
            attn, ffn = layer.attn_struct, layer.ffn_struct
            for proj_name, proj in zip(attn.qkv_proj_names, attn.qkv_proj, strict=True):
                logger.debug(f"- Rotating {proj_name} (in)")
                rotation = rotation.to(proj.weight.device)
                rotate_in_channels(proj.weight, rotation=rotation)
            logger.debug(f"- Rotating {attn.out_proj_name} (out)")
            rotation = rotation.to(attn.out_proj.weight.device)
            rotate_out_channels(attn.out_proj.weight, rotation=rotation, bias=attn.out_proj.bias)
            # 8.21 这里config.transformers对应配置文件里的rotation/transformer
            if attn.out_proj_key in config.transforms:
                logger.debug(f"- Rotating {attn.v_proj_name} (out)")
                rotate_out_channels(attn.v_proj.weight, rotation=head_rotation, bias=attn.v_proj.bias) # 左乘 输出通道变换
                logger.debug(f"- Rotating {attn.o_proj_name} (in)")
                rotate_in_channels(attn.o_proj.weight, rotation=head_rotation)  # 右乘 输入通道变换
            for fc_name, fc in zip(ffn.up_proj_names, ffn.up_projs, strict=True):
                logger.debug(f"- Rotating {fc_name} (in)")
                rotation = rotation.to(fc.weight.device)
                rotate_in_channels(fc.weight, rotation=rotation)
            if ffn.moe_gate is not None:
                logger.debug(f"- Rotating {ffn.moe_gate_name} (in)")
                rotation = rotation.to(ffn.moe_gate.weight.device)
                rotate_in_channels(ffn.moe_gate.weight, rotation=rotation)
            for fc_name, fc in zip(ffn.down_proj_names, ffn.down_projs, strict=True):
                logger.debug(f"- Rotating {fc_name} (out)")
                rotation = rotation.to(fc.weight.device)
                rotate_out_channels(fc.weight, rotation=rotation, bias=fc.bias)
            if ffn.down_proj_key in config.transforms:
                down_proj.extend(ffn.down_projs)
            tools.logging.Formatter.indent_dec()
            gc.collect()
            torch.cuda.empty_cache()
    if backbone.proj_out is not None:
        logger.debug(f"- Rotating {backbone.proj_out_name} (in)")
        weight = backbone.proj_out.weight
        rotation = rotation.to(weight.device)
        rotate_in_channels(weight, rotation=rotation)
        logger.debug(f"- Rotating {backbone.proj_out_name} (out)")
        rotation = rotation.to(weight.device)
        rotate_out_channels(weight, rotation=rotation, bias=backbone.proj_out.bias)
    # endregion
    if down_proj:
        logger.debug(f"- Applying Hadamard transform on {backbone.name}.down_proj (in)")
        hadamard_in_channels(down_proj)
    if backbone.proj_out is not None:
        logger.debug(f"- Rotating {backbone.proj_out_name} (in)")
        weight = backbone.proj_out.weight
    else:
        logger.debug(f"- Rotating {model.head_name} (in)")
        weight = model.head.weight
    rotation = rotation.to(weight.device)
    rotate_in_channels(weight, rotation=rotation)
    for device, dtype, linear in zip(devices, dtypes, linears, strict=True):
        linear.to(device=device, dtype=dtype)
    return rotation.cpu()
