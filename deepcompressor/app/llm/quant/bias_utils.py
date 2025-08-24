import torch
from collections import OrderedDict
from typing import Optional
# 假设实际类路径，根据你的项目结构调整导入
from deepcompressor.data.cache import TensorsCache, TensorCache  # 注意导入方式

@torch.no_grad()
def bias_subtracted_tensors_cache(tcache: Optional[TensorsCache], bias: torch.Tensor) -> Optional[TensorsCache]:
    """
    返回新的 TensorsCache, 其内部每个 TensorCache.data (list[Tensor]) 中的每个张量按最后一维减去 bias.
    不修改原 tcache.
    """
    if tcache is None:
        return None
    first_tc = tcache.front()
    bias = bias.to(first_tc.data[0].device)
    new_dict = OrderedDict()
    for key, tc in tcache.items():
        new_data_list = []
        for t in tc.data:
            if t.dim() >= 2 and t.size(-1) == bias.numel():
                new_data_list.append(t - bias)
            else:
                new_data_list.append(t)
        new_tc = TensorCache(
            data=new_data_list,
            channels_dim=getattr(tc, "channels_dim", 1),
            reshape=getattr(tc, "reshape", None),
            num_cached=getattr(tc, "num_cached", 0),
            num_total=getattr(tc, "num_total", 0),
            num_samples=getattr(tc, "num_samples", 0),
            orig_device=getattr(tc, "orig_device", new_data_list[0].device),
        )
        new_dict[key] = new_tc
    return TensorsCache(new_dict)