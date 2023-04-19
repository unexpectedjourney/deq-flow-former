def build_flowformer(cfg, deq_cfg):
    name = cfg.transformer
    if name == 'latentcostformer':
        from .LatentCostFormer.transformer import FlowFormer
    else:
        raise ValueError(f"FlowFormer = {name} is not a valid architecture!")

    return FlowFormer(cfg[name], deq_cfg)
