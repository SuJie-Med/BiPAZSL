from .BiPAZSL import build_BiPAZSL

_GZSL_META_ARCHITECTURES = {
    "Model": build_BiPAZSL,
}

def build_gzsl_pipeline(cfg):
    meta_arch = _GZSL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
