from mmcv.utils import Registry, build_from_cfg

from mmdet.models.builder import BACKBONES, HEADS, LOSSES, NECKS

FUSIONMODELS = Registry("fusion_models")
BEVFORMERMODELS = Registry("fusion_models")
VTRANSFORMS = Registry("vtransforms")
FUSERS = Registry("fusers")
SEG_ENCODER = Registry('seg_encoder')


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_vtransform(cfg):
    return VTRANSFORMS.build(cfg)


def build_fuser(cfg):
    return FUSERS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_seg_encoder(cfg, **default_args):
    """Builder of box sampler."""
    return build_from_cfg(cfg, SEG_ENCODER, default_args)


def build_fusion_model(cfg, train_cfg=None, test_cfg=None):
        return FUSIONMODELS.build(
            cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
        )


def build_model(cfg, train_cfg=None, test_cfg=None):
    if cfg.type == "BEVFusion":
        return build_fusion_model(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    elif cfg.type == "BEVFormer":
        return BEVFORMERMODELS.build(
            cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
        )
    else:
        raise "model type wrong!"

