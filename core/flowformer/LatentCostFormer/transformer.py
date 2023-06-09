import torch
import torch.nn as nn

from ..encoders import twins_svt_large
from .encoder import MemoryEncoder
from .decoder import MemoryDecoder
from .cnn import BasicEncoder

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class FlowFormer(nn.Module):
    def __init__(self, cfg, deq_cfg):
        super(FlowFormer, self).__init__()
        self.cfg = cfg
        self.deq_cfg = deq_cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg, deq_cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(
                pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(
                output_dim=256, norm_fn='instance')

    def forward(
        self, image1, image2, output=None, flow_init=None, sradius_mode=False,
        cached_result=None, **kwargs,
    ):
        # Following https://github.com/princeton-vl/RAFT/
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            image_inp = torch.cat([image1, image2], dim=1)
        else:
            image_inp = image1

        with autocast(enabled=self.deq_cfg.mixed_precision):
            context = self.context_encoder(image_inp)
            cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions = self.memory_decoder(
            cost_memory, context, data, flow_init=flow_init,
            sradius_mode=sradius_mode, cached_result=cached_result, **kwargs,
        )

        return flow_predictions
