
import torch
import torch.nn as nn

from ...core import register

__all__ = [
    "DFINE",
]


@register()
class DFINE(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
        "VisualClassifier",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        VisualClassifier: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.VisualClassifier = VisualClassifier


    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        outputs = self.decoder(x, targets)
        outputs1= self.VisualClassifier(x, outputs, targets)

        return outputs, outputs1

    @torch.no_grad()
    def sample(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        outputs = self.decoder(x)

        outputs= self.VisualClassifier(x, outputs, targets)

        return outputs, x
    

    def get_losses(self, outputs, ref_cls_outputs=None, old_cls_outputs=None):
        return self.VisualClassifier.get_losses(outputs, ref_cls_outputs=ref_cls_outputs, old_cls_outputs=old_cls_outputs)

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
