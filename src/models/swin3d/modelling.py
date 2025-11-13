import torch
from transformers import PreTrainedModel, ImageProcessingMixin
from transformers.modeling_outputs import ImageClassifierOutput
from .configuration import Swin3DConfig
from torchvision.models.video import (
    swin3d_t,
    swin3d_s,
    swin3d_b,
    Swin3D_T_Weights,
    Swin3D_S_Weights,
    Swin3D_B_Weights,
)


MODELS = {
    "swin3d_t": (swin3d_t, Swin3D_T_Weights),
    "swin3d_s": (swin3d_s, Swin3D_S_Weights),
    "swin3d_b": (swin3d_b, Swin3D_B_Weights),
}


class Swin3DImageProcessor(ImageProcessingMixin):
    def __init__(self, config: Swin3DConfig = Swin3DConfig(), **kwargs) -> None:
        super().__init__(**kwargs)
        _, weights_class = MODELS[config.arch]
        weights = weights_class.verify(config.pretrained)

        self.mean = weights.transforms.keywords["mean"]
        self.std = weights.transforms.keywords["std"]

        self.min_resize_size = weights.transforms.keywords["resize_size"][0]
        self.max_resize_size = 320

        height, width = weights.transforms.keywords["crop_size"]
        self.size = {
            "height": height,
            "width": width,
        }

        self.num_frames = config.num_frames


class Swin3DForVideoClassification(PreTrainedModel):
    config_class = Swin3DConfig

    def __init__(
        self,
        config: Swin3DConfig = Swin3DConfig(),
        label2id: dict = None,
        id2label: dict = None,
    ) -> None:
        super().__init__(config=config)
        model_class, _ = MODELS[config.arch]
        self.model = model_class(config.pretrained)
        self.label2id = label2id if label2id is not None else config.label2id
        self.id2label = id2label if id2label is not None else config.id2label
        self.num_classes = len(self.label2id)

        for i, param in enumerate(self.model.parameters()):
            if i >= config.num_frozen_layers:
                break
            param.requires_grad = False

        if self.model.num_classes != self.num_classes:
            self.model.head = torch.nn.Linear(
                in_features=self.model.head.in_features,
                out_features=self.num_classes,
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = self.model(pixel_values.permute(0, 2, 1, 3, 4))
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return ImageClassifierOutput(loss=loss, logits=logits)
        return ImageClassifierOutput(logits=logits)
