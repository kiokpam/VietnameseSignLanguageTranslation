import torch
import logging
from pathlib import Path
from argparse import Namespace
from configs import ModelConfig
from dataclasses import dataclass, field
from simple_parsing import ArgumentParser
from tools import load_model, get_input_shape
from utils import (
    POSE_BASED_MODELS,
    RGB_BASED_MODELS,
    config_logger,
    upload_to_hf,
)


def get_args() -> Namespace:
    parser = ArgumentParser(description="Export model to ONNX format")
    parser.add_arguments(ModelConfig, "model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/onnx",
        help="Path to output ONNX file",
    )
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload ONNX file to Hugging Face",
    )
    return parser.parse_args()


@dataclass
class ONNXConfig:
    f: str
    dynamic_axes: dict
    input_names: list
    output_names: list = field(default_factory=lambda: ["logits"])
    do_constant_folding: float = True
    opset_version: int = 14


@dataclass
class RGBONNXConfig(ONNXConfig):
    input_names: list = field(default_factory=lambda: ["pixel_values"])
    dynamic_axes: dict = field(
        default_factory=lambda: {
            "pixel_values": {
                0: "batch_size",
                1: "num_frames",
                2: "num_channels",
                3: "height",
                4: "width",
            }
        }
    )


@dataclass
class PoseONNXConfig(ONNXConfig):
    input_names: list = field(default_factory=lambda: ["poses"])


@dataclass
class SLGCNONNXConfig(PoseONNXConfig):
    dynamic_axes: dict = field(
        default_factory=lambda: {
            "poses": {
                0: "batch_size",
                1: "num_channels",
                2: "num_frames",
                3: "num_points",
                4: "num_people",
            }
        }
    )


@dataclass
class SPOTERONNXConfig(PoseONNXConfig):
    dynamic_axes: dict = field(
        default_factory=lambda: {
            "poses": {
                0: "batch_size",
                1: "num_frames",
                2: "num_points",
                3: "num_channels",
            }
        }
    )


def main(args: Namespace) -> None:
    model_config = args.model
    logging.info(model_config)

    output_name = model_config.pretrained.split("/")[-1]
    output_file = Path(args.output_dir) / f"{output_name}.onnx"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    _, processor, model = load_model(model_config)
    logging.info("Model loaded")

    batch_size = 1
    input_shape = get_input_shape(model_config.arch, processor, batch_size)

    if model_config.arch in RGB_BASED_MODELS:
        config_class = RGBONNXConfig
    elif model_config.arch in POSE_BASED_MODELS:
        if model_config.arch == "spoter":
            config_class = SPOTERONNXConfig
        elif model_config.arch in ["sl_gcn", "dsta_slr"]:
            config_class = SLGCNONNXConfig
        else:
            logging.error(f"Model {model_config.arch} is not supported")
            exit(1)
    else:
        logging.error(f"Model {model_config.arch} is not supported")
        exit(1)
    config = config_class(f=str(output_file))
    logging.info("Config loaded")

    torch.onnx.export(model, torch.randn(*input_shape), **vars(config))
    logging.info(f"Model exported to {config.f}")

    if args.upload_to_hf:
        upload_to_hf(
            model_config.pretrained,
            output_file,
            output_file.name,
            "model",
        )
        logging.info(f"Model uploaded to {model_config.pretrained}")


if __name__ == "__main__":
    args = get_args()
    config_logger()
    main(args=args)
