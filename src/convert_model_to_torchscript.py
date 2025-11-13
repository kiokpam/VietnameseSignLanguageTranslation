import torch
import logging
from pathlib import Path
from argparse import Namespace
from configs import ModelConfig
from simple_parsing import ArgumentParser
from tools import load_model, get_input_shape
from utils import config_logger, upload_to_hf


def get_args() -> Namespace:
    parser = ArgumentParser(description="Export model to Torchscript format")
    parser.add_arguments(ModelConfig, "model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/torchscript",
        help="Path to output Torchscript file",
    )
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload Torchscript file to Hugging Face",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    model_config = args.model
    logging.info(model_config)

    output_name = model_config.pretrained.split("/")[-1]
    output_file = Path(args.output_dir) / f"{output_name}.pt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    _, processor, model = load_model(model_config)
    logging.info("Model loaded")

    batch_size = 1
    input_shape = get_input_shape(model_config.arch, processor, batch_size)

    torch.jit.save(
        m=torch.jit.trace(
            func=model,
            example_inputs=[torch.randn(input_shape)],
            strict=False,
        ),
        f=str(output_file),
    )
    logging.info(f"Model exported to {str(output_file)}")

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
