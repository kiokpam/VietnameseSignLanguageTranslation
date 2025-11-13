from pathlib import Path
from typing import Union
from datasets import DatasetDict, Dataset
from pytorchvideo.data import LabeledVideoDataset
from transformers import FeatureExtractionMixin, ImageProcessingMixin
from .base_dataset import BaseDataset
from .hf_builders import load_visl_98


class VISL98Dataset(BaseDataset):
    def _load_from_local(self, data_dir: str, **kwargs) -> Dataset:
        data_dir = Path(data_dir)
        meta_file = data_dir / "meta.json"
        gloss2id_file = data_dir / "gloss.csv"
        data_dir = data_dir / "data"

        train_df, test_df, gloss2id = load_visl_98(
            meta_file, gloss2id_file, data_dir
        )
        id2gloss = {v: k for k, v in gloss2id.items()}

        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df),
        })

        return dataset, gloss2id, id2gloss

    def get_split(
        self, split: str,
        processor: Union[ImageProcessingMixin, FeatureExtractionMixin]
    ) -> LabeledVideoDataset:
        if split == "validation":
            split = "test"
        return super().get_split(split, processor)
