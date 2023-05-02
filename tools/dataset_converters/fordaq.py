import json
import os

from dataclases import dataclass
from pathlib import Path

import click
import mmcv
import numpy as np

from mmengine.utils import ProgressBar, mkdir_or_exist


BASE_DIR = Path("/mnt/researchdev/data")
OUT_DIR = Path("/mnt/researchdev/danoneata/work/wood-generator/data")


def extract_id_default(annot):
    *id1, _ = annot["filename"].split("-")
    return "-".join(id1)


def extract_id_fehrensen(annot):
    return annot["filename"].split("-")[0]


class Resize:
    pass


@dataclass
class Reduce(Resize):
    factor: int


@dataclass
class Fixed(Resize):
    width: int
    height: int


CONFIGS = {
    "fehrensen-v6": {
        "dataset-name": "fehrensen",
        "version": 6,
        "resize": Reduce(factor=2),
        "extract-id": extract_id_fehrensen,
    },
}


def load_splits(dataset, version):
    filename = f"{dataset.lower()}_v0.{version}.txt"
    path_splits = BASE_DIR / dataset / "split" / filename
    with open(path_splits, "r") as f:
        return json.load(f)


def write_filelist(dataset, name, data):
    folder = os.path.join("data", dataset.lower(), "filelists")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".txt")
    with open(path, "w") as f:
        for datum in data:
            annot, filename = datum
            id1, _ = os.path.splitext(filename)
            f.write("{},{}\n".format(annot, id1))


@click.command()
@click.option("-c", "--config", "config_name", required=True)
def main(config_name):
    SPLITS = ["train", "validation", "test"]
    config = CONFIGS[config_name]

    dataset = config["dataset-name"]
    version = config["version"]
    extract_id = config["extract-id"]

    print("Making directories...")
    for split in SPLITS:
        for type_ in ["image", "annot"]:
            path = OUT_DIR / config_name / type_ / split
            mkdir_or_exist(path)

    split_to_ids = load_splits(dataset, version)
    annotators = os.listdir(BASE_DIR / dataset / "defects")

    annotations = [
        {
            "annotator": a,
            "filename": p,
        }
        for a in annotators
        for p in os.listdir(BASE_DIR / dataset / "defects" / a)
    ]

    for split in SPLITS:
        ids = set(split_to_ids[split])
        annots = [annot for annot in annotations if extract_id(annot) in ids]
        print(len(annots))
        # filelist_name = "v{}-{}".format(version, split)
        # write_filelist(dataset, filelist_name, annots)


if __name__ == "__main__":
    main()
