import json
import os
import pdb

from typing import Dict, List, Tuple

from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import click
import cv2

# import mmcv
import numpy as np
import numpy.typing as npt
import streamlit as st

from PIL import Image, ImageColor
from tqdm import tqdm

from mmengine.utils import mkdir_or_exist


OUT_DIR = Path("/mnt/researchdev/danoneata/work/wood-generator/data/mmseg")
Key = namedtuple("Key", "annotator board_id")


DEFECT_CODES = {
    "Wane": 1,
    "Split": 2,
    "Knot_(Unsound)": 3,
    "Bark": 4,
    "Bark_(line)": 4,
    "Pith": 5,
    "Check": 6,
    "Knot_(Sound)": 7,
    "Stain": 8,
    "Decay": 9,
    "Shake": 10,
    "Manufactoring_Error": 11,
    "Hole": 12,
    "Other": 13,
    "Insect_Hole": 14,
}

COLORS = [
    "#000000",
    "#CC0000",
    "#A666E3",
    "#1BD4BB",
    "#8A99D3",
    "#D71B93",
    "#EE7F7F",
    "#D4BB1B",
    "#871D66",
    "#E3BE92",
    "#077D1B",
    "#B4DC91",
    "#1B88D4",
    "#1B88D4",
    "#1B88D4",
]


def extract_id_default(key):
    *id1, _ = key.board_id.split("-")
    return "-".join(id1)


def extract_id_fehrensen(key):
    return key.board_id.split("-")[0]


class Dataset:
    def __init__(self, name: str, version: int):
        self.name = name
        self.version = version

        self.base_dir = Path("/mnt/researchdev/data")
        self.base_dir = self.base_dir / name
        self.metadata_dir = self.base_dir / "metadata"
        self.image_dir = self.base_dir / "crop"
        self.annotation_dir = self.base_dir / "defects"

    def load_splits(self):
        path = self.base_dir / "split" / f"{self.name.lower()}_v0.{self.version}.txt"
        with open(path, "r") as f:
            return json.load(f)

    def load_annotation(self, key: Key) -> List[Dict]:
        with open(self.get_annotation_path(key), "r") as f:
            return json.load(f)

    def get_board_ids(self, annotator):
        path = os.path.join(self.annotation_dir, annotator)
        return [os.path.splitext(f)[0] for f in os.listdir(path)]

    def get_keys(self) -> List[Key]:
        annotators = os.listdir(self.annotation_dir)
        return [
            Key(annotator, board_id)
            for annotator in annotators
            for board_id in self.get_board_ids(annotator)
        ]

    def get_image_path(self, key: Key) -> Path:
        return self.image_dir / (key.board_id + ".jpg")

    def get_image_size(self, key: Key) -> Tuple[float, float]:
        image = Image.open(self.get_image_path(key))
        return image.size

    def get_annotation_path(self, key: Key) -> Path:
        return self.annotation_dir / key.annotator / (key.board_id + ".json")

    def load_image(self, key: Key):
        path = self.get_image_path(key)
        image = cv2.imread(str(path))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class Resize:
    def __call__(self, image):
        raise NotImplementedError


@dataclass
class Reduce(Resize):
    factor: int

    def __call__(self, image):
        # TODO Double check
        h, w = image.shape[:2]
        new_size = w // self.factor, h // self.factor
        image = cv2.resize(image, new_size)
        return image


@dataclass
class Fixed(Resize):
    width: int
    height: int


CONFIGS = {
    "fehrensen-v6": {
        "dataset": Dataset(name="fehrensen", version=6),
        "resize": Reduce(factor=2),
        "extract-id": extract_id_fehrensen,
    },
    "fehrensen-v7": {
        "dataset": Dataset(name="fehrensen", version=7),
        "resize": Reduce(factor=2),
        "extract-id": extract_id_fehrensen,
    },
}


def draw_annot_as_image(
    annot: Dict, image_size: Tuple[int, int]
) -> npt.NDArray[np.uint8]:
    # def draw_bbox(image, a):
    #     assert False

    def draw_polygon(image, a):
        points = np.array([(p["x"], p["y"]) for p in a["polygon"]])
        points = points.reshape(-1, 1, 2).astype(int)
        code = DEFECT_CODES[a["name"]]
        image = cv2.fillPoly(image, [points], (code, ))
        return image

    def draw_line(image, a):
        points = np.array([(p["x"], p["y"]) for p in a["line"]])
        points = points.reshape(-1, 1, 2).astype(int)
        code = DEFECT_CODES[a["name"]]
        image = cv2.polylines(image, [points], False, (code,), 10)
        return image

    def draw_point(image, a):
        point = [int(a["point"]["x"]), int(a["point"]["y"])]
        code = DEFECT_CODES[a["name"]]
        # negative thickness value mean filled
        image = cv2.circle(image, point, 5, (code,), -1)
        return image

    DRAW_FUNCS = {
        # "bbox": draw_bbox,
        "polygon": draw_polygon,
        "line": draw_line,
        "point": draw_point,
    }

    def get_shape(a):
        for shape in DRAW_FUNCS.keys():
            if shape in a:
                return shape
        raise ValueError("Unknown annotation type")

    # Decay should have highest priority
    LOW_PRIORITY = ["Stain"]

    w, h = image_size
    image = np.zeros((h, w), dtype=np.uint8)
    annot = sorted(annot, key=lambda a: a["name"] in LOW_PRIORITY)

    for a in annot:
        if a["name"] == "Red_Heart":
            continue
        shape = get_shape(a)
        image = DRAW_FUNCS[shape](image, a)

    return image


def show_annot(image):
    image_l = Image.fromarray(image, "L")
    image_p = image_l.convert("P")
    image_p.putpalette([rgb for c in COLORS for rgb in ImageColor.getcolor(c, "RGB")])
    image_out = np.array(image_p.convert("RGB"))
    st.image(image_out)


def prepare_images(dataset, resize, keys, out_dir):
    for key in tqdm(keys):
        image = dataset.load_image(key)
        image = resize(image)
        path = out_dir / (key.board_id + "-" + key.annotator + ".jpg")
        cv2.imwrite(str(path), image)


def prepare_annots(dataset, resize, keys, out_dir):
    for key in tqdm(keys):
        annot = dataset.load_annotation(key)
        image = draw_annot_as_image(annot, dataset.get_image_size(key))
        image = resize(image)
        path = out_dir / (key.board_id + "-" + key.annotator + ".png")
        cv2.imwrite(str(path), image)

        # st.image(str(dataset.get_image_path(key)))
        # show_annot(image)
        # pdb.set_trace()


@click.command()
@click.option("-c", "--config", "config_name", required=True)
def main(config_name):
    SPLITS = ["train", "validation", "test"]
    config = CONFIGS[config_name]

    dataset = config["dataset"]
    extract_id = config["extract-id"]
    resize = config["resize"]

    for split in SPLITS:
        for type_ in ["image", "annot"]:
            path = OUT_DIR / config_name / type_ / split
            mkdir_or_exist(path)

    split_to_ids = dataset.load_splits()
    keys = dataset.get_keys()

    for split in SPLITS:
        ids = set(split_to_ids[split])
        keys1 = [key for key in keys if extract_id(key) in ids]
        prepare_images(dataset, resize, keys1, OUT_DIR / config_name / "image" / split)
        prepare_annots(dataset, resize, keys1, OUT_DIR / config_name / "annot" / split)


if __name__ == "__main__":
    main()
