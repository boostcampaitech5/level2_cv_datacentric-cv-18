import json
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset import *


class ValidDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="valid",
        image_size=2048,
        ignore_tags=[],
        ignore_under_threshold=10,
        drop_under_threshold=1,
        normalize=True,
    ):
        with open(osp.join(root_dir, "ufo/{}.json".format(split)), "r") as f:
            anno = json.load(f)

        self.anno = anno
        self.image_fnames = sorted(anno["images"].keys())
        self.image_dir = osp.join(root_dir, "img", split)

        self.image_size = image_size
        self.normalize = normalize

        self.ignore_tags = ignore_tags

        self.drop_under_threshold = drop_under_threshold
        self.ignore_under_threshold = ignore_under_threshold

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_fpath = osp.join(self.image_dir, image_fname)

        vertices, labels = [], []
        for word_info in self.anno["images"][image_fname]["words"].values():
            word_tags = word_info["tags"]

            ignore_sample = any(elem for elem in word_tags if elem in self.ignore_tags)
            num_pts = np.array(word_info["points"]).shape[0]

            # skip samples with ignore tag and
            # samples with number of points greater than 4
            if ignore_sample or num_pts > 4:
                continue

            vertices.append(np.array(word_info["points"]).flatten())
            labels.append(int(not word_info["illegibility"]))
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

        vertices, labels = filter_vertices(
            vertices, labels, ignore_under=self.ignore_under_threshold, drop_under=self.drop_under_threshold
        )

        image = Image.open(image_fpath)
        image, vertices = valid_resize_img(image, vertices, self.image_size)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = np.array(image)
        word_bboxes = np.reshape(vertices, (-1, 4, 2))

        image = torch.FloatTensor(image).permute(2, 0, 1)
        word_bboxes = torch.from_numpy(word_bboxes)
        return image, word_bboxes


def collate_fn(batch):
    images = []
    labels = []
    for sample in batch:
        images.append(sample[0])
        labels.append(sample[1])
    return torch.stack(images, 0), labels
