import os
import pickle
from argparse import ArgumentParser

from dataset import SceneTextDataset
from east_dataset import EASTDataset
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "../data/medical"))
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ignore_tags", type=list, default=["masked", "excluded-region", "maintable", "stamp"])

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


def pickle_dataset(dataset, type="train"):
    for img_num, data in tqdm(enumerate(iter(dataset)), total=len(dataset)):
        with open(file=f"/opt/ml/input/data/medical/img/{type}_pickled1/{img_num}.pkl", mode="wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    args = parse_args()

    train_dataset = SceneTextDataset(
        args.data_dir, split="train", image_size=args.image_size, crop_size=args.input_size, ignore_tags=args.ignore_tags
    )
    train_dataset = EASTDataset(train_dataset)
    pickle_dataset(train_dataset, type='train')