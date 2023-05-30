import math
import os
import os.path as osp
import time
import warnings
from argparse import ArgumentParser
from datetime import timedelta

import torch
import wandb
from dataset import SceneTextDataset
from custom_dataset import ValidDataset, collate_fn
from detect import detect
from deteval import calc_deteval_metrics
from east_dataset import EASTDataset
from model import EAST
from torch import cuda
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings(action="ignore")

SEED = 142857 * 18
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# np.random.seed(SEED)


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument("--exp_name", type=str, default="000")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "../data/medical"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "trained_models"))

    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=6)

    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epoch", type=int, default=150)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--ignore_tags", type=list, default=["masked", "excluded-region", "maintable", "stamp"])
    parser.add_argument("--validation", default=True)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


def do_training(
    exp_name,
    data_dir,
    model_dir,
    device,
    image_size,
    input_size,
    num_workers,
    batch_size,
    learning_rate,
    max_epoch,
    save_interval,
    ignore_tags,
    validation,
):
    train_dataset = SceneTextDataset(
        data_dir, split="train", image_size=image_size, crop_size=input_size, ignore_tags=ignore_tags
    )
    train_dataset = EASTDataset(train_dataset)
    num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    if validation:
        val_ratio = 2
        valid_dataset = ValidDataset(data_dir, split="valid", image_size=image_size, ignore_tags=ignore_tags)
        num_val_batches = math.ceil(len(train_dataset) / (batch_size // val_ratio))
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size // val_ratio,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(pretrained=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    mean_times = {"train": timedelta(), "valid": timedelta()}

    for epoch in range(max_epoch):
        epoch_loss, min_f1_score, train_start = 0, math.inf, time.time()

        model.train()
        pbar = tqdm(train_loader, total=num_batches)
        for img, gt_score_map, gt_geo_map, roi_mask in pbar:
            pbar.set_description("[Epoch {}]".format(epoch + 1))

            loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val

            val_dict = {
                "Cls loss": extra_info["cls_loss"],
                "Angle loss": extra_info["angle_loss"],
                "IoU loss": extra_info["iou_loss"],
            }

            pbar.set_postfix(val_dict)

        scheduler.step()

        mean_times["train"] += timedelta(seconds=time.time() - train_start)
        print(
            "Mean loss: {:.4f} | Train time: {}".format(
                epoch_loss / num_batches, timedelta(seconds=time.time() - train_start)
            )
        )
        wandb.log(
            {
                "Train Cls loss": extra_info["cls_loss"],
                "Train Angle loss": extra_info["angle_loss"],
                "Train IoU loss": extra_info["iou_loss"],
                "Train Mean loss": epoch_loss / num_batches,
            }
        )

        transcriptions_dict = {}
        pred_dict = {}
        gt_dict = {}
        f1_score = math.inf

        if epoch + 1 >= 10 and validation:
            valid_start = time.time()

            model.eval()
            pbar = tqdm(enumerate(valid_loader), total=num_val_batches)
            for i, (image, bbox) in pbar:
                pbar.set_description(f"[Epoch {epoch + 1} Validation]")

                pred = detect(model, image.permute(0, 2, 3, 1).contiguous().detach().cpu().numpy(), input_size)

                for j in range(len(image)):
                    transcriptions_dict[num_val_batches * i + j] = ["1"] * len(bbox[j])
                    pred_dict[num_val_batches * i + j] = pred[j]
                    gt_dict[num_val_batches * i + j] = bbox[j]

            metric = calc_deteval_metrics(pred_dict, gt_dict, transcriptions_dict)
            precision = metric["total"]["precision"]
            recall = metric["total"]["recall"]
            f1_score = metric["total"]["hmean"]
            val_time = timedelta(seconds=time.time() - valid_start)

            mean_times["valid"] += val_time
            print(
                f"F1 score: {f1_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f} | Valid time: {val_time}"
            )
            wandb.log(
                {
                    "Valid F1 Score": f1_score,
                    "Valid Precision": precision,
                    "Valid Recall": recall,
                }
            )

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(osp.join(model_dir, exp_name), f"{exp_name}_latest.pth")
            torch.save(model.state_dict(), ckpt_fpath)

        if f1_score < min_f1_score:
            print("Best F1 Score Renewal!")
            print("Save Model Weights...")
            ckpt_fpath = osp.join(osp.join(model_dir, exp_name), f"{exp_name}_bset.pth")
            torch.save(model.state_dict(), ckpt_fpath)
            min_f1_score = f1_score

    print("-" * 10)
    print(f"train mean time: {mean_times['train'] / max_epoch}")
    if validation:
        print(f"valid mean time: {mean_times['valid'] / max_epoch}")


def main(args):
    wandb.init(
        project="data-centric",
        entity="cv-18",
        name=args.exp_name,
    )
    wandb.config.update(args)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(osp.join(args.model_dir, args.exp_name), exist_ok=True)

    do_training(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    main(args)
