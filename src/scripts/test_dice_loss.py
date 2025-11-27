import logging
from omegaconf import OmegaConf
import os
import sys
from typing import Any

from dinov3.eval.segmentation.config import SegmentationConfig
from dinov3.eval.segmentation.eval import test_segmentation
from dinov3.eval.segmentation.train import train_segmentation
from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
from functools import partial
import logging
import numpy as np
import os
import random

import torch
import torch.distributed as dist

from dinov3.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader, make_dataset
import dinov3.distributed as distributed
from dinov3.eval.segmentation.eval import evaluate_segmentation_model
from dinov3.eval.segmentation.loss import MultiSegmentationLoss
from dinov3.eval.segmentation.metrics import SEGMENTATION_METRICS
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.schedulers import build_scheduler
from dinov3.eval.segmentation.transforms import make_segmentation_eval_transforms, make_segmentation_train_transforms
from dinov3.logging import MetricLogger, SmoothedValue

from dinov3.hub.backbones import dinov3_vitl16, dinov3_vitb16
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-semantic-segmentation.csv"
MAIN_METRICS = ["mIoU"]
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def reverse_imagenet_normalization(tensor):

    img = tensor.clone()

    # Unnormalize: img = img * std + mean
    for c, (m, s) in enumerate(zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)):
        img[c] = img[c] * s + m

    # Clip and convert to numpy
    img = img.clamp(0, 1)  # ensure valid range
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    return img_np

class MaskToLongTensor:
    def __call__(self, image, target):
        mask = torch.as_tensor(np.array(target), dtype=torch.long)
        return image, mask

def make_bop_segmentation_transforms(img_size):
    return v2.Compose([
        # 1. Resize image AND mask
        v2.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
        
        # 2. Convert image to tensor
        v2.ToTensor(),
        
        # 3. Normalize image
        v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        
        # 4. Convert mask to long tensor (no distortion!)
        MaskToLongTensor(),
    ])

class InfiniteDataloader:
    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self.data_iterator = iter(dataloader)
        self.sampler = dataloader.sampler
        if not hasattr(self.sampler, "epoch"):
            self.sampler.epoch = 0  # type: ignore

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.dataloader)

    def __next__(self):
        try:
            data = next(self.data_iterator)
        except StopIteration:
            self.sampler.epoch += 1
            self.data_iterator = iter(self.dataloader)
            data = next(self.data_iterator)
        return data


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.
    The seed of each worker equals to num_worker * rank + worker_id + user_seed
    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def test_metrics(
    backbone,
    config,
):
    assert config.decoder_head.type == "linear", (
        "Only linear head is supported for training"
    )
    # 1- load the segmentation decoder
    logger.info("Initializing the segmentation model")
    segmentation_model = build_segmentation_decoder(
        backbone,
        config.decoder_head.backbone_out_layers,
        "linear",
        num_classes=config.decoder_head.num_classes,
        autocast_dtype=config.model_dtype.autocast_dtype,
        dropout=config.decoder_head.dropout,
    )
    global_device = distributed.get_rank()
    local_device = torch.cuda.current_device()
    # segmentation_model = torch.nn.parallel.DistributedDataParallel(
    #     segmentation_model.to(local_device), device_ids=[local_device]
    # )  # should be local rank
    segmentation_model = segmentation_model.to(local_device)

    model_parameters = filter(
        lambda p: p.requires_grad, segmentation_model.parameters()
    )
    logger.info(
        f"Number of trainable parameters: {sum(p.numel() for p in model_parameters)}"
    )

    # 2- create data transforms + dataloaders
    train_transforms = make_bop_segmentation_transforms(img_size=config.transforms.train.img_size)
    val_transforms = make_bop_segmentation_transforms(img_size=config.transforms.eval.img_size)

    # train_transforms = make_segmentation_train_transforms(
    #     img_size=config.transforms.train.img_size,
    #     random_img_size_ratio_range=config.transforms.train.random_img_size_ratio_range,
    #     crop_size=config.transforms.train.crop_size,
    #     flip_prob=config.transforms.train.flip_prob,
    #     reduce_zero_label=config.eval.reduce_zero_label,
    #     mean=config.transforms.mean,
    #     std=config.transforms.std,
    # )
    # val_transforms = make_segmentation_eval_transforms(
    #     img_size=config.transforms.eval.img_size,
    #     inference_mode=config.eval.mode,
    #     mean=config.transforms.mean,
    #     std=config.transforms.std,
    # )

    train_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.train}:root={config.datasets.root}",
            transforms=train_transforms,
        )
    )
    train_sampler_type = None
    if distributed.is_enabled():
        train_sampler_type = SamplerType.DISTRIBUTED
    init_fn = partial(
        worker_init_fn,
        num_workers=config.num_workers,
        rank=global_device,
        seed=config.seed + global_device,
    )
    train_dataloader = InfiniteDataloader(
        make_data_loader(
            dataset=train_dataset,
            batch_size=config.bs,
            num_workers=config.num_workers,
            sampler_type=train_sampler_type,
            shuffle=True,
            persistent_workers=False,
            worker_init_fn=init_fn,
        )
    )

    val_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.val}:root={config.datasets.root}",
            transforms=val_transforms,
        )
    )
    val_sampler_type = None

    val_dataloader = make_data_loader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        sampler_type=val_sampler_type,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )

    # 3- define and create scaler, optimizer, scheduler, loss
    scaler = None
    if config.model_dtype.autocast_dtype is not None:
        scaler = torch.amp.GradScaler("cuda")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": filter(
                    lambda p: p.requires_grad, segmentation_model.parameters()
                ),
                "lr": config.optimizer.lr,
                "betas": (config.optimizer.beta1, config.optimizer.beta2),
                "weight_decay": config.optimizer.weight_decay,
            }
        ]
    )
    scheduler = build_scheduler(
        config.scheduler.type,
        optimizer=optimizer,
        lr=config.optimizer.lr,
        total_iter=config.scheduler.total_iter,
        constructor_kwargs=config.scheduler.constructor_kwargs,
    )
    criterion = MultiSegmentationLoss(
        diceloss_weight=config.train.diceloss_weight,
        celoss_weight=config.train.celoss_weight,
    )
    total_iter = config.scheduler.total_iter
    global_step = 0
    global_best_metric_values = {metric: 0.0 for metric in SEGMENTATION_METRICS}

    ## Debugging
    train_dataloader_iter = iter(train_dataloader)
    batch = next(train_dataloader_iter)

    pred = segmentation_model(batch[0].to(local_device))
    # pred is torch.Size([2, 150, 32, 42])

    # sample_loss = criterion(pred, batch[1][1].to(local_device).squeeze())

    # print(batch)
    image = batch[0][0].detach().cpu().numpy()
    mask = batch[1][1].detach().cpu().numpy()
    mask = mask[0].squeeze()
    # mask = np.transpose(mask, (1, 2, 0))

    import cv2
    mask = (mask*255).astype(np.uint8)
    cv2.imwrite("output/train_image.png", reverse_imagenet_normalization(torch.tensor(image)))
    cv2.imwrite("output/train_mask.png", mask)
    print("exit test")


def load_model_from_weights(config: SegmentationConfig):
    weights_path = config.backbone_custom.weights_path

    model = None
    if config.backbone_custom.type == "dinov3_vitl16":
        model = dinov3_vitl16(weights=weights_path, pretrained=True)
    elif config.backbone_custom.type == "dinov3_vitb16":
        model = dinov3_vitb16(weights=weights_path, pretrained=True)
    else:
        raise NotImplementedError(f"Backbone {config.backbone.type} not implemented")

    print(f"Loaded custom backbone from {weights_path}")
    return model


def run_segmentation_with_dinov3(
    backbone,
    config,
):
    if config.load_from:
        logger.info("Testing model performance on a pretrained decoder head")
        return test_segmentation(backbone=backbone, config=config)
    assert config.decoder_head.type == "linear", (
        "Only linear head is supported for training"
    )
    return train_segmentation(backbone=backbone, config=config)


def benchmark_launcher(eval_args: dict[str, object]) -> dict[str, Any]:
    if "config" not in eval_args:  # using a config yaml file, useful for training
        raise ValueError("Config file must be provided in eval_args")

    base_config_path = eval_args.pop("config")
    output_dir = eval_args["output_dir"]
    base_config = OmegaConf.load(base_config_path)
    structured_config = OmegaConf.structured(SegmentationConfig)
    dataclass_config: SegmentationConfig = OmegaConf.to_object(
        OmegaConf.merge(
            structured_config,
            base_config,
            OmegaConf.create(eval_args),
        )
    )

    backbone = None
    backbone = load_model_from_weights(dataclass_config)
    logger.info(f"Segmentation Config:\n{OmegaConf.to_yaml(dataclass_config)}")
    segmentation_file_path = os.path.join(output_dir, "segmentation_config.yaml")
    OmegaConf.save(config=dataclass_config, f=segmentation_file_path)

    logger.info("Running segmentation with DINOv3...")
    results_dict = test_metrics(
        backbone=backbone, config=dataclass_config
    )
    # write_results(results_dict, output_dir, RESULTS_FILENAME)
    # return results_dict


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    eval_args = cli_parser(argv)

    benchmark_launcher(eval_args=eval_args)


if __name__ == "__main__":
    sys.exit(main())
