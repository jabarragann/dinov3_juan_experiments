import os
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

# from .decoders import Decoder, DenseTargetDecoder, ImageDataDecoder
# from .extended import ExtendedVisionDataset
from dinov3.data.datasets.decoders import Decoder, DenseTargetDecoder, ImageDataDecoder
from dinov3.data.datasets.extended import ExtendedVisionDataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def dirname(self) -> str:
        return {
            _Split.TRAIN: "train_real",
            _Split.VAL: "train_real",
            _Split.TEST: "test",
        }[self]


def _file_to_segmentation_path(file_name: str, segm_base_path: str) -> str:
    file_name_noext = os.path.splitext(file_name)[0]
    return os.path.join(segm_base_path, file_name_noext + ".png")


def _load_segmentation(root: str, split_file_names: List[str]):
    segm_base_path = "annotations"
    segmentation_paths = [
        _file_to_segmentation_path(file_name, segm_base_path)
        for file_name in split_file_names
    ]
    return segmentation_paths


class tudl_bop(ExtendedVisionDataset):
    Split = Union[_Split]
    Labels = Union[Image.Image]

    def __init__(
        self,
        split: "tudl_bop.Split",
        root: Optional[str] = None,
        scene_id: Optional[int] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = DenseTargetDecoder,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )

        assert root is not None
        assert scene_id is not None
        root_plib = Path(root)
        self.image_paths, self.target_paths = self.load_file_paths(
            root_plib, split, scene_id
        )

    def load_file_paths(
        self, root: Path, split: _Split, scene_id: Optional[int] = None
    ) -> Tuple[List[str], List[str]]:
        image_dir = root / split.dirname / f"{scene_id:06d}" / "rgb"
        target_dir = root / split.dirname / f"{scene_id:06d}" / "mask_visib"

        assert image_dir.exists(), f"Image directory does not exist: {image_dir}"
        assert target_dir.exists(), f"Target directory does not exist: {target_dir}"

        image_paths = sorted(
            [str(p.relative_to(root)) for p in image_dir.glob("*.png")]
        )
        target_paths = sorted(
            [str(p.relative_to(root)) for p in target_dir.glob("*.png")]
        )

        return image_paths, target_paths

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self.image_paths[index]
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        target_relpath = self.target_paths[index]
        target_full_path = os.path.join(self.root, target_relpath)
        with open(target_full_path, mode="rb") as f:
            target_data = f.read()
        return target_data

    def __len__(self) -> int:
        return len(self.image_paths)


def test_dataset_visualization(
    dataset: tudl_bop,
    batch_size=1,
    num_workers=3,
):
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    for image, mask in dataloader:
        if isinstance(image, list):  # List[PIL.Image]
            image = image[0]
            mask = mask[0]
        else:
            image = image[0]
            mask = mask[0]

        # Convert to NumPy for plotting
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image

        mask_np = np.array(mask)

        # Normalize mask to 0/1 for overlay
        if mask_np.max() > 1:
            mask_np = (mask_np > 0).astype(np.uint8)

        # Create blended image
        image_np = np.transpose(image_np, (1, 2, 0))  # (H, W, 3)
        mask_np = mask_np.squeeze(0)                  # (H, W)

        overlay = image_np.copy()
        overlay[mask_np == 1] = (
            overlay[mask_np == 1] * 0.3 + np.array([255, 0, 0]) * 0.7
        ).astype(np.uint8)

        # -----------------------------------------------------
        # Plot results
        # -----------------------------------------------------
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(image_np)
        ax[0].set_title("RGB Image")
        ax[0].axis("off")

        ax[1].imshow(mask_np, cmap="gray")
        ax[1].set_title("Segmentation Mask")
        ax[1].axis("off")

        ax[2].imshow(overlay)
        ax[2].set_title("Overlay (Blended)")
        ax[2].axis("off")

        plt.tight_layout()
        plt.savefig("./output/debug_tudl.jpeg")
        # plt.show()

        break  # Only first batch


if __name__ == "__main__":
    from torchvision.transforms import v2

    print("testing tudl_bop dataset...")
    image_transform = v2.Compose([v2.PILToTensor()]) # Torchvision interface changed now we can call transform(image, mask)

    dataset = tudl_bop(
        split=tudl_bop.Split.TEST,
        root="/home/juan95/JuanData/6d_pose_datasets/tudl/",
        scene_id=1,
        # transform=image_transform,
        # target_transform=image_transform,
        transforms=image_transform
    )

    print(f"Dataset length: {len(dataset)}")
    image, target = dataset[4000]
    print(f"Image size: {image.size()}, Target size: {target.size()}")
    print(target.max(), target.min())
    test_dataset_visualization(dataset)
