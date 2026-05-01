from __future__ import annotations

import argparse
from pathlib import Path

from intrep.image_classification import ImageFolderClassificationDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a torchvision ImageFolder dataset.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--show-classes", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    image_size = tuple(args.image_size) if args.image_size is not None else None
    dataset = ImageFolderClassificationDataset(args.root, image_size=image_size)
    first_image, first_label = dataset[0]
    class_preview = ",".join(dataset.label_names[: args.show_classes])
    print("intrep image folder")
    print(f"root={args.root}")
    print(f"classes={len(dataset.label_names)}")
    print(f"images={len(dataset)}")
    print(f"image_shape={dataset.image_shape}")
    print(f"channels={dataset.channel_count}")
    print(f"first_label={int(first_label.item())}")
    print(f"first_image_shape={tuple(first_image.shape)}")
    print(f"first_classes={class_preview}")


if __name__ == "__main__":
    main()
