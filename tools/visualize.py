import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack import distributed as dist
from torchpack.utils.config import configs
# from torchpack.utils.tqdm import tqdm
import cv2

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def show_seg(labels, car_img):

    PALETTE = [[255, 255, 255], [220, 20, 60], [0, 0, 128], [0, 100, 0],
               [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
               [192, 192, 128], [64, 64, 128], [128, 0, 192], [192, 0, 64]]
    mask_colors = np.array(PALETTE)
    img = np.zeros((650, 400, 3))

    for index, mask_ in enumerate(labels):
        color_mask = mask_colors[index]
        mask_ = mask_.astype(bool)
        img[mask_] = color_mask

    # 这里需要水平翻转，因为这样才可以保证与在图像坐标系下，与习惯相同

    img = np.flip(img, axis=0)
    # 可视化小车
    car_img = np.where(car_img == [0, 0, 0], [255, 255, 255], car_img)[16: 84, 5:, :]
    car_img = cv2.resize(car_img.astype(np.uint8), (30, 16))
    img[img.shape[0] // 2 - 8: img.shape[0] // 2 + 8, img.shape[1] // 2 - 15: img.shape[1] // 2 + 15, :] = car_img

    return img


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="pred", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dist", action="store_true", default=False, help="train distributed")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.25)
    parser.add_argument("--out-dir", type=str, default="viz")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    cfg.dist = args.dist
    if cfg.dist: dist.init()

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    car_img_cv = cv2.imread('assets/car.png')
    
    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)

        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        load_checkpoint(model, args.checkpoint, map_location="cpu")

        if cfg.dist:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
            )
        else:
            model = MMDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
        model.eval()

    for data in dataflow:
        if "metas" in data:
            metas = data["metas"].data[0][0]
            name = "{}-{}".format(metas["timestamp"], metas["token"])
            lidar2image = metas["lidar2image"] if "lidar2image" in metas else None
        else:
            metas = data["img_metas"][0].data[0][0]
            name = metas["token"]
            lidar2image = metas["lidar2img"] if "lidar2img" in metas else None

        if args.mode == "pred":
            # 比 torch.no_grad() 有更好的性能
            with torch.inference_mode():
                outputs = model(return_loss=False, rescale=True,**data)

        bboxes = None
        labels = None
        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        # bevformer det output
        elif args.mode == "pred" and "pts_bbox" in outputs[0] and outputs[0]["pts_bbox"] is not None:
            pass

        if "img" in data and lidar2image is not None:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=lidar2image[k],
                    classes=cfg.object_classes,
                )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        # 语义分割
        if "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}_gt.png"),
                masks,
                classes=cfg.map_classes,
            )

        if "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )

        # bevformer seg output
        if "semantic_indices" in outputs[0] and outputs[0]["seg_preds"] is not None:
            semantic = outputs[0]['seg_preds']
            semantic = onehot_encoding(semantic).cpu().numpy()
            fpath = os.path.join(args.out_dir, "semantic", f"{name}.png")
            mmcv.mkdir_or_exist(os.path.dirname(fpath))
            mmcv.imwrite(show_seg(semantic.squeeze(), car_img_cv), fpath)

            drawgt = True
            if drawgt:
                target_semantic_indices = outputs[0]['semantic_indices'][0].unsqueeze(0)
                one_hot = target_semantic_indices.new_full(semantic.shape, 0)
                one_hot.scatter_(1, target_semantic_indices, 1)
                semantic = one_hot.cpu().numpy().astype(np.float)
                fpath = os.path.join(args.out_dir, "semantic", f"{name}_gt.png")
                mmcv.mkdir_or_exist(os.path.dirname(fpath))
                mmcv.imwrite(show_seg(semantic.squeeze(), car_img_cv), fpath)


if __name__ == "__main__":
    main()
