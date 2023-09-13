import torch
import random
import copy
from os import path as osp
import tempfile
import numpy as np
from typing import Any, Dict
import mmcv
from mmcv.parallel import DataContainer as DC
from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from mmdet3d.datasets.builder import DATASETS
from torch.utils.data import Dataset
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import get_box_type
from mmdet3d.metrics import IntersectionOverUnion


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


@DATASETS.register_module()
class CustomLocationDataset(Dataset):
    def __init__(self, 
                 dataset_root,
                 ann_file,
                 queue_length=4, 
                 bev_size=(200, 200), 
                 overlap_test=False, 
                 grid_conf=None, 
                 test_mode=False, 
                 pipeline=None, 
                 box_type_3d="LiDAR",
                 num_map_class=4,
                 *args, 
                 **kwargs):
        super().__init__()
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.dataset_root = dataset_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.num_map_class = num_map_class

        # for vector maps
        map_xbound, map_ybound = grid_conf['xbound'], grid_conf['ybound']
        patch_h = map_ybound[1] - map_ybound[0]
        patch_w = map_xbound[1] - map_xbound[0]
        canvas_h = int(patch_h / map_ybound[2])
        canvas_w = int(patch_w / map_xbound[2])
        self.map_patch_size = (patch_h, patch_w)
        self.map_canvas_size = (canvas_h, canvas_w)

        # init seg data
        # self.data = self.init_metas(self.dataset_root)
        self.data_infos = mmcv.load(self.ann_file)
        
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]

        data = dict(
            token=info["scene_token"],
            scene_token=info['scene_token'],
            timestamp=info["scene_token"],
            prev=info["prev"],
            next=info["next"],
            can_bus=info["can_bus"],
            semantic_indices_file=info["semantic_indices_file"],
            img_filename=info["img_filename"]
        )
        
        return data

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def prepare_train_data(self, index): # 这里是否可以进行优化，改进作者提的采样方案，如果index前面的帧不在一个时间邻域内，那么就抛弃该帧，对在邻域内的进行过采样，补齐队列
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """

        queue = [] # 设置时序依赖数据保存在队列里
        index_list = list(range(index-self.queue_length, index)) # 取出后三个
        random.shuffle(index_list) # 打乱
        index_list = sorted(index_list[1:]) #三选2，增加不确定性
        index_list.append(index) # 构成t-2， t-1， t的数据索引，-1为当前帧
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(index)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            queue.append(example)
        return self.union2one(queue)
    
    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.data_infos[index]
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            # first token
            if metas_map[i]['prev'] == None or metas_map[i]['prev'] != prev_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][:3] = np.abs(metas_map[i]['can_bus'][:3])
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
            prev_token = metas_map[i]['scene_token']

        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue = queue[-1]
        return queue

    def evaluate(
        self,
        results,
        metric="bbox",
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        **kwargs,
    ):
        """
            Custom evaluation
        """
        metrics = {}

        if (results[0]['seg_preds'] is not None) and ("semantic_indices" in results[0]):
            metrics.update(self.evaluate_seg(results))

        # TODO: det task
        # if results[0]['pts_bbox'] is not None:
        #     result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        #     if isinstance(result_files, dict):
        #         for name in result_names:
        #             print("Evaluating bboxes of {}".format(name))
        #             ret_dict = self._evaluate_single(result_files[name])
        #         metrics.update(ret_dict)
        #     elif isinstance(result_files, str):
        #         metrics.update(self._evaluate_single(result_files))

        #     if tmp_dir is not None:
        #         tmp_dir.cleanup()

        return metrics
    
    def evaluate_seg(self, results):
        semantic_map_iou_val = IntersectionOverUnion(self.num_map_class)
        semantic_map_iou_val = semantic_map_iou_val.cuda()

        for result in results:
            pred = result['seg_preds']
            pred = onehot_encoding(pred)
            num_cls = pred.shape[1]
            indices = torch.arange(0, num_cls).reshape(-1, 1, 1).to(pred.device)
            pred_semantic_indices = torch.sum(pred * indices, axis=1).int()
            target_semantic_indices = result['semantic_indices'][0].cuda()

            semantic_map_iou_val(pred_semantic_indices,
                                 target_semantic_indices)

        scores = semantic_map_iou_val.compute()
        mIoU = sum(scores[1:]) / (len(scores) - 1)
        seg_dict = dict(
            Validation_num=len(results),
            Divider=round(scores[1:].cpu().numpy()[0], 4),
            Pred_Crossing=round(scores[1:].cpu().numpy()[1], 4),
            Boundary=round(scores[1:].cpu().numpy()[2], 4),
            mIoU=round(mIoU.cpu().numpy().item(), 4)
        )
        return seg_dict

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)



@DATASETS.register_module()
class CLDataset(Dataset):
    def __init__(self, 
                 dataset_root,
                 ann_file,
                 modality=None, 
                 filter_empty_gt=True, 
                 test_mode=False, 
                 pipeline=None, 
                 box_type_3d="LiDAR",
                 map_classes=None,
                 *args, 
                 **kwargs):
        super().__init__()
        self.dataset_root = dataset_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.map_classes = map_classes

        # init seg data
        self.data_infos = mmcv.load(self.ann_file)
        
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set self.flag
        if not self.test_mode:
            self._set_group_flag()

        self.epoch = -1

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
    
    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
    
    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]

        data = dict(
            token=info["scene_token"],
            sample_idx=info['scene_token'],
            timestamp=info["scene_token"],
            prev=info["prev"],
            next=info["next"],
            semantic_indices_file=info["semantic_indices_file"],
        )

        if self.modality["use_camera"]:
            data["img_filename"] = info["img_filename"]
        
        return data


    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self, "pipeline"):
            for transform in self.pipeline.transforms:
                if hasattr(transform, "set_epoch"):
                    transform.set_epoch(epoch)

    def evaluate(
        self,
        results,
        metric="bbox",
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        **kwargs,):

        metrics = {}

        if "masks_bev" in results[0]:
            metrics.update(self.evaluate_map(results))

        if "boxes_3d" in results[0]:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if isinstance(result_files, dict):
                for name in result_names:
                    print("Evaluating bboxes of {}".format(name))
                    ret_dict = self._evaluate_single(result_files[name])
                metrics.update(ret_dict)
            elif isinstance(result_files, str):
                metrics.update(self._evaluate_single(result_files))

            if tmp_dir is not None:
                tmp_dir.cleanup()

        return metrics
    
    def evaluate_map(self, results):
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result["masks_bev"]
            label = result["gt_masks_bev"]

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            # 最后一个维度与 thresholds 广播比较
            # (n_class, H*W) --> (n_class, H*W, 1) --> (n_class, H*W, n_thresholds)
            # (3,260000) --> (3,260000,1) --> (3,260000,7)
            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics


if __name__ == "__main__":
    pipline = [
        {"type": "LoadMultiImageCustom", "to_float32": True},
        {"type": "NormalizeMultiviewImage", "mean": [103.530, 116.280, 123.675], "std": [1.0, 1.0, 1.0], "to_rgb": False},
        {"type": "PadMultiViewImage", "size": [480, 416]},  # 32 的倍数
        {"type": "DefaultFormatBundle_2"}
    ]

    data = CustomLocationDataset("data/customLocation/", 4, (200, 200), False, 
                                 {"xbound": [-20.0, 20.0, 0.1], "ybound": [-35.0, 30.0, 0.1]}, False, pipline)
    
    for i in data:
        print(i)
    