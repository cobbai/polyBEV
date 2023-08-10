import torch
import random
import copy
from os import path as osp
import tempfile
import numpy as np
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
        self.data = self.init_metas(self.dataset_root)
        
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()


    def init_metas(self, dataset_root):
        result = []
        metas_path = self.ann_file

        if not osp.exists(metas_path):
            raise FileNotFoundError("need metas.txt file")
        
        # load metas.txt
        with open(metas_path, "r", encoding="utf-8") as r:
            line = r.readline()
            while line:
                temp = {}
                seg = [float(x.strip("").strip("\n")) for x in line.split(",")]
                time = int(seg[0])  # time name
                temp["scene_token"] = time

                # can_bus
                can_bus = np.array(seg[1:])
                rotation = Quaternion(can_bus[3:7])
                can_bus[3:7] = rotation
                patch_angle = quaternion_yaw(rotation) / np.pi * 180
                if patch_angle < 0:
                    patch_angle += 360
                can_bus[-2] = patch_angle / 180 * np.pi
                can_bus[-1] = patch_angle
                temp["can_bus"] = can_bus

                temp["img_filename"] = [
                    osp.join(dataset_root, "images", "30_30", str(time) + ".jpg"),
                    osp.join(dataset_root, "images", "40_40", str(time) + ".jpg"),
                    osp.join(dataset_root, "images", "40_45", str(time) + ".jpg"),
                    ]
                temp["semantic_indices"] = osp.join(dataset_root, "label", "40_65", str(time) + ".jpg")
                result.append(temp)
                line = r.readline()

        # sort by timestamp
        result = sorted(result, key=lambda x: x["scene_token"])

        # pre next timestamp
        for i in range(len(result)):
            if i == 0:
                result[i]["prev"] = None
                result[i]["next"] = result[i+1]["scene_token"]
            elif i == len(result) - 1:
                result[i]["prev"] = result[i-1]["scene_token"]
                result[i]["next"] = None
            else:
                result[i]["prev"] = result[i-1]["scene_token"]
                result[i]["next"] = result[i+1]["scene_token"]

        return result

    def __len__(self):
        return len(self.data)

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
            input_dict = self.data[i]
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            queue.append(copy.deepcopy(example)) # 深拷贝防止第一帧图片改到自己
        return self.union2one(queue)
    
    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.data[index]
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
        result = {}
        imgs_list = []
        metas_map = {}
        prev_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            imgs_list.append(each["img"].data)

            # first token
            if metas_map[i]['scene_token'] != prev_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        result['img_metas'] = DC(metas_map, cpu_only=True)
        result["img"] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        result["semantic_indices"] = torch.tensor(queue[-1]["semantic_indices"], dtype=torch.int64)  # int64 long
        return result

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
    