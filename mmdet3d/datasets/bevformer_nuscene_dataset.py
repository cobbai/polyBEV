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

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from ..utils import VectorizedLocalMap
from ..utils import NuScenesEval_custom
from mmdet3d.metrics import IntersectionOverUnion


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, grid_conf=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

        # for vector maps
        self.map_dataroot = self.dataset_root

        map_xbound, map_ybound = grid_conf['xbound'], grid_conf['ybound']
        patch_h = map_ybound[1] - map_ybound[0]
        patch_w = map_xbound[1] - map_xbound[0]
        canvas_h = int(patch_h / map_ybound[2])
        canvas_w = int(patch_w / map_xbound[2])
        self.map_patch_size = (patch_h, patch_w)
        self.map_canvas_size = (canvas_h, canvas_w)

        # add seg label
        self.nusc = NuScenes(version=self.version, dataroot=self.dataset_root, verbose=False)
        self.vector_map = VectorizedLocalMap(
            dataroot=self.map_dataroot,
            patch_size=self.map_patch_size,
            canvas_size=self.map_canvas_size,
        )

    # sample_augmentation 和 img_transform 用于得到LSS算法所需参数
    def sample_augmentation(self, scale=0.8):
        fH, fW = (int(900 * scale + 900 * scale % 32), int(1600 * scale))
        resize = (fW / 1600, fH / 900)
        resize_dims = (fW, fH)
        return resize, resize_dims

    @staticmethod
    def img_transform(resize):
        post_rot2 = torch.eye(2)
        post_tran2 = torch.zeros(2)

        rot_resize = torch.Tensor([[resize[0], 0],
                                   [0, resize[1]]])
        post_rot2 = rot_resize @ post_rot2
        post_tran2 = rot_resize @ post_tran2

        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2
        return post_rot, post_tran

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]#[:10]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        # whether test-set
        self.test_submission = 'test' in ann_file
        return data_infos

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
        index_list.append(index) # 构成t-2， t-1， t的数据索引
        # index_list = [202, 203, 204, 205]
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example) # 将以及处理好的数据送入队列，随后将送入网络
        return self.union2one(queue)

    def get_map_ann_info(self, info):

        # get the annotations of HD maps
        vectors = self.vector_map.gen_vectorized_samples(
            info['location'], info['ego2global_translation'], info['ego2global_rotation'])

        # type0 = 'divider'
        # type1 = 'pedestrain'
        # type2 = 'boundary'

        for vector in vectors:
            pts = vector['pts']
            vector['pts'] = np.concatenate(
                (pts, np.zeros((pts.shape[0], 1))), axis=1)

        return vectors

    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
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
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue # 数据格式化

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        # add scene loc
        info['location'] = self.nusc.get('log', self.nusc.get('scene', info['scene_token'])['log_token'])['location']


        #  获取LSS分割标注
        anno = self.nusc.get('sample', info['token'])['anns']
        info['inst'] = []
        for tok in anno:
            inst = self.nusc.get('sample_annotation', tok)
            info['inst'].append(inst)

        info['ego_pose'] = self.nusc.get('ego_pose',
                                         self.nusc.get('sample_data',
                                                        self.nusc.get('sample', info['token'])['data']['LIDAR_TOP'])['ego_pose_token'])


        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            location=info['location'],  # BEVFusion 分割标注生成
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            inst=info['inst'],   # LSS分割标注生成
            egopose=info['ego_pose']
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            # lss need
            trans = []
            rots = []
            post_trans = []
            post_rots = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['camera_intrinsics']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

                # lss need
                resize, resize_dims = self.sample_augmentation()
                post_rot, post_tran = self.img_transform(resize)
                post_trans.append(post_tran)
                post_rots.append(post_rot)
                samp = self.nusc.get('sample_data', cam_info['sample_data_token'])
                sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
                trans.append(torch.Tensor(sens['translation']))
                rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))


            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    trans=trans,
                    rots=rots,
                    post_trans=post_trans,
                    post_rots=post_rots,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        input_dict['vectors'] = self.get_map_ann_info(info)
        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

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


    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                if name == 'seg_preds' or name == 'semantic_indices':
                    pass
                    #print(f'\nCalculate seg iou from {name}')
                    # 怎么计算iou？如何获得gt label？
                else:
                    print(f'\nFormating bboxes of {name}')
                    results_ = [out[name] for out in results]
                    tmp_file_ = osp.join(jsonfile_prefix, name)
                    result_files.update(
                        {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

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

        if results[0]['pts_bbox'] is not None:
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

    def evaluate_seg(self, results):
        num_map_class = 4
        semantic_map_iou_val = IntersectionOverUnion(num_map_class)
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

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.dataset_root,
                             verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        print(f'result_path:{result_path}, eval_detection_configs:{self.eval_detection_configs}, '
              f'eval_set_map:{eval_set_map[self.version]}')
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail