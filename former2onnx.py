import os
import argparse
import torch

from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.utils import recursive_eval

def parse_args():
    parser = argparse.ArgumentParser(description='bevformer to onnx')
    parser.add_argument('config', help='train config file path')

    parser.add_argument('--load_from',
                    default=None,
                    help='the checkpoint file to load from')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # cfg = Config.fromfile(args.config)
    # cfg.load_from = args.load_from

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    model = build_model(cfg.model)
    model = model.cpu()

    # dummy data
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    for data in data_loader:
        inputs = {}
        inputs["img"] = data['img'][0].data[0]  # tensor (1, 2, 1, 650, 400)
        inputs['img_metas'] = [1]
        inputs['img_metas'][0] = [1]
        inputs['img_metas'][0][0] = {}
        inputs['img_metas'][0][0]['can_bus'] = torch.from_numpy(data['img_metas'][0].data[0][0]['can_bus'])
        inputs['img_metas'][0][0]['scene_token'] = torch.from_numpy(data['img_metas'][0].data[0][0]['scene_token'])
        inputs['img_metas'][0][0]['filename'] = torch.from_numpy(data['img_metas'][0].data[0][0]['filename'])
        inputs['img_metas'][0][0]['img_norm_cfg'] = torch.from_numpy(data['img_metas'][0].data[0][0]['img_norm_cfg'])
        inputs['img_metas'][0][0]['token'] = torch.from_numpy(data['img_metas'][0].data[0][0]['token'])
        inputs['img_metas'][0][0]['prev'] = torch.from_numpy(data['img_metas'][0].data[0][0]['prev'])
        inputs['img_metas'][0][0]['next'] = torch.from_numpy(data['img_metas'][0].data[0][0]['next'])

        torch.onnx.export(model, inputs, cfg.load_from.split("/")[-1].split(".")[0] + '.onnx',
                    export_params=True, opset_version=11,
                    keep_initializers_as_inputs=True,
                    do_constant_folding=False,
                    verbose=False,
                    # input_names = ['input'], output_names=["output"]
                    )

        break

    state_dict = torch.load(args.load_from, map_location='cpu')['net']

    return


if __name__ == "__main__":
    main()