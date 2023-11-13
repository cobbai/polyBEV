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

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.load_from = args.load_from

    model = build_model(cfg.model)
    # model = model.cuda()  # Couldn't export Python operator MultiScaleDeformableAttnFunction_fp32

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

        with torch.no_grad():  # 不计算梯度，不反响传播
            inputs = {}
            inputs["img"] = [data['img'][0].data[0]]  # [ tensor (1, 2, 1, 650, 400) ]
            inputs['img_metas'] = [1]
            inputs['img_metas'][0] = [1]
            inputs['img_metas'][0][0] = {}
            inputs['img_metas'][0][0]['can_bus'] = torch.from_numpy(data['img_metas'][0].data[0][0]['can_bus'])
            inputs['img_metas'][0][0]['scene_token'] = data['img_metas'][0].data[0][0]['scene_token']
            inputs['img_metas'][0][0]['filename'] = data['img_metas'][0].data[0][0]['filename']
            inputs['img_metas'][0][0]['img_norm_cfg'] = data['img_metas'][0].data[0][0]['img_norm_cfg']
            inputs['img_metas'][0][0]['prev'] = data['img_metas'][0].data[0][0]['prev']
            inputs['img_metas'][0][0]['next'] = data['img_metas'][0].data[0][0]['next']

            torch.onnx.export(
                model, 
                inputs,  # inputs输入格式与模型test保持一致
                cfg.load_from.split(".")[0] + '.onnx',
                export_params=True, opset_version=11,
                keep_initializers_as_inputs=True,
                do_constant_folding=False,
                verbose=False,
                input_names=list(inputs.keys()),
                output_names=["output"],
                )

            break

    return


if __name__ == "__main__":
    main()