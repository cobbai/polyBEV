# poly BEV

### 一、环境初始化

### 1.build docker
```bash
cd docker && docker build . -t polybev
```

### 2.docker run
```bash
sudo docker run -it -v `pwd`/../polyBEV:/home/polyBEV -p 10001:22 --gpus all --shm-size 16g bevfusion --name polyBEV /bin/bash
```

### 3.build cuda extention
```bash
python setup.py develop
```

### 二、nuscenes 数据准备
按照[该教程](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md)准备数据集。执行 create_data.py 创建pkl文件后的数据目录结构应该如下图所示：
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --version v1.0-mini --extra-tag nuscenes
```
```
polyBEV
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl

```

### 三、Training
```bash
python tools/train.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

### 四、Evaluation
### 五、Visualize

