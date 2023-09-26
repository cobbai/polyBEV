import os
import mmcv
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

def inter():
    _f30 = "data/out_123/images/30_30"
    _f40 = "data/out_123/images/40_40"
    _f45 = "data/out_123/images/40_45"
    _f465 = "data/out_123/images/40_65"
    _f65 = "data/out_123/label/40_65"

    _f30 = os.listdir(_f30)
    _f40 = os.listdir(_f40)
    _f45 = os.listdir(_f45)
    _f465 = os.listdir(_f465)
    _f65 = os.listdir(_f65)

    a = set(_f40).intersection(set(_f45))
    # a = set(_f465)
    b = a.intersection(set(_f30))
    c = b.intersection(set(_f65))
    print(c)

    with open("data/out_123/metas.txt", "r", encoding="utf-8") as r:
        line = r.readline()
        temp = []
        while line:
            seg = [float(x.strip("").strip(" ").strip("\n")) for x in line.split(",")]
            time = str(seg[0])
            if time + "00000.png" not in c:
                line = r.readline()
                continue
            temp.append(line)
            line = r.readline()
    
    with open("data/out_123/metas_intersection.txt", "w", encoding="utf-8") as w:
        for i in temp:
            w.write(i)

    # with open("data/out_123/val_intersection.txt", "w", encoding="utf-8") as w:
    #     for i in temp[-int(len(temp) * 0.2):]:
    #         w.write(i)

    return

def walk(dirname, l:list):
    for name in os.listdir(dirname):
        path = os.path.join(dirname, name)
 
        if os.path.isfile(path):
            if path.endswith("png"): l.append(path)
        else:
            walk(path, l)

def check():
    l = []
    walk("data/out_123/", l)
    for i in l:
        img = mmcv.imread(i, "unchanged")
        if img.max() == 0:
            print(i)
        
        # if img.min() != 0:
        #     print(i)

    return

def init_metas(dataset_root, metas_path):
    result = []

    if not os.path.exists(metas_path):
        raise FileNotFoundError("need metas.txt file")
    
    # load metas.txt
    with open(metas_path, "r", encoding="utf-8") as r:
        line = r.readline()
        while line:
            temp = {}
            seg = [float(x.strip("").strip(" ").strip("\n")) for x in line.split(",")]
            time = seg[0]  # time name
            temp["scene_token"] = time

            # can_bus
            can_bus = seg[1:]
            rotation = Quaternion([can_bus[6]] + can_bus[3:6])
            can_bus[3:7] = rotation
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle
            temp["can_bus"] = np.array(can_bus)

            temp["img_filename"] = [
                os.path.join(dataset_root, "images", "30_30", str(time) + "00000.png"),
                # os.path.join(dataset_root, "images", "40_40", str(time) + "00000.png"),
                # os.path.join(dataset_root, "images", "40_45", str(time) + "00000.png"),
                os.path.join(dataset_root, "images", "40_65", str(time) + "00000.png"),
                ]
            temp["semantic_indices_file"] = os.path.join(dataset_root, "label", "40_65", str(time) + "00000.png")
            result.append(temp)
            line = r.readline()

    # sort by timestamp
    result = list(sorted(result, key=lambda x: x["scene_token"]))
    
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

    result = result[800:1000]
    datalen = len(result)

    mmcv.dump(result, "data/out_123/metas_train.pkl")
    mmcv.dump(result[-int(datalen * 0.2):], "data/out_123/metas_val.pkl")

    return result


def main():
    inter()
    init_metas("data/out_123", "data/out_123/metas_intersection.txt")

if __name__ == "__main__":
    # inter()
    main()
    a = mmcv.load("data/out_123/metas_train.pkl")

    print(a)