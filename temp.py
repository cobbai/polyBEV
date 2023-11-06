import os
import mmcv
import numpy as np
import copy
import re
import warnings
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


def parse_header(lines):
    """ Parse header of PCD files.
    """
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            warnings.warn("warning: can't understand line: %s" % ln)
            continue
        key, value = match.group(1).lower(), match.group(2)

        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = [int(v) for v in value.split()]# map(int, value.split())
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = [float(v) for v in value.split()]# map(float, value.split())
        elif key == 'data':
            metadata[key] = value.strip().lower()
        # TODO apparently count is not required?
    # add some reasonable defaults
    if 'count' not in metadata:
        metadata['count'] = [1]*len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def _build_dtype(metadata):
    """ Build numpy structured array dtype from pcl metadata.

    Note that fields with count > 1 are 'flattened' by creating multiple
    single-count fields.

    *TODO* allow 'proper' multi-count fields.
    """
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
            typenames.extend([np_type]*c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def parse_binary_pc_data(f, dtype, metadata):
    rowstep = metadata['points']*dtype.itemsize  # 数据一共多少字节。（dtype.itemsize = metadata中size总和）
    # for some reason pcl adds empty space at the end of files
    buf = f.read(rowstep)
    return np.fromstring(buf, dtype=dtype)


def point_cloud_from_fileobj(f):
    """ Parse pointcloud coming from file object f
    """
    header = []
    while True:
        ln = f.readline().strip()
        ln = ln.decode(encoding='utf-8')
        header.append(ln)

        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    if metadata['data'] == 'ascii':
        print("TODO ...")
        # pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        print("TODO ...")
        # pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        print('DATA field is neither "ascii" or "binary" or\
                "binary_compressed"')
    return pc_data


def point_cloud_from_path(fname):
    """ load point cloud in binary format
    """
    with open(fname, 'rb') as f:
        pc = point_cloud_from_fileobj(f)
    return pc

k = (3 + 5) / (80 + 80)
def select_points(total_points):
    points = []
    label = 0
    for point in total_points:
        # source == 16 车道线
        if point[5] == 16:
            # 16 路沿， 21 栅栏
            if point[6] == 16 or point[6] == 21:
                label = 2
            else:
                label = 1
        elif point[5] == 19:  # 箭头
            label = 3

        if label == 0: continue
        if point[0] < -35 or point[0] > 30: continue
        if point[1] < -20 or point[1] > 20: continue
        # if point[2] < -5 or point[2] > 3: continue

        points.append([point[0], point[1], -5 + k * (point[2] + 80), label])
    return np.array(points)


def inter():
    _f30 = "data/out_123/images/30_30"
    _f40 = "data/out_123/images/40_40"
    _f45 = "data/out_123/images/40_45"
    _f465 = "data/out_123/images/40_65"
    _f65 = "data/out_123/label/40_65"
    _f80 = "data/out_123/lidar/80"

    _f30 = os.listdir(_f30)
    _f40 = os.listdir(_f40)
    _f45 = os.listdir(_f45)
    _f465 = os.listdir(_f465)
    _f65 = os.listdir(_f65)
    _f80 = [x.replace("pcd.bin", "png") for x in os.listdir(_f80)]

    # a = set(_f40).intersection(set(_f45))
    a = set(_f465).intersection(set(_f80))
    b = a.intersection(set(_f30))
    c = b.intersection(set(_f65))
    print(len(c))

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

    # result = result[800:1000]
    # result = result[850:891] + result[1300:1361] + result[2260:2301] + result[3790:3831] + \
    #          result[6060:6101] + result[6570:6631] + result[7520:7561]

    e_num = [[850, 891], [1300, 1361], [2260, 2301], [3790, 3831], [6060, 6101], [6570, 6631], [7520, 7561]]
    e_num = {"token_" + str(k):v for k, v in enumerate(e_num)}
    ans = []
    for token, idx in e_num.items():
        cut = copy.deepcopy(result[idx[0]: idx[1]])
        for meta in cut:
            meta["token"] = meta["scene_token"]
            # meta["scene_token"] = token
            meta["points"] = select_points(point_cloud_from_path(os.path.join(dataset_root, "lidar", "80", str(meta["token"]) + "00000.pcd.bin")))  # 2753582.500000.pcd.bin
            ans.append(meta)
    result = ans

    datalen = len(result)

    mmcv.dump(result[:-int(datalen * 0.15)], "data/out_123/metas_train.pkl")
    mmcv.dump(result[-int(datalen * 0.15):], "data/out_123/metas_val.pkl")

    return result


def main():
    inter()  # 数据交集
    init_metas("data/out_123", "data/out_123/metas_intersection.txt")

if __name__ == "__main__":
    # inter()
    main()
    a = mmcv.load("data/out_123/metas_train.pkl")

    print(a)