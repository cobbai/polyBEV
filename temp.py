import os

def inter():
    _f30 = "data/out_123/images/30_30"
    _f40 = "data/out_123/images/40_40"
    _f45 = "data/out_123/images/40_45"
    _f65 = "data/out_123/label/40_65"

    _f30 = os.listdir(_f30)
    _f40 = os.listdir(_f40)
    _f45 = os.listdir(_f45)
    _f65 = os.listdir(_f65)

    a = set(_f40).intersection(set(_f45))
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

    return 

if __name__ == "__main__":
    inter()