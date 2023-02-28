import os
import numpy as np


def make_LEVIR_txt(root, ratio):
    """
    Args:
        root: 数据集目录
        ratio: 有监督数据所占的比例
    """
    txt_dir = os.path.join("/data/home/jinjuncan/UniMatch/partitions/levircd", "1_{}".format(int(ratio*100)))
    if os.path.exists(txt_dir):
        return 
    os.makedirs(txt_dir)
    
    labels = os.listdir(os.path.join(root, "train/label/"))
    labels = list(filter(lambda x: x.endswith('.png'), labels))
    labels = list(map(lambda x: os.path.join(root, "train/label/", x), labels))
    np.random.seed(42)
    np.random.shuffle(labels)
    train_sup = labels[:int(len(labels) * ratio)]
    train_unsup = labels[int(len(labels) * ratio):]
    write_txt(os.path.join(txt_dir, "train_l.txt"), train_sup)
    write_txt(os.path.join(txt_dir, "train_u.txt"), train_unsup)

    labels = os.listdir(os.path.join(root, "val/label/"))
    labels = list(filter(lambda x: x.endswith('.png'), labels))
    labels = list(map(lambda x: os.path.join(root, "val/label/", x), labels))
    write_txt(os.path.join(txt_dir, "val.txt"), labels)

    labels = os.listdir(os.path.join(root, "test/label/"))
    labels = list(filter(lambda x: x.endswith('.png'), labels))
    labels = list(map(lambda x: os.path.join(root, "test/label/", x), labels))
    write_txt(os.path.join(txt_dir, "test.txt"), labels)

def write_txt(txt_name, list_string):
    with open(txt_name, "w") as f:
        for item in list_string:
            f.write(item.replace("label", "A"))
            f.write(" ")
            f.write(item.replace("label", "B"))
            f.write(" ")
            f.write(item)
            f.write("\n")

if __name__ == "__main__":
    make_LEVIR_txt("/data/home/jinjuncan/dataset/LEVIR_CD", 0.1)