import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
from itertools import cycle
import yaml

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from supervised import evaluate

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log



parser = argparse.ArgumentParser(description='Semi-Supervised Change Detection')
parser.add_argument('--config', type=str, default = '/data/home/jinjuncan/UniMatch/configs/levir_cd.yaml')
parser.add_argument('--labeled-id-path', type=str, default = "/data/home/jinjuncan/UniMatch/partitions/levircd/1_10/train_l.txt")
parser.add_argument('--unlabeled-id-path', type=str, default = "/data/home/jinjuncan/UniMatch/partitions/levircd/1_10/train_u.txt")
parser.add_argument('--save-path', type=str, default = '/data/home/jinjuncan/dataset/exp_levir')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def eval_metric(predict, target):
    target = target.cpu()
    TP = torch.bitwise_and((predict == 1), (target == 1)).long().sum()
    TN = torch.bitwise_and((predict == 0), (target == 0)).long().sum()
    FP = torch.bitwise_and((predict == 1), (target == 0)).long().sum()
    FN = torch.bitwise_and((predict == 0), (target == 1)).long().sum()
    assert TP + TN + FP + FN == torch.numel(target), "count error"
    change_IOU = TP / (TP + FP + FN) if TP + FP + FN != 0 else torch.tensor(0)
    unchange_IOU = TN / (TN + FN + FP) if TN + FN + FP != 0 else torch.tensor(0)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP != 0 else torch.tensor(0)
    recall = TP / (TP + FN) if TP + FN != 0 else torch.tensor(0)
    f1score = 2 * precision * recall / (
            precision + recall + torch.tensor(1e-8)) if precision + recall != 0 else torch.tensor(0)
    result = {
        'Accuracy': accuracy.item(),
        'Precision': precision.item(),
        'Recall': recall.item(),
        'F1_score': f1score.item(),
        "change_IOU": change_IOU.item(),
        "unchange_IOU": unchange_IOU.item()
    }
    return result

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    # output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def evaluate_testset():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    testset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', 
                          id_path="/data/home/jinjuncan/UniMatch/partitions/levircd/1_10/test.txt")
    cudnn.enabled = True
    cudnn.benchmark = True
    
    testloader = DataLoader(dataset = testset, batch_size=1,shuffle=False,drop_last=False)
    
    model = DeepLabV3Plus(cfg)
    
    load_state_dict = torch.load("/data/home/jinjuncan/dataset/exp_levir/exp1/resnet50_90.07.pth")
    model.load_state_dict(load_state_dict)
    model = model.cuda()
    
    Accuracy, Precision, Recall, F1_score, change_IOU, unchange_IOU = [],[],[],[],[],[]
    model.eval()
    with torch.no_grad():
        for i, (imgA, imgB, mask, id) in enumerate(testloader):
            # print("start")
            imgA, imgB = imgA.cuda(), imgB.cuda()
            pred = model(torch.cat((imgA, imgB)), mode = "eval").argmax(dim=1)
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            if union[1] == 0:
                IOU1 = 1.0
                IOU0 = intersection[0] / (union[0] + 1e-10)
            else:
                IOU0, IOU1 = intersection / (union + 1e-10)
            change_IOU.append(IOU1)
            unchange_IOU.append(IOU0)
            print("unchange_IOU:{:.2f}, change_IOU:{:.2f}".format(unchange_IOU[-1], change_IOU[-1]))
            # result = eval_metric(pred.cpu(), mask)
            # Accuracy.append(result['Accuracy'])
            # Precision.append(result['Precision'])
            # Recall.append(result['Recall'])
            # F1_score.append(result['F1_score'])
            # change_IOU.append(result['change_IOU'])
            # unchange_IOU.append(result['unchange_IOU'])
            # print("F1_score:{:.2f}, change_IOU:{:.2f}".format(F1_score[-1], change_IOU[-1]))
    # Accuracy: {:.2f},
    # Precision: {:.2f},
    # Recall: {:.2f},
    # F1_score: {:.2f},
    print("""
    change_IOU: {:.2f},
    unchange_IOU: {:.2f},
    """.format(
        # sum(Accuracy) / len(Accuracy),
        # sum(Precision) / len(Precision),
        # sum(Recall) / len(Recall),
        # sum(F1_score) / len(F1_score),
        sum(change_IOU) / len(change_IOU),
        sum(unchange_IOU) / len(unchange_IOU),
    ))
    
if __name__ == "__main__":
    evaluate_testset()