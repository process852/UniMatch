import argparse
from copy import deepcopy
import logging
import os
import pprint

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
from itertools import cycle


parser = argparse.ArgumentParser(description='Semi-Supervised Change Detection')
parser.add_argument('--config', type=str, default = '/data/home/jinjuncan/UniMatch/configs/levir_cd.yaml')
parser.add_argument('--labeled-id-path', type=str, default = "/data/home/jinjuncan/UniMatch/partitions/levircd/1_10/train_l.txt")
parser.add_argument('--unlabeled-id-path', type=str, default = "/data/home/jinjuncan/UniMatch/partitions/levircd/1_10/train_u.txt")
parser.add_argument('--save-path', type=str, default = '/data/home/jinjuncan/dataset/exp_levir')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', id_path="/data/home/jinjuncan/UniMatch/partitions/levircd/1_10/val.txt")

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0

    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss, total_loss_x, total_loss_s, total_loss_w_fp = 0.0, 0.0, 0.0, 0.0
        total_mask_ratio = 0.0

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(cycle(trainloader_l), trainloader_u)

        for i, ((imgA_x, imgB_x, mask_x),
                (imgA_u_w, imgB_u_w, imgA_u_s1, imgB_u_s1, 
                 imgA_u_s2, imgB_u_s2, mask_u)) in enumerate(loader):

            imgA_x, imgB_x, mask_x = imgA_x.cuda(), imgB_x.cuda(), mask_x.cuda()
            imgA_u_w, imgB_u_w = imgA_u_w.cuda(), imgB_u_w.cuda()
            imgA_u_s1, imgB_u_s1 = imgA_u_s1.cuda(), imgB_u_s1.cuda()
            imgA_u_s2, imgB_u_s2 = imgA_u_s2.cuda(), imgB_u_s2.cuda()
            # img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            # cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            # img_u_w_mix = img_u_w_mix.cuda()
            # img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            # ignore_mask_mix = ignore_mask_mix.cuda()

            # with torch.no_grad():
            #     model.eval()

            #     pred_u_w_mix = model(img_u_w_mix).detach()
            #     conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
            #     mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            # img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
            #     img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            # img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
            #     img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()
            # for name, param in model.named_parameters():
            #     print(name)
            #     print(param._version)
            # for name, param in model.named_buffers():
            #     print(name)
            #     print(param._version)

            # num_lb, num_ulb = imgA_x.shape[0], imgA_u_w.shape[0] # batch 数目

            # preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            # preds_l = model(torch.cat((imgA_x, imgB_x, imgA_u_w, imgB_u_w, imgA_u_s1, imgB_u_s1, imgA_u_s2, imgB_u_s2)))
            # preds_w, preds_w_fp = model(torch.cat((imgA_u_w, imgB_u_w)), True)
            # preds_s1 = model(torch.cat((imgA_u_s1, imgB_u_s1)))
            # preds_s2 = model(torch.cat((imgA_u_s2, imgB_u_s2)))

            preds_l, preds_w, preds_w_fp, preds_s1, preds_s2 = model(torch.cat((imgA_x, imgB_x, imgA_u_w, \
                                                                                imgB_u_w, imgA_u_s1, imgB_u_s1, imgA_u_s2, imgB_u_s2)))
            preds_w = preds_w.detach()
            conf_w = preds_w.softmax(dim=1).max(dim=1)[0]
            mask_w = preds_w.argmax(dim=1)

            # mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
            #     mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            # mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
            #     mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            # mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            # conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            # ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            # mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            # conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            # ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(preds_l, mask_x)

            # loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            # loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            # loss_u_s1 = torch.sum(loss_u_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()

            loss_s1 = criterion_u(preds_s1, mask_w)
            loss_s1 = loss_s1*(conf_w >= cfg['conf_thresh'])
            loss_s1 = torch.sum(loss_s1) / torch.sum(conf_w >= cfg['conf_thresh']).item()

            # loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            # loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            # loss_u_s2 = torch.sum(loss_u_s2) / torch.sum(ignore_mask_cutmixed2 != 255).item()

            loss_s2 = criterion_u(preds_s2, mask_w)
            loss_s2 = loss_s2*(conf_w >= cfg['conf_thresh'])
            loss_s2 = torch.sum(loss_s2) / torch.sum(conf_w >= cfg['conf_thresh']).item()

            # loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            # loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            # loss_u_w_fp = torch.sum(loss_u_w_fp) / torch.sum(ignore_mask != 255).item()

            loss_fp = criterion_u(preds_w_fp, mask_w)
            loss_fp = loss_fp*(conf_w >= cfg['conf_thresh'])
            loss_fp = torch.sum(loss_fp) / torch.sum(conf_w >= cfg['conf_thresh']).item()

            # print(loss_x, loss_s1, loss_s2, loss_fp)
            loss = (loss_x + loss_s1 * 0.25 + loss_s2 * 0.25 + loss_fp * 0.5) / 2.0
            # loss_s = torch.nn.MSELoss()(preds_s1, preds_s2)
            # loss = loss_x + loss_s

            torch.distributed.barrier()
            # for name, param in model.named_parameters():
            #     print(name)
            #     print(param._version)
            # for name, param in model.named_buffers():
            #     print(name)
            #     print(param._version)
            # with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_s += (loss_s1.item() + loss_s2.item()) / 2.0
            total_loss_w_fp += loss_fp.item()
            total_mask_ratio += (conf_w >= cfg['conf_thresh']).sum().item() / torch.numel(conf_w)


            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, '
                            'Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask: {:.3f}'.format(
                    i, total_loss / (i+1), total_loss_x / (i+1), total_loss_s / (i+1),
                    total_loss_w_fp / (i+1), total_mask_ratio / (i+1)))

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'
        iou0, iou1, acc, r, p, f1 = evaluate(model, valloader, eval_mode, cfg)
        mIOU = (iou1 + iou0)*50.0
        if rank == 0:
            logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.2f} IOU1: {:.2f}, IOU1: {:.2f}, \
                        Accuracy: {:.2f}, Recall: {:.2f}, Precision: {:.2f}, F1: {:.2f}\n'.format(
                            eval_mode, mIOU, iou1*100.0, iou0*100.0, acc*100.0, r*100.0, p*100.0, f1*100.0
                            ))

        if mIOU > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], mIOU)))


if __name__ == '__main__':
    main()
