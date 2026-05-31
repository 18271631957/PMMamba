import numpy as np
import os
import logging
import glob
import cv2
from skimage.morphology import skeletonize

def cal_global_acc(pred, gt):
    h,w = gt.shape
    return [np.sum(pred==gt), float(h*w)]

def get_statistics_seg(pred, gt, num_cls=2):
    h,w = gt.shape
    statistics = []
    for i in range(num_cls):
        tp = np.sum((pred==i)&(gt==i))
        fp = np.sum((pred==i)&(gt!=i))
        fn = np.sum((pred!=i)&(gt==i))
        statistics.append([tp, fp, fn])
    return statistics

def get_statistics_prf(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def segment_metrics(pred_list, gt_list, num_cls = 2):
    global_accuracy_cur = []
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (pred / 255).astype('uint8')
        global_accuracy_cur.append(cal_global_acc(pred_img, gt_img))
        statistics.append(get_statistics_seg(pred_img, gt_img, num_cls))


    global_acc = np.sum([v[0] for v in global_accuracy_cur]) / np.sum([v[1] for v in global_accuracy_cur])
    counts = []
    for i in range(num_cls):
        tp = np.sum([v[i][0] for v in statistics])
        fp = np.sum([v[i][1] for v in statistics])
        fn = np.sum([v[i][2] for v in statistics])

        counts.append([tp, fp, fn])

    mean_acc = np.sum([v[0] / (v[0] + v[2]) for v in counts]) / num_cls
    mean_iou_acc = np.sum([v[0] / (np.sum(v)) for v in counts]) / num_cls

    return global_acc, mean_acc, mean_iou_acc

def prf_metrics(pred_list, gt_list):
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (((pred / np.max(pred))>0.5)).astype('uint8')
        statistics.append(get_statistics_prf(pred_img, gt_img))

    tp = np.sum([v[0] for v in statistics])
    fp = np.sum([v[1] for v in statistics])
    fn = np.sum([v[2] for v in statistics])
    print("tp:{}, fp:{}, fn:{}".format(tp,fp,fn))
    p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    r_acc = tp / (tp + fn)
    f_acc = 2 * p_acc * r_acc / (p_acc + r_acc)
    return p_acc,r_acc,f_acc


def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            statistics.append(get_statistics(pred_img, gt_img))
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])

        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        r_acc = tp / (tp + fn)
        final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])

    return final_accuracy_all

def thred_half(src_img_list, tgt_img_list):
    Precision, Recall, F_score = prf_metrics(src_img_list, tgt_img_list)
    Global_Accuracy, Class_Average_Accuracy, Mean_IOU = segment_metrics(src_img_list, tgt_img_list)
    print("Global Accuracy:{}, Class Average Accuracy:{}, Mean IOU:{}, Precision:{}, Recall:{}, F score:{}".format(
        Global_Accuracy, Class_Average_Accuracy, Mean_IOU, Precision, Recall, F_score))

def get_statistics(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01):
    final_F1_list = []
    for pred, gt in zip(pred_list, gt_list):
        p_acc_list = []
        r_acc_list = []
        F1_list = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc=0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc==0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)

            p_acc_list.append(p_acc)
            r_acc_list.append(r_acc)
            F1_list.append(F1)

        assert len(p_acc_list)==100, "p_acc_list is not 100"
        assert len(r_acc_list)==100, "r_acc_list is not 100"
        assert len(F1_list)==100, "F1_list is not 100"

        max_F1 = np.max(np.array(F1_list))
        final_F1_list.append(max_F1)

    final_F1 = np.sum(np.array(final_F1_list))/len(final_F1_list)
    return final_F1

def cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01):
    save_data = {
        "ODS": [],
    }
    final_ODS = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        ODS_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            # calculate precision
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc=0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc==0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            ODS_list.append(F1)

        ave_F1 = np.mean(np.array(ODS_list))
        final_ODS.append(ave_F1)
    ODS = np.max(np.array(final_ODS))
    return ODS

def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01, pred_imgs_names=None, gt_imgs_names=None):
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            if (FN + FP + TP) <= 0:
                iou = 0
            else:
                iou_1 = TP / (FN + FP + TP)
                iou_0 = TN / (FN + FP + TN)
                iou = (iou_1 + iou_0)/2
            iou_list.append(iou)
        ave_iou = np.mean(np.array(iou_list))
        final_iou.append(ave_iou)
    mIoU = np.max(np.array(final_iou))
    return mIoU


def get_cldice(pred_img, gt_img):
    """计算单张图像的 clDice"""
    # 确保输入是纯粹的 boolean 类型，骨架提取需要这个
    pred_b = pred_img > 0
    gt_b = gt_img > 0

    # 1. 强行剥洋葱，提取 1 像素宽的中心线骨架
    pred_skel = skeletonize(pred_b)
    gt_skel = skeletonize(gt_b)

    # 2. 计算拓扑精确率 (T_prec): 预测的骨架有多少落在了真实的 mask 内？
    t_prec_num = np.sum(pred_skel & gt_b)
    t_prec_den = np.sum(pred_skel)
    t_prec = t_prec_num / t_prec_den if t_prec_den > 0 else 0.0

    # 3. 计算拓扑召回率 (T_sens): 真实的骨架有多少落在了预测的 mask 内？(抓断裂的利器！)
    t_sens_num = np.sum(gt_skel & pred_b)
    t_sens_den = np.sum(gt_skel)
    t_sens = t_sens_num / t_sens_den if t_sens_den > 0 else 0.0

    # 4. 计算调和平均数得到 clDice
    if t_prec + t_sens == 0:
        return 0.0
    return 2.0 * t_prec * t_sens / (t_prec + t_sens)


def cal_clDice_metrics(pred_list, gt_list, thresh=0.5):
    """
    计算整个测试集的平均 clDice。
    ⚠️ 避坑指南：提取骨架非常吃 CPU 算力！
    所以我们不像 ODS 那样循环 100 个阈值（那会慢到死机），
    而是直接选用最标准的 0.5 作为二值化阈值。
    """
    cldice_list = []
    for pred, gt in zip(pred_list, gt_list):
        # 还原到 0 和 1
        gt_img = (gt / 255).astype('uint8')
        pred_img = (pred / 255 > thresh).astype('uint8')

        cldice_val = get_cldice(pred_img, gt_img)
        cldice_list.append(cldice_val)

    return np.mean(cldice_list)


def eval(log_eval, src_img_list, tgt_img_list, epoch):
    """
        现在的 eval 不再读取磁盘，而是直接接收内存中的图像列表。
        src_img_list: 包含所有预测图 numpy 数组的列表
        tgt_img_list: 包含所有标签图 numpy 数组的列表
    """
    assert len(src_img_list) == len(tgt_img_list)

    # 1. 计算 PR 指标 (注意：这里内部调用的 cal_prf_metrics 稍后也需要微调)
    final_accuracy_all = cal_prf_metrics(src_img_list, tgt_img_list)
    final_accuracy_all = np.array(final_accuracy_all)

    # 获取不同阈值下的 P, R, F
    # F_list[0] 通常对应阈值为 0 的情况，或者你可以根据需要选择
    Precision_list, Recall_list, F_list = final_accuracy_all[:, 1], final_accuracy_all[:, 2], final_accuracy_all[:, 3]

    # 2. 计算 mIoU, ODS, OIS (这些函数本身就是处理 list 的，基本不用改)
    mIoU = cal_mIoU_metrics(src_img_list, tgt_img_list)
    ODS = cal_ODS_metrics(src_img_list, tgt_img_list)
    OIS = cal_OIS_metrics(src_img_list, tgt_img_list)

    # ======== 【新增的这一行】 ========
    # clDice_score = cal_clDice_metrics(src_img_list, tgt_img_list, thresh=0.5)

    return {
        'epoch': epoch,
        'mIoU': mIoU,
        'ODS': ODS,
        'OIS': OIS,
        'F1': F_list[0],
        'Precision': Precision_list[0],
        'Recall': Recall_list[0],
        # 'clDice': clDice_score,
    }



    

