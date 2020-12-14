# _*_ coding:utf-8 _*_
import numpy as np
from scipy import stats
from model.log_configure import logger
import math


def compute_verification(predicts, labels):
    """
       !!!invalid value has been -5
        :param labels: tensor shape:[B, H, W, outC]
        :return:  acc_arr: continous_acc
                           multicata_acc
        """
    # logger.debug("%s\n %s\n", str(np.squeeze(predicts)), str(np.squeeze(labels)))

    labels[labels < 0] = np.nan
    predicts[np.isnan(labels)] = np.nan
    try:
        if np.count_nonzero(np.isnan(labels)) != np.count_nonzero(np.isnan(predicts)):
            raise ValueError("labels' nan != predicts' nan")
    except ValueError as e:
        logger.debug("%s\n %s\n %s\n %s", repr(e), str(predicts), str(labels),
                     str(np.argwhere(np.bitwise_xor(np.isnan(labels), np.isnan(predicts)))))

    predicts[predicts < 0] = 0

    continous_acc = {"mean": count_mean(labels, predicts),
                     "bias": count_bias(labels, predicts),
                     "mae": count_MAE(labels, predicts),
                     "rmse": count_RMSE(labels, predicts),
                     "corr": count_corr(labels, predicts),
                     "nse": count_nse(labels, predicts)}

    contingency_table = make_contingency_table(labels, predicts)
    multicata_acc = {}
    multicata_acc.update(count_acc_hss_hk(contingency_table))

    continous_acc.update(multicata_acc)

    return continous_acc


#####################################################################
# continous verification
#####################################################################
# mean  pridict can be 0  Range: (-∞, ∞). Perfect score: 0.
def count_mean(labels, predicts):
    return np.nansum(labels - predicts) / np.maximum(1e-8, np.count_nonzero(~np.isnan(labels)))


# bias  pridict can be 0  Range: (-∞, ∞). Perfect score: 1.
def count_bias(labels, predicts):
    return np.nansum(predicts) / np.maximum(1e-8, np.nansum(labels))


# MAE Range: (0, ∞).  Perfect score: 0
def count_MAE(labels, predicts):
    return np.nansum(np.abs(predicts - labels)) / np.maximum(1e-8, np.count_nonzero(~np.isnan(labels)))


# rmse  Range: (0, ∞).  Perfect score: 0
def count_RMSE(labels, predicts):
    return np.sqrt(np.nansum(np.square(predicts - labels)) / np.maximum(1e-8, np.count_nonzero(~np.isnan(labels))))


# corr  Range: (-1, 1).  Perfect score: 1  sum[(l-l_)*(p-p_)]/sqrt[sum(l-l_)^2]*sqrt[sum(p-p_)^2]
def count_corr(labels, predicts):
    l = labels - np.nanmean(labels)
    p = predicts - np.nanmean(predicts)
    return np.nansum(np.multiply(l, p)) / np.maximum(1e-8,
                                                     np.sqrt(np.nansum(np.square(l))) * np.sqrt(
                                                         np.nansum(np.square(p))))


# nse Nash-Sutcliffe model efficiency coefficient range(−∞, 1)    Perfect score: 1
def count_nse(labels, predicts):
    return 1 - np.nansum(np.square(labels - predicts)) / (np.nansum(np.square((labels - np.nanmean(labels)))) + 1e-12)


# Wilcoxon signed-rank test
def count_wilcoxon_signed_rank_test(labels, predicts):
    mean_p = 0
    for B in range(np.shape(labels)[0]):
        label, predict = labels[B].flatten(), predicts[B].flatten()

        if np.count_nonzero(np.isnan(label)) != np.count_nonzero(np.isnan(predict)):
            logger.debug("Wilcoxon: label nan != predict nan")
            for i in range(np.shape(label)[0]):
                if (np.isnan(label[i]) ^ np.isnan(predict[i])):
                    label[i] = np.nan
                    predict[i] = np.nan

        predict = predict[~np.isnan(predict)]
        label = label[~np.isnan(label)]
        w, p = stats.wilcoxon(label, predict)
        # logger.debug("w %s, p %s", str(w), str(p))
        mean_p += p
    mean_p = mean_p / np.shape(labels)[0]
    return mean_p


# # r2  Range: (0, 1).  Perfect score: 1  statistical model
# def count_R2(labels, predicts):
#     return 1 - np.nansum(np.square(labels - predicts)) / (np.nansum(labels - np.nanmean(labels)) + 1e-12)


#####################################################################
# multi-catalogue verification
#####################################################################
def _graded_output(tensor):
    # sunny 1  light rain 2  moderate rain 3  heavy rain4
    graded = (tensor >= 0).astype(np.float32) + (tensor >= 0.1).astype(np.float32) + \
             (tensor >= 10.1).astype(np.float32) + (tensor >= 25.1).astype(np.float32) + \
             (tensor >= 50.1).astype(np.float32)
    return graded


def make_contingency_table(labels, predicts):
    """
    :param labels:  [B, H, W, outC]
    :param predicts:  [B, H, W, outC]
    :return contingency_table: [5, 5]  predict i, label j  last r/c sum
    """
    labels_graded = _graded_output(labels)
    predicts_graded = _graded_output(predicts)
    contingency_table = np.zeros((6, 6))
    r = 5
    for i in range(r):  # forecast
        for j in range(r):  # obvious
            contingency_table[i, j] = np.count_nonzero(np.multiply(np.where(labels_graded == j + 1, 1, 0),
                                                                   np.where(predicts_graded == i + 1, 1, 0)))
        contingency_table[i, -1] = np.nansum(contingency_table[i, :-1])
    for i in range(r + 1):  # obvious
        contingency_table[-1, i] = np.nansum(contingency_table[:-1, i])
    return contingency_table


# acc=1/N * ∑1->4 (i,i)  Range: [0, 1].  Perfect score: 1
# Heidke skill score    Range:(-∞, 1), 0 indicates no skill.  Perfect score: 1
# Hanssen and Kuipers discriminant (true skill statistic, Peirce’s skill score)
#  Range: [-1, 1], 0 indicates no skill. Perfect score: 1
def count_acc_hss_hk(contingency_table):
    c = 0
    for i in range(5):
        c += contingency_table[i, i]
    acc = c / contingency_table[-1, -1]
    tmean = np.nansum(np.multiply(contingency_table[:-1, -1],
                                  contingency_table[-1, :-1])) / (contingency_table[-1, -1] ** 2)  # ∑ n(F)*n(O)
    hss = (acc - tmean) / (1 - tmean + 1e-10)
    hk = (acc - tmean) / (
            1 - np.nansum(contingency_table[-1, :-1] ** 2) / (contingency_table[-1, -1] ** 2) + 1e-10)

    k = np.shape(contingency_table)[0] - 1
    s = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            a_1_sum = 0
            for r in range(j - 1):
                p_sum = 0
                for p in range(r + 1):
                    p_sum += contingency_table[-1, p] / contingency_table[-1, -1]
                a_r = (1 - p_sum) / max(p_sum, 1e-8)
                a_1_sum += 1 / a_r
            a_sum = 0
            for r in range(j, k - 1, 1):
                p_sum = 0
                for p in range(r + 1):
                    p_sum += contingency_table[-1, p] / contingency_table[-1, -1]
                a_r = (1 - p_sum) / max(p_sum, 1e-8)
                a_sum += a_r

            if j == i:
                s[i][j] = (1 / (k - 1)) * (a_1_sum + a_sum)
            elif j > i:
                s[i][j] = (1 / (k - 1)) * (a_1_sum - (j - i) + a_sum)

    for i in range(k):
        for j in range(k):
            if j < i:
                s[i][j] = s[j][i]
    gs = 0
    for i in range(k):
        for j in range(k):
            gs += contingency_table[i][j] * s[i][j]
    gs = gs / max(contingency_table[-1][-1], 1e-8)

    return {"acc": acc, "hss": hss, "hk": hk, "gs": gs}



if __name__ == '__main__':
    from model.utils import *
    import numpy as np
    import time
    import os
    from config import D
    from scipy.interpolate import griddata


    def mean_method(features, labels):  # [sampleN, h, w, c]  [sampleN, h, w, 1]
        predicts = np.mean(features, axis=-1)  # (sampleN, h, w)
        labels = np.squeeze(labels)  # (sampleN, h, w)
        return predicts, labels


    def median_quantile(features, labels):
        predicts = np.nanmedian(features, axis=-1)  # (sampleN, h, w)
        labels = np.squeeze(labels)  # (sampleN, h, w)
        return predicts, labels


    def best_percentile(features, labels):
        B = np.shape(features)[0]
        member = np.shape(features)[-1]

        def fuse(tensor):  # [inC]
            tensor = -np.sort(-tensor)
            n = np.shape(tensor)[0]
            if tensor[0] >= 100:
                return tensor[0]
            elif tensor[int(0.05 * n)] >= 50:
                return tensor[int(0.05 * n)]
            elif tensor[int(0.35 * n)] >= 25:
                return tensor[int(0.35 * n)]
            elif tensor[int(0.75 * n)] >= 10:
                return tensor[int(0.75 * n)]
            else:
                return tensor[int(0.9 * n)]

        result_arr = []
        for b in range(B):
            mean_sample = np.mean(features[b], axis=-1)  # [H, W]
            flat_mean = mean_sample.flatten()  # [HxW]
            position = stats.rankdata(-flat_mean, method='min') - 1
            fuse_result = np.zeros_like(mean_sample)
            for i in range(D.input_h):
                for j in range(D.input_w):
                    fuse_result[i, j] = fuse(features[b, i, j].flatten())
            # check_fuse = fuse_result.copy()
            fuse_result = -np.sort(-fuse_result, axis=None)
            result = fuse_result[position]
            result = np.reshape(result, (D.input_h, D.input_w))
            result_arr.append(result)

        return np.array(result_arr), np.squeeze(labels)  # , check_fuse


    def probability_matching(features, labels):  # [sampleN, h, w, c]  [sampleN, h, w, 1]
        member = np.shape(features)[-1]
        B = np.shape(features)[0]
        result_arr = []
        for b in range(B):
            mean_sample = np.mean(features[b], axis=-1)  # [H, W]
            flat_mean = mean_sample.flatten()  # [HxW]
            position = stats.rankdata(-flat_mean, method='min') - 1  # decending
            flat_ensemble = -np.sort(-features[b], axis=None)  # [HxWxc]
            fuse_result = np.zeros_like(flat_mean)  # [HxW]
            for i in range(D.input_h * D.input_w):
                fuse_result[i] = np.median(flat_ensemble[i * member:(i + 1) * member])  # [c] -> 1
            result = fuse_result[position]
            result = np.reshape(result, (D.input_h, D.input_w))
            result_arr.append(result)
        return np.array(result_arr), np.squeeze(labels)  # [sampleN, h, w]  [sampleN, h, w]


    def weighted_bias_removed_mean(tfeatures, tlabels, efeatures):
        member = np.shape(tfeatures)[-1]
        eB = np.shape(efeatures)[0]
        tmean_error = np.nanmean(tfeatures - np.tile(tlabels, (1, 1, 1, member)), axis=0)  # (h, w, member)
        Ei = 1 / tmean_error  # (h, w, member)
        alphai = Ei / np.nansum(Ei, axis=-1, keepdims=True)  # (h, w, member)
        Obar = np.nanmean(tlabels, axis=0)  # (h, w, 1)
        Fibar = np.nanmean(tfeatures, axis=0)  # (h, w, member)
        F = np.tile(np.expand_dims(Obar, axis=0), (eB, 1, 1, 1)) + np.nansum(
            np.multiply(np.tile(np.expand_dims(alphai, axis=0), (eB, 1, 1, 1)),
                        (efeatures - np.tile(np.expand_dims(Fibar, axis=0), (eB, 1, 1, 1)))),
            axis=-1, keepdims=True)  # (sampleN, h, w, 1)
        return F
