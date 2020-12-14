# -*- coding: utf-8 -*-
from config import D
import tensorflow as tf
import multiprocessing as mt
import os
import glob
import re
import math
import shutil
import numpy as np
import datetime as dt
from model.log_configure import logger
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use('PDF')  # Agg


def read_train_eval_test_filename(testB_percent=0.0,
                                  testE_percent=0.85):
    sample_filepaths = np.array(glob.glob(os.path.join('../input_forecast', D.input_dataset, '*')))
    sample_filepaths = np.sort(sample_filepaths)
    # VT-YYYYMMDDHH_IT-YYYYMMDDHH_FH-FF.npy
    labels_filepaths = []
    prior_dic = {}  # 'YYYYMMDDHH': [50]
    p = 0
    while p < np.shape(sample_filepaths)[0]:
        _, filename = os.path.split(sample_filepaths[p])
        vt = filename.split('_')[0].split('-')[1]  # forecast time
        if filename.split('.')[0].split('_')[-1].split('-')[1] == '24':
            prior_it = filename.split('_')[1].split('-')[1]  # basetime
            prior_arr = np.load(sample_filepaths[p])  # (50, 33, 33) contain nan
            compare_arr = np.loadtxt(os.path.join('../input_real', '190927_r24_usable_0.25_33x33_locMean',
                                                  vt + '.txt'))  # (33, 33) contain nan
            element_number = np.shape(prior_arr)[0]
            inverse_rmse = np.zeros(element_number)  # [50] 1/rmse  larger better
            for i in range(element_number):
                inverse_rmse[i] = 1 / (np.nanmean(np.square(prior_arr[i] - compare_arr)) + 1e-6)
            prior_dic[prior_it] = np.exp(inverse_rmse) / np.nansum(np.exp(inverse_rmse))  # [50] larger better
            sample_filepaths = np.delete(sample_filepaths, p, axis=0)
        else:
            labels_filepaths.append(os.path.join('../input_real', '190927_r24_usable_0.25_33x33_locMean',
                                                 vt + '.txt'))  # YYYYMMDDHH.txt
            p += 1
    labels_filepaths = np.array(labels_filepaths)

    p = 0
    while p < np.shape(sample_filepaths)[0]:
        _, filename = os.path.split(sample_filepaths[p])
        it = filename.split('_')[1].split('-')[1]  # basetime
        if it not in prior_dic.keys():
            sample_filepaths = np.delete(sample_filepaths, p, axis=0)
            labels_filepaths = np.delete(labels_filepaths, p, axis=0)
        else:
            p += 1

    if np.shape(sample_filepaths)[0] != np.shape(labels_filepaths)[0]:
        raise Exception('def read_train_eval_test_filename() sample num != label num')
    else:
        n = np.shape(sample_filepaths)[0]
        test_indeces = np.arange(int(n * testB_percent), int(n * testE_percent))
        train_indeces = np.array([i for i in np.arange(n) if i not in test_indeces])
        eval_indeces = test_indeces.copy()

    logger.info("all sample: %d", n)

    train_sample_filepaths, train_labels_filepaths = np.take(sample_filepaths, train_indeces, axis=0), \
                                                     np.take(labels_filepaths, train_indeces, axis=0)
    eval_sample_filepaths, eval_labels_filepaths = np.take(sample_filepaths, eval_indeces, axis=0), \
                                                   np.take(labels_filepaths, eval_indeces, axis=0)
    test_sample_filepaths, test_labels_filepaths = np.take(sample_filepaths, test_indeces, axis=0), \
                                                   np.take(labels_filepaths, test_indeces, axis=0)

    return train_sample_filepaths, train_labels_filepaths, \
           eval_sample_filepaths, eval_labels_filepaths, \
           test_sample_filepaths, test_labels_filepaths, prior_dic


def read_files(f_datepath_arr, l_datepath_arr, prior_dic, invalid_value=-5):
    features, labels = [], []
    for filepath in f_datepath_arr:
        temp = np.load(filepath)
        _, filename = os.path.split(filepath)
        it = filename.split('_')[1].split('-')[1]  # basetime
        rank = np.argsort(-prior_dic[it])[:D.splited_channel]  # lagest rank
        temp = temp[rank, :, :]  # (sC, 33, 33)
        features.append(temp)
    for filepath in l_datepath_arr:
        labels.append(np.expand_dims(np.loadtxt(filepath), axis=2))
    features, labels = np.array(features).transpose((0, 2, 3, 1)), np.array(labels)
    labels[np.isnan(labels)] = invalid_value
    features[np.isnan(features)] = invalid_value
    return features, labels

def visualize_ensembles(arr, outpath, filenames, dpi=90):
    """
    :param arr: (n, h, w, c)
    """
    assert isinstance(arr, np.ndarray)
    assert len(np.shape(arr))
    cmap = colors.ListedColormap(['white', 'lime', 'cyan', 'royalblue', 'fuchsia', 'red'])
    cmap.set_over('darkred')
    cmap.set_under('gray')

    bounds = [-0.001, 0.0999, 10.099, 25.099, 50.099, 100.099, 250.099]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    channel = np.shape(arr)[-1]
    n = np.shape(filenames)[0]
    nrows = int(np.sqrt(channel * 2))
    ncols = math.ceil(channel / nrows)
    nrows = math.ceil(channel / ncols)
    figsizeW, figsizeH = ncols * 12, nrows * 6
    # logger.debug('channel:%d, nrows:%d, ncols:%d, figsize=%d %d', channel, nrows, ncols, figsizeW, figsizeH)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for c in range(n):
        array = arr[c]
        arr_neg = array.copy()
        arr_neg[np.isnan(arr_neg)] = -1
        fig, _ax = plt.subplots(nrows=nrows, ncols=ncols * 2, figsize=(figsizeW, figsizeH),
                                sharex='col', sharey='row')
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.35, hspace=0.25)
        axs = _ax.flatten()
        for i in range(channel):
            p1 = axs[i * 2].imshow(array[:, :, i], cmap=plt.cm.rainbow)
            axs[i * 2].set_title(str(i) + ' feature map [c]', fontsize=50, pad=12)
            axs[i * 2].tick_params(labelsize=36)

            divider = make_axes_locatable(axs[i * 2])
            cax1 = divider.append_axes("right", size="5%", pad=0.2)
            cbar1 = plt.colorbar(p1, cax=cax1)
            cbar1.ax.tick_params(labelsize=36)

            axs[i * 2 + 1].imshow(arr_neg[:, :, i], cmap=cmap, norm=norm)
            axs[i * 2 + 1].set_title(str(i) + ' feature map [s]', fontsize=50, pad=12)
            axs[i * 2 + 1].tick_params(labelsize=36)

        plt.savefig(os.path.join(outpath, filenames[c] + '.png'), dpi=dpi)
        plt.close()


def visualize_inputs(xarr, yarr, outpath, filenames, dpi=300):
    """
        :param xarr: (B, h, w, sC)
        :param yarr: (B, h, w, outC=1)
    """
    assert isinstance(xarr, np.ndarray) and isinstance(yarr, np.ndarray)
    assert len(np.shape(xarr))

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    channel = np.shape(xarr)[-1]
    n = np.shape(filenames)[0]

    print(channel, n)

    yarr[yarr < 0] = np.nan
    xarr[np.isnan(np.tile(yarr, (1, 1, 1, channel)))] = np.nan

    for c in range(n):
        # array = xarr[c]   # (h, w, sC)

        vmin, vmax = 0.0, 0.0
        for i in range(channel):
            if vmax < np.nanmax(xarr[c]):
                vmax = np.nanmax(xarr[c])

        for i in range(channel):
            fig = plt.figure(figsize=(2, 2), dpi=300)
            ax = plt.subplot(111)
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            p1 = ax.imshow(xarr[c][:, :, i], cmap=plt.cm.rainbow, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

            print(i, c, os.path.join(outpath, filenames[c] + '%d.png' % i))
            plt.savefig(os.path.join(outpath, filenames[c] + '%d.png' % i), dpi=dpi)
            plt.close()

        fig = plt.figure(figsize=(2, 2), dpi=300)
        ax = plt.subplot(111)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        p1 = ax.imshow(yarr[c,:,:,0], cmap=plt.cm.rainbow, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(os.path.join(outpath, filenames[c] + 'label.png'), dpi=dpi)
        plt.close()

        fig = plt.figure(figsize=(1.5, 6), dpi=300)
        cbar_ax = fig.add_axes([0.15, 0.05, 0.30, 0.85])  # colorbar 左 下 宽 高
        v1 = np.linspace(vmin, vmax, 6, endpoint=True)
        cbar1 = plt.colorbar(p1, cax=cbar_ax, ticks=v1)
        cbar1.ax.set_yticklabels(["{:.2f}".format(i) for i in v1])
        cbar1.ax.tick_params(labelsize=16)
        plt.savefig(os.path.join(outpath, filenames[c] + 'ax.png'), dpi=dpi)
        plt.close()



def dic_add(x, y):
    assert isinstance(x, dict)
    assert isinstance(y, dict)
    for key, value in y.items():
        if key in x:
            x[key] += value
        else:
            x[key] = value
    return x


def dic_div_constant(x, contant):
    z = {}
    assert isinstance(x, dict)
    assert contant != 0
    for key, value in x.items():
        z[key] = value / contant
    return z


def dict_divide_dict(dividend, divisor, contant):
    assert isinstance(dividend, dict)
    assert isinstance(divisor, dict)
    for dividend_key in dividend.keys():
        flag = False
        for key in divisor.keys():
            if key in dividend_key:
                flag = True
                dividend[dividend_key] = dividend[dividend_key] / divisor[key]
                break
        if not flag:
            dividend[dividend_key] = dividend[dividend_key] / contant
    return dividend


