# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib
import seaborn as snb
from os.path import join
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

from config import D
from model.utils import read_train_eval_test_filename
from bayesiantests import correlated_ttest, correlated_ttest_MC
from verification import make_contingency_table

matplotlib.use('TkAgg')  # PDF Agg
root = os.path.dirname(os.path.abspath(__file__))
date_dir = join(root, 'datasets')


def _visual_matplot(arr, ste):
    cmap = colors.ListedColormap(['white', 'lime', 'cyan', 'royalblue', 'fuchsia', 'red'])
    cmap.set_over('darkred')
    cmap.set_under('gray')
    bounds = [-0.001, 0.0999, 10.099, 25.099, 50.099, 100.099, 250.099]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    surface = np.loadtxt(os.path.join(D.data_dir, 'surfacemask.txt'))
    n = np.shape(arr)[0]
    arr[np.tile(np.expand_dims(surface, axis=0), (n, 1, 1)) == 0] = np.nan

    fig, _ax = plt.subplots(nrows=1, ncols=n, figsize=(11, 1.8),
                            sharex='col', sharey='row')
    fig.subplots_adjust(left=0.05, bottom=0, right=0.9, top=0.95, wspace=0, hspace=0)
    axs = _ax.flatten()
    for i in range(n):
        im = axs[i].imshow(arr[i], cmap=cmap, norm=norm)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.8])  # colorbar 左 下 宽 高
    cbar1 = plt.colorbar(im, cax=cbar_ax, extend='both')
    cbar1.ax.set_yticklabels(['0', '0.1', '10', '25', '50', '100', '250'], fontsize=16)
    cbar1.ax.tick_params(labelsize=20)
    # plt.show()
    plt.savefig(ste + ".png")


def _count_frequency(pre_listdir):
    gt_files = os.listdir(pre_listdir)
    sum = np.zeros((33, 33))
    for file in gt_files:
        gt_arr = np.loadtxt(os.path.join(pre_listdir, file))
        sum += np.where(gt_arr >= 0.1, np.ones_like(gt_arr), np.zeros_like(gt_arr))
    print(sum)
    probabbility = sum / np.nanmax(sum)
    return probabbility


def _subfigure_plot_of_figure_1(array, figdir, vmax):  # paint Smod sample figure
    fig, _ax = plt.subplots(nrows=5, ncols=10, figsize=(9, 3.7), sharex='col', sharey='row')
    fig.subplots_adjust(left=0.00, bottom=0.00, right=0.83, top=1.0, wspace=0.0, hspace=0.0)
    axs = _ax.flatten()
    for i in range(5 * 10):
        p = axs[i].imshow(array[:, :, i], cmap=plt.cm.rainbow, vmin=0.0, vmax=vmax)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    cbar_ax = fig.add_axes([0.89, 0.07, 0.018, 0.6])  # colorbar left bottom wid high
    v1 = np.linspace(0.0, vmax, 6)
    cbar1 = plt.colorbar(p, cax=cbar_ax, ticks=v1)
    cbar1.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
    cbar1.ax.tick_params(labelsize=16)
    # plt.show()
    plt.savefig(figdir, dpi=300)
    plt.close()


def _fullfigure_plot_of_figure_1(array, figdir, vmax):  # paint Smod sample figure
    fig, _ax = plt.subplots(nrows=1, ncols=1, figsize=(1, 1), sharex='col', sharey='row')
    fig.subplots_adjust(left=0.00, bottom=0.00, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    _ax.imshow(array[:, :, 0], cmap=plt.cm.rainbow, vmin=0.0, vmax=vmax)
    _ax.set_xticks([])
    _ax.set_yticks([])

    plt.savefig(figdir, dpi=300)
    plt.close()


def text_figure_1():
    smod_fea = np.load(join(date_dir, 'test_features_ec_pf_tp_AT24_33x33_025-7.npy'))  # (33, 33, inC)
    smod_lab = np.load(join(date_dir, 'test_labels_ec_pf_tp_AT24_33x33_025-7.npy'))  # (33, 33, outC)
    for i in range(np.shape(smod_lab)[0]):
        if np.max(smod_lab[i]) - 86 > 0:
            print(i)
            vmax = max(np.nanmax(smod_lab[i]), np.nanmax(smod_fea[i]))
            smod_lab[i][smod_lab[i] < 0] = np.nan
            smod_fea[i][np.isnan(np.tile(smod_lab[i], (1, 1, 50)))] = np.nan
            _subfigure_plot_of_figure_1(smod_fea[i], join(date_dir, 'p1f.png'), vmax)
            _fullfigure_plot_of_figure_1(smod_lab[i], join(date_dir, 'p1l.png'), vmax)
            break

def _precipitation_events_in_datasets_statistic():
    for round in range(10):
        train_sample_filepaths, train_labels_filepaths, \
        eval_sample_filepaths, eval_labels_filepaths, \
        test_sample_filepaths, test_labels_filepaths, prior_dic = read_train_eval_test_filename(round * 0.1,
                                                                                                round * 0.1 + 0.1)
        all_day_dict = 0
        all_grid_dict = 0

        rain_day_dict = {'nozero': 0, 'small': 0, 'mid': 0, 'large': 0, 'storm': 0, 'violence': 0}
        rain_grid_dict = {'nozero': 0, 'small': 0, 'mid': 0, 'large': 0, 'storm': 0, 'violence': 0}

        for filepath in test_labels_filepaths:
            label = np.loadtxt(filepath)

            all_day_dict += 1
            all_grid_dict += np.count_nonzero(~np.isnan(label))

            labels_classed = [np.where(label >= 0.1, label, np.zeros_like(label)),
                              np.where(label >= 10.1, label, np.zeros_like(label)),
                              np.where(label >= 25.1, label, np.zeros_like(label)),
                              np.where(label >= 50.1, label, np.zeros_like(label)),
                              np.where(label >= 100.1, label, np.zeros_like(label))]

            rain_grid_dict['nozero'] += np.count_nonzero(label)
            rain_grid_dict['small'] += np.count_nonzero(labels_classed[0])
            rain_grid_dict['mid'] += np.count_nonzero(labels_classed[1])
            rain_grid_dict['large'] += np.count_nonzero(labels_classed[2])
            rain_grid_dict['storm'] += np.count_nonzero(labels_classed[3])
            rain_grid_dict['hugestorm'] += np.count_nonzero(labels_classed[4])

            rain_day_dict['nozero'] += np.where(np.count_nonzero(label) > 0, 1, 0)
            rain_day_dict['small'] += np.where(np.count_nonzero(labels_classed[0]) > 0, 1, 0)
            rain_day_dict['mid'] += np.where(np.count_nonzero(labels_classed[1]) > 0, 1, 0)
            rain_day_dict['large'] += np.where(np.count_nonzero(labels_classed[2]) > 0, 1, 0)
            rain_day_dict['storm'] += np.where(np.count_nonzero(labels_classed[3]) > 0, 1, 0)
            rain_day_dict['hugestorm'] += np.where(np.count_nonzero(labels_classed[4]) > 0, 1, 0)

        with open(os.path.join(D.data_dir, 'precipitation_time_distribution.txt'), 'a') as f:
            f.write('#####\n')
            f.write('round ' + str(round))
            f.write('\n#####\n\n')

            f.write('all_day_dict ' + str(all_day_dict) + '\n')
            f.write('rain_day_dict: nozero:%d, small:%d, mid:%d, large:%d, storm:%d, violence:%d\n'
                    % (rain_day_dict['nozero'], rain_day_dict['small'], rain_day_dict['mid'],
                       rain_day_dict['large'], rain_day_dict['storm'], rain_day_dict['violence']))
            f.write('all_grid_dict ' + str(all_grid_dict) + '\n')
            f.write('rain_grid_dict: nozero:%d, small:%d, mid:%d, large:%d, storm:%d, violence:%d\n\n\n'
                    % (rain_grid_dict['nozero'], rain_grid_dict['small'], rain_grid_dict['mid'],
                       rain_grid_dict['large'], rain_grid_dict['storm'], rain_grid_dict['violence']))

def text_figure_4():
    def draw_histogram(interval, y):
        fig = plt.figure(figsize=(5.5, 4.9), dpi=300)
        ax = plt.subplot(111)
        fig.subplots_adjust(left=0.16, bottom=0.16, right=0.97, top=0.95, wspace=0, hspace=0)

        y = y
        ymax = 0.8
        ax.bar(range(len(y)), y, width=1, edgecolor='black')
        ax.set_xlabel("precipitation", fontsize=20)
        ax.set_ylabel("Proportion", fontsize=20)
        ax.set_ylim(ymax=ymax, ymin=0.0)
        index = range(len(interval))
        index = [float(c) - 0.5 for c in index]
        ax.set_xticks(index)
        ax.set_xticklabels(interval, fontsize=16)
        ax.set_yticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'], fontsize=16)
        # plt.show()
        plt.savefig("./01.png")

    count_delete_sun = [1197824, 255082, 98192, 24937, 3221]
    y = np.array(count_delete_sun) / sum(count_delete_sun)
    print(y)

    interval = [0.1, 10, 25, 50, 100, 500]
    draw_histogram(interval, y)


def text_figure_5_draw_ploygon():
    def draw_screen_poly(lats, lons, m):
        x, y = m(lons, lats)
        xy = zip(x, y)
        poly = Polygon(list(xy), facecolor=None, edgecolor='red', alpha=1, linewidth=3, fill=False)
        plt.gca().add_patch(poly)

    fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    fig.subplots_adjust(left=0.095, bottom=0.09, right=0.98, top=0.98)
    plt.subplot(111)

    lats = [21, 29, 29, 21]
    lons = [109.5, 109.5, 117.5, 117.5]

    m = Basemap(projection='lcc', resolution='i', lat_0=24, lon_0=113, lat_1=19., lat_2=31,
                width=3.2E6, height=2.4E6)  # projection='sinu' 'merc' 'cyl'

    m.bluemarble(scale=1)

    # draw parallels.
    parallels = np.arange(15.0, 35.1, 5)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=20, linewidth=1.5, rotation=-35)
    # Draw the latitude labels on the map

    # draw meridians
    meridians = np.arange(95, 130.1, 5)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=20, linewidth=1.5, rotation=0)

    draw_screen_poly(lats, lons, m)

    # plt.show()
    plt.savefig("picture.png", dpi=300)


def text_figure_5_draw_matrix_2():
    fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
    fig.subplots_adjust(left=0.05, bottom=0.09, right=0.98, top=0.98)
    plt.subplot(111)

    m = Basemap(projection='merc', resolution='h', llcrnrlon=108.5, llcrnrlat=20, urcrnrlon=118.5, urcrnrlat=30)

    m.bluemarble(scale=2)

    # draw parallels.
    parallels = np.arange(21, 29.1, 2.5)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=20, linewidth=1.5, rotation=-35)
    # Draw the latitude labels on the map

    # draw meridians
    meridians = np.arange(109, 118.1, 2.5)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=20, linewidth=1.5, rotation=0)

    precipitation = np.loadtxt(os.path.join(D.data_dir, '2013051614.txt'))
    precipitation[np.isnan(precipitation)] = -5
    lats = np.arange(20.875, 29.2, 0.25)
    lons = np.arange(109.375, 117.7, 0.25)
    xintrp, yintrp = np.meshgrid(lons, lats)

    cmap = colors.ListedColormap(['white', 'lime', 'cyan', 'royalblue', 'fuchsia', 'red'])
    cmap = cmap(np.arange(cmap.N))  # Get the colormap colors
    cmap[:, -1] = np.tile([0.7], 6)
    cmap = colors.ListedColormap(cmap)  # Create new colormap
    cmap.set_over('darkred', alpha=0.7)
    cmap.set_under('gray', alpha=0.7)
    bounds = [-0.001, 0.1, 10.1, 25.1, 50.1, 100.1, 250.1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    colormesh = m.pcolormesh(xintrp, yintrp, np.flipud(precipitation), latlon=True, cmap=cmap, norm=norm)
    # 'RdBu_r', norm=norm
    cbar = m.colorbar(colormesh, location='right', pad="6%", extend='both')  # plot the colorbar on the map
    cbar.ax.tick_params(labelsize=15)

    # plt.show()
    plt.savefig("pictureb.png", dpi=300)


def text_figure_6_draw_plot():
    x = [1, 2, 3, 4, 5]
    group_labels = ['', '$L=1$', '$L=2$', '$L=3$', '$L=4$', '$L=5$']
    # plt.plot(x, single[0], 'r', label='broadcast')
    # plt.plot(x, single[1], 'b', label='join')
    # plt.xticks(x, group_labels, rotation=0)

    single = [
        [0.3912, 0.3890, 0.3910, 0.3914, 0.3894],
        [0.6129, 0.6015, 0.6023, 0.6030, 0.5960]
    ]
    fig, ax1 = plt.subplots(figsize=(7.6, 5.1))
    # fig.subplots_adjust(left=0.22, bottom=0.20, right=0.8, top=0.85)   # has title
    fig.subplots_adjust(left=0.22, bottom=0.05, right=0.8, top=0.85)
    ax2 = ax1.twinx()
    # plt.title('(a) Single-model', y=-0.25, fontsize=25)  # has title

    plt.grid(axis='y', color='grey', linestyle='--', lw=1.5, alpha=0.5)

    plot1 = ax1.plot(x, single[0], 'r.--', label='$HSS$', markersize=13)
    ax1.set_ylabel('$HSS$', fontsize=20, labelpad=10)
    ax1.tick_params(labelsize=20, which='both', length=0)
    ax1.set_ylim(0.3730, 0.3950)  # single
    # for tl in ax1.get_yticklabels():
    #     tl.set_color('r')

    plot2 = ax2.plot(x, single[1], 'b.-', label='$Acc$', markersize=13)
    ax2.set_ylabel('$Acc$', fontsize=20, labelpad=10)
    ax2.set_ylim(0.5920, 0.6300)  # single
    ax2.tick_params(axis='y', labelsize=20, which='both', length=0)
    # for tl in ax2.get_yticklabels():
    #     tl.set_color('g')
    ax2.set_xlim(0.5, 5.5)
    ax2.set_xticklabels(group_labels)
    ax2.tick_params(labelsize=20)

    lines = plot1 + plot2
    ax1.legend(lines, [l.get_label() for l in lines], loc='center', fontsize=15,
               bbox_to_anchor=(0.22, 1.1, 0.55, 0.1), mode="expand", ncol=2)
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 5))
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 5))
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # plt.setp(ax.get_xticklabels(), visible=False)
        # plt.setp(ax.get_yticklabels(), visible=False)
    # plt.show()
    plt.savefig('single-layer' + ".png", dpi=300)
    plt.close()

    multi = [
        [0.2942, 0.2671, 0.3211, 0.3681, 0.3151],
        [0.5451, 0.5273, 0.5545, 0.5784, 0.5514]
    ]

    fig, ax1 = plt.subplots(figsize=(7.6, 5.1))
    # fig.subplots_adjust(left=0.22, bottom=0.20, right=0.8, top=0.85)   # has title
    fig.subplots_adjust(left=0.22, bottom=0.05, right=0.8, top=0.85)
    ax2 = ax1.twinx()
    # plt.title('(b) Multi-model', y=-0.25, fontsize=25)  # has title

    plt.grid(axis='y', color='grey', linestyle='--', lw=1.5, alpha=0.5)

    plot1 = ax1.plot(x, multi[0], 'r.--', label='$HSS$', markersize=13)
    ax1.set_ylabel('$HSS$', fontsize=20, labelpad=10)
    ax1.tick_params(labelsize=20, which='both', length=0)
    ax1.set_ylim(0.2000, 0.3750)  # multi

    plot2 = ax2.plot(x, multi[1], 'b.-', label='$Acc$', markersize=13)
    ax2.set_ylabel('$Acc$', fontsize=20, labelpad=10)
    ax2.set_ylim(0.5200, 0.6350)  # multi
    ax2.tick_params(axis='y', labelsize=20, which='both', length=0)
    ax2.set_xlim(0.5, 5.5)
    ax2.set_xticklabels(group_labels)
    ax2.tick_params(labelsize=20)

    lines = plot1 + plot2
    ax1.legend(lines, [l.get_label() for l in lines], loc='center', fontsize=15,
               bbox_to_anchor=(0.22, 1.1, 0.55, 0.1), mode="expand", ncol=2)
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 5))
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 5))
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    # plt.show()
    plt.savefig('multi-layer' + ".png", dpi=300)


def _make_Bayesian_Correlated_ttest(data1, data2, runs, name1, name2, xlabel, category, rope=0.05):
    """
    Perform Bayesian Correlated t-test, save probabilities to file
    and save plot.
    Args:
        data1 (numpy.ndarray): Results of performing m runs of k-fold CV using first classifier
        data2 (numpy.ndarray): Results of performing m runs of k-fold CV using second classifier
        runs (int): m
        name1 (str): Name of first classifier
        name2 (str): Name of second classifier
        xlabel (str): x-axis label for the plot
        category (str): Category being predicted
        rope (float): Region of practical equivalence
    """

    # Get names and make data matrix.
    names = (name1, name2)
    data_mat = np.empty((len(data1), 2), dtype=float)
    data_mat[:, 0] = data1
    data_mat[:, 1] = data2

    # Generate samples from posterior (it is not necesssary because the posterior is a Student).
    samples = correlated_ttest_MC(data_mat, rope=rope, runs=runs, nsamples=50000)
    l, w, r = correlated_ttest(data_mat, rope=rope, runs=runs, verbose=True, names=names)

    # Plot posterior.
    snb.kdeplot(samples, shade=True, cut=1)

    # Plot rope region.
    plt.axvline(x=-rope, color='orange')
    plt.axvline(x=rope, color='orange')

    # Set x-axis label.
    plt.xlabel(xlabel)
    plt.savefig('./datasets/bctt_' + category.replace('-', '_') + '_' + \
                name1.replace(' ', '_').lower() + '_' + name2.replace(' ', '_').lower() + '.png')

    # Save probabilities to file.
    with open(os.path.join(D.data_dir, 'bctt_15_results.txt'), 'a') as f:
        f.write('#####\n')
        f.write(category)
        f.write('\n#####\n\n')
        f.write('p({0} > {1})={2}, '.format(name1, name2, l))
        f.write('p({0})={1}, '.format('EQ', w))
        f.write('p({0} > {1})={2}\n\n'.format(name2, name1, r))


def table_4_significance_test():
    # acc hss rmse
    muti_em = np.loadtxt(os.path.join(D.data_dir, 'multi_em.txt'))
    muti_bp = np.loadtxt(os.path.join(D.data_dir, 'multi_bp.txt'))
    single_em = np.loadtxt(os.path.join(D.data_dir, 'single_em.txt'))
    single_bp = np.loadtxt(os.path.join(D.data_dir, 'single_bp.txt'))
    multi_mlnl = np.loadtxt(os.path.join(D.data_dir, 'multi_mlnl.txt'))
    single_mlnl = np.loadtxt(os.path.join(D.data_dir, 'single_mlnl.txt'))

    metric_list = ['rmse', 'acc', 'hss']
    alternative_list = ["less", "greater", "greater"]
    rope = 0.01

    for i in range(2):
        a, b = multi_mlnl[:5, i], muti_em[:5, i]  # np.concatenate((, multi_mlnl[:, i]), axis=0)
        print(a)
        print(b)
        _make_Bayesian_Correlated_ttest(a, b, 1, 'mlnl', 'em', xlabel='multi',
                                        category='multi' + '(' + metric_list[i] + ')',
                                        rope=rope)
        statistic, pvalue = stats.wilcoxon(a, b,
                                           correction=False, alternative=alternative_list[i], mode='auto')
        print("multi_mlnl", alternative_list[i], 'em', metric_list[i], "statistic =", statistic, 'p =', pvalue)
    print('\n')

    for i in range(2):
        a, b = multi_mlnl[:5, i], muti_bp[:5, i]
        print(a)
        print(b)
        _make_Bayesian_Correlated_ttest(a, b, 1, 'mlnl', 'bp', xlabel='multi',
                                        category='multi' + '(' + metric_list[i] + ')',
                                        rope=rope)
        statistic, pvalue = stats.wilcoxon(a, b,
                                           correction=True, alternative=alternative_list[i], mode='auto')
        print("multi_mlnl", alternative_list[i], 'bp', metric_list[i], "statistic =", statistic, 'p =', pvalue)
    print('\n')

    for i in range(2):
        a, b = single_mlnl[:5, i], single_em[:5, i]
        print(a)
        print(b)
        _make_Bayesian_Correlated_ttest(a, b, 1, 'mlnl', 'em', xlabel='single',
                                        category='single' + '(' + metric_list[i] + ')',
                                        rope=rope)
        statistic, pvalue = stats.wilcoxon(a, b,
                                           correction=True, alternative=alternative_list[i], mode='exact')
        print("single_mlnl", alternative_list[i], 'em', metric_list[i], "statistic =", statistic, 'p =%.10f' % pvalue)
    print('\n')

    for i in range(2):
        a, b = single_mlnl[:5, i], single_bp[:5, i]
        print(a)
        print(b)
        _make_Bayesian_Correlated_ttest(a, b, 1, 'mlnl', 'bp', xlabel='single',
                                        category='single' + '(' + metric_list[i] + ')',
                                        rope=rope)
        statistic, pvalue = stats.wilcoxon(a, b,
                                           correction=True, alternative=alternative_list[i], mode='exact')
        print("single_mlnl", alternative_list[i], 'bp', metric_list[i], "statistic =", statistic, 'p = %.10f' % pvalue)


def text_figure_7_draw_plot():
    labels = [np.load(os.path.join(D.data_dir, 'eval_labels_ec_pf_tp_AT24_33x33_025-7.npy')),
              np.load(os.path.join(D.data_dir, 'eval_labels_multiorigin_cf_tp_AT24_33x33_025-7.npy'))]
    b_forecast = [np.load(os.path.join(D.data_dir, 'avg_valid7.npy')), np.load(os.path.join(D.data_dir, 'PM_valid7.npy')),
                  np.load(os.path.join(D.data_dir, 'BP_valid7.npy')), np.load(os.path.join(D.data_dir, 'WBRE_valid7.npy')),
                  np.load(os.path.join(D.data_dir, 'avg_valid7m.npy')), np.load(os.path.join(D.data_dir, 'PM_valid7m.npy')),
                  np.load(os.path.join(D.data_dir, 'BP_valid7m.npy')), np.load(os.path.join(D.data_dir, 'WBRE_valid7m.npy'))]

    mlnl_forc = [np.load(os.path.join(D.data_dir, 'valid_predict79.npy')),
                 np.load(os.path.join(D.data_dir, 'valid_predict.npy')),
                 np.load(os.path.join(D.data_dir, 'valid_predict49.npy')),
                 np.load(os.path.join(D.data_dir, 'valid_predictm.npy'))]
    mlnl_label = [np.load(os.path.join(D.data_dir, 'valid_label79.npy')),
                  np.load(os.path.join(D.data_dir, 'valid_label.npy')),
                  np.load(os.path.join(D.data_dir, 'valid_label49.npy')),
                  np.load(os.path.join(D.data_dir, 'valid_labelm.npy'))]

    def graded_output(tensor):
        graded = (tensor >= 0).astype(np.float32) + (tensor >= 0.1).astype(np.float32) + \
                 (tensor >= 10.1).astype(np.float32) + (tensor >= 25.1).astype(np.float32) + \
                 (tensor >= 50.1).astype(np.float32)
        graded[np.isnan(tensor)] = np.nan
        return graded

    def count_pre_frequency(tensor, threshold):
        fre = np.where(tensor >= threshold, np.ones_like(tensor), np.zeros_like(tensor))
        # fre = np.nansum(fre, axis=0) / np.shape(fre)[0]
        fre[np.isnan(tensor)] = np.nan
        return fre

    def count_ets(l, o):
        h = np.nansum(np.where((l == 1.0) * (o == 1.0), np.ones_like(l), np.zeros_like(l)), axis=0)
        z = np.nansum(np.where((l == 0.0) * (o == 0.0), np.ones_like(l), np.zeros_like(l)), axis=0)
        f = np.nansum(np.where((l == 0.0) * (o == 1.0), np.ones_like(l), np.zeros_like(l)), axis=0)
        m = np.nansum(np.where((l == 1.0) * (o == 0.0), np.ones_like(l), np.zeros_like(l)), axis=0)

        h_random = (h + m) * (h + f) / (h + m + f + z)
        dive_num = h + f + m - h_random
        ets = (h - h_random) / np.where(dive_num == 0.0, np.ones_like(dive_num) * 1e-8, dive_num)
        ets[np.isnan(l[0])] = np.nan
        return ets

    def count_rmse(l, o):
        rmse = np.sqrt(np.nanmean(np.square(l - o), axis=0))
        rmse[np.isnan(l[0])] = np.nan
        return rmse

    labels[0][labels[0] < 0] = np.nan
    labels[1][labels[1] < 0] = np.nan
    # hits = []
    etss = []
    storme_ets = []
    rmses = []

    label_ets = [count_pre_frequency(labels[0], 0.1), count_pre_frequency(labels[1], 0.1)]
    label_storme_ets = [count_pre_frequency(labels[0], 25.1), count_pre_frequency(labels[1], 25.1)]
    label_grade = [graded_output(labels[0]), graded_output(labels[1])]

    for i in range(len(b_forecast) // 2):
        b_forecast[i][labels[0] < 0] = np.nan
        etss.append(np.squeeze(count_ets(label_ets[0], count_pre_frequency(b_forecast[i], 0.1))))
        storme_ets.append(np.squeeze(count_ets(label_storme_ets[0], count_pre_frequency(b_forecast[i], 25.1))))
        rmses.append(np.squeeze(count_rmse(label_ets[0], b_forecast[i])))

    for i in range(len(mlnl_forc) // 2):
        mlnl_forc[i][mlnl_label[i] < 0] = np.nan
        mlnl_label[i][mlnl_label[i] < 0] = np.nan
        etss.append(np.squeeze(count_ets(count_pre_frequency(mlnl_label[i], 0.1),
                                         count_pre_frequency(mlnl_forc[i], 0.1))))
        storme_ets.append(np.squeeze(count_ets(count_pre_frequency(mlnl_label[i], 25.1),
                                               count_pre_frequency(mlnl_forc[i], 25.1))))
        rmses.append(np.squeeze(count_rmse(mlnl_label[i], mlnl_forc[i])))

    for i in range(len(b_forecast) // 2, np.shape(b_forecast)[0]):
        b_forecast[i][labels[1] < 0] = np.nan
        etss.append(np.squeeze(count_ets(label_ets[1], count_pre_frequency(b_forecast[i], 0.1))))
        storme_ets.append(np.squeeze(count_ets(label_storme_ets[1], count_pre_frequency(b_forecast[i], 25.1))))
        rmses.append(np.squeeze(count_rmse(label_ets[1], b_forecast[i])))

    for i in range(len(mlnl_forc) // 2, np.shape(mlnl_forc)[0]):
        mlnl_forc[i][mlnl_label[i] < 0] = np.nan
        mlnl_label[i][mlnl_label[i] < 0] = np.nan
        etss.append(np.squeeze(count_ets(count_pre_frequency(mlnl_label[i], 0.1),
                                         count_pre_frequency(mlnl_forc[i], 0.1))))
        storme_ets.append(np.squeeze(count_ets(count_pre_frequency(mlnl_label[i], 25.1),
                                               count_pre_frequency(mlnl_forc[i], 25.1))))
        rmses.append(np.squeeze(count_rmse(mlnl_label[i], mlnl_forc[i])))

    print(np.nanmax(np.array(etss)), np.nanmin(np.array(etss)),
          np.nanmax(np.array(storme_ets)), np.nanmin(np.array(storme_ets)),
          np.nanmax(np.array(rmses)), np.nanmin(np.array(rmses)))
    text = ['ground truth', 'EM', 'PM', 'BP', 'WEM', 'IC-MLNet', 'IC-MLNet-Stest(Mtest)']

    # ---------- rmse plot------------
    ncolum = np.shape(rmses)[0] // 2
    fig, _ax = plt.subplots(nrows=2, ncols=ncolum, figsize=(11.12, 3.75),
                            sharex='col', sharey='row')
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.89, top=0.88, wspace=0, hspace=0)
    axs = _ax.flatten()
    vmin, vmax = 100, 0
    for i in range(2 * ncolum):
        if vmin > np.nanmin(rmses[i]):
            vmin = np.nanmin(rmses[i])
        if vmax < np.nanmax(rmses[i]):
            vmax = np.nanmax(rmses[i])

    vmax = 60.0
    for i in range(2 * ncolum):
        im = axs[i].imshow(rmses[i], cmap=plt.cm.gist_rainbow, vmin=vmin, vmax=vmax)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        if i < ncolum:
            axs[i].set_title(text[i + 1], fontsize=14)

    cbar_ax = fig.add_axes([0.915, 0.1, 0.018, 0.8])  # colorbar 左 下 宽 高
    v1 = np.linspace(vmin, vmax, 6, endpoint=True)
    cbar1 = plt.colorbar(im, cax=cbar_ax, ticks=v1, extend='max')
    cbar1.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
    cbar1.ax.tick_params(labelsize=16)
    # plt.show()
    plt.savefig("rmse.png", dpi=300)
    plt.close()

    # ---------- occurence plot------------
    ncolum = np.shape(etss)[0] // 2
    fig, _ax = plt.subplots(nrows=2, ncols=ncolum, figsize=(11.05, 3.75),
                            sharex='col', sharey='row')
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.9, top=0.88, wspace=0, hspace=0)
    axs = _ax.flatten()
    vmin, vmax = 1, 0
    for i in range(2 * ncolum):
        if vmin > np.nanmin(etss[i]):
            vmin = np.nanmin(etss[i])
        if vmax < np.nanmax(etss[i]):
            vmax = np.nanmax(etss[i])

    vmin = 0.0
    for i in range(2 * ncolum):
        im = axs[i].imshow(etss[i], cmap=plt.cm.gist_rainbow, vmin=vmin, vmax=vmax)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        if i < ncolum:
            axs[i].set_title(text[i + 1], fontsize=14)

    cbar_ax = fig.add_axes([0.92, 0.1, 0.018, 0.8])  # colorbar 左 下 宽 高
    v1 = np.linspace(vmin, vmax, 6, endpoint=True)
    cbar1 = plt.colorbar(im, cax=cbar_ax, ticks=v1, extend='min')
    cbar1.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
    cbar1.ax.tick_params(labelsize=16)
    # plt.show()
    plt.savefig("occur_ets.png", dpi=300)
    plt.close()

    # ---------- storm plot------------
    ncolum = np.shape(storme_ets)[0] // 2
    fig, _ax = plt.subplots(nrows=2, ncols=ncolum, figsize=(11.05, 3.75),
                            sharex='col', sharey='row')
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.9, top=0.88, wspace=0, hspace=0)
    axs = _ax.flatten()
    vmin, vmax = 1, 0
    for i in range(2 * ncolum):
        if vmin > np.nanmin(storme_ets[i]):
            vmin = np.nanmin(storme_ets[i])
        if vmax < np.nanmax(storme_ets[i]):
            vmax = np.nanmax(storme_ets[i])

    vmin = 0.0
    for i in range(2 * ncolum):
        im = axs[i].imshow(storme_ets[i], cmap=plt.cm.gist_rainbow, vmin=vmin, vmax=vmax)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        if i < ncolum:
            axs[i].set_title(text[i + 1], fontsize=14)

    cbar_ax = fig.add_axes([0.92, 0.1, 0.018, 0.8])  # colorbar 左 下 宽 高
    v1 = np.linspace(vmin, vmax, 6, endpoint=True)
    cbar1 = plt.colorbar(im, cax=cbar_ax, ticks=v1, extend='min')
    cbar1.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
    cbar1.ax.tick_params(labelsize=16)
    # plt.show()
    plt.savefig("storm_ets.png", dpi=300)
    plt.close()
