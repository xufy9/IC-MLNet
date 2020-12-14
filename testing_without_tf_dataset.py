# -*- coding:UTF-8 -*-
import tensorflow as tf
import os
import numpy as np
import time
from model.utils import *
from config import D
from model.model_multistage import MultiStageModel
from verification import compute_verification
from model.log_configure import logger


def read_data(round):
    # ------------------------------ data name --------------------------------------
    train_sample_filepaths, train_labels_filepaths, \
    eval_sample_filepaths, eval_labels_filepaths, \
    test_sample_filepaths, test_labels_filepaths, prior_dic = read_train_eval_test_filename(round * 0.1,
                                                                                            round * 0.1 + 0.1)

    train_batchs_perE = math.ceil(np.shape(train_sample_filepaths)[0] / D.batch_size)
    eval_batchs_perE = math.ceil(np.shape(eval_sample_filepaths)[0] / D.batch_size)
    test_batchs_perE = math.ceil(np.shape(test_sample_filepaths)[0] / D.batch_size)

    logger.info(
        "batch_size: %d\n "
        "train sample: %d, %d batchs every train epoch\n "
        "valid sample: %d, %d batchs every valid epoch\n "
        "test sample: %d, %d batchs every test epoch\n",
        D.batch_size,
        np.shape(train_sample_filepaths)[0], train_batchs_perE,
        np.shape(eval_sample_filepaths)[0], eval_batchs_perE,
        np.shape(test_sample_filepaths)[0], test_batchs_perE)

    # ---------------------------- read epoch data ----------------------------------
    logger.info("read all train data begin")
    load_data_begin = time.time()

    features, labels = read_files(train_sample_filepaths, train_labels_filepaths, prior_dic, -5)
    # (?, h, w, sC)  (?, h, w, outC)
    eval_features, eval_labels = read_files(eval_sample_filepaths, eval_labels_filepaths, prior_dic, -5)
    test_features, test_labels = read_files(test_sample_filepaths, test_labels_filepaths, prior_dic, -5)

    np.save(os.path.join(D.result_dir, 'test_features_' + D.input_dataset + '-' + str(round)), test_features)
    np.save(os.path.join(D.result_dir, 'test_labels_' + D.input_dataset + '-' + str(round)), test_labels)

    read_elapse = time.time() - load_data_begin
    logger.info("read all train data end, read_elapse=%.4fs", read_elapse)

    filepaths = np.array(np.char.split(test_sample_filepaths, '.').tolist())
    filenames = [os.path.split(filepath)[-1] for filepath in filepaths[:, -2]]

    logger.debug("test_features: %s, test_labels: %s", str(np.shape(test_features)), str(np.shape(test_labels)))

    return test_features, test_labels, test_batchs_perE, filenames


def print_result_graph_and_para(test_features, test_labels, test_batchs_perE, filenames, inner_dir):
    ckpt = tf.train.get_checkpoint_state(os.path.join(D.model_dir, inner_dir))
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    graph = tf.get_default_graph()

    session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

    with tf.Session(config=session_config) as sess:
        test_begin = time.time()
        saver.restore(sess, ckpt.model_checkpoint_path)

        epoch_xx, epoch_yy, epoch_yy_ = [], [], []
        epoch_filename = []
        epoch_l = 0

        for b in range(test_batchs_perE):  # test_batchs_perE
            batch_begin = b * D.batch_size

            # avoid empty batch
            tower_size = math.floor(D.batch_size / D.num_gpus)
            if (b == test_batchs_perE - 1):
                final_b_sample = np.shape(test_features[b * D.batch_size:])[0]
                if final_b_sample < D.num_gpus:
                    test_batchs_perE -= 1
                    break
                else:
                    tower_size = math.floor(final_b_sample / D.num_gpus)

            t_xx, t_yy, t_yy_, t_batch_filenames, tb_l = sess.run(
                [[graph.get_tensor_by_name("GPU_0/x:0"), graph.get_tensor_by_name("GPU_1/x:0"),
                  graph.get_tensor_by_name("GPU_2/x:0"), graph.get_tensor_by_name("GPU_3/x:0")],
                 [graph.get_tensor_by_name("GPU_0/y:0"), graph.get_tensor_by_name("GPU_1/y:0"),
                  graph.get_tensor_by_name("GPU_2/y:0"), graph.get_tensor_by_name("GPU_3/y:0")],
                 [graph.get_tensor_by_name('GPU_0/multilayer/add_6:0'), graph.get_tensor_by_name('GPU_1/multilayer/add_6:0'),
                  graph.get_tensor_by_name('GPU_2/multilayer/add_6:0'),
                  graph.get_tensor_by_name('GPU_3/multilayer/add_6:0')],
                 [graph.get_tensor_by_name("GPU_0/filename:0"), graph.get_tensor_by_name("GPU_1/filename:0"),
                  graph.get_tensor_by_name("GPU_2/filename:0"), graph.get_tensor_by_name("GPU_3/filename:0")],
                 [graph.get_tensor_by_name('GPU_0/compute_loss/add_7:0'),
                  graph.get_tensor_by_name('GPU_1/compute_loss/add_7:0'),
                  graph.get_tensor_by_name('GPU_2/compute_loss/add_7:0'),
                  graph.get_tensor_by_name('GPU_3/compute_loss/add_7:0')]],
                feed_dict={
                    'GPU_0/x:0': test_features[batch_begin:(batch_begin + tower_size)],
                    'GPU_0/y:0': test_labels[batch_begin:(batch_begin + tower_size)],
                    "GPU_0/mask:0": (test_labels[batch_begin:(batch_begin + tower_size)] != -5),
                    'GPU_0/filename:0': filenames[batch_begin:(batch_begin + tower_size)],

                    'GPU_1/x:0': test_features[(batch_begin + tower_size):(batch_begin + 2 * tower_size)],
                    'GPU_1/y:0': test_labels[(batch_begin + tower_size):(batch_begin + 2 * tower_size)],
                    "GPU_1/mask:0": (test_labels[(batch_begin + tower_size):(batch_begin + 2 * tower_size)] != -5),
                    'GPU_1/filename:0': filenames[(batch_begin + tower_size):(batch_begin + 2 * tower_size)],

                    'GPU_2/x:0': test_features[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)],
                    'GPU_2/y:0': test_labels[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)],
                    "GPU_2/mask:0": (
                            test_labels[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)] != -5),
                    'GPU_2/filename:0': filenames[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)],

                    'GPU_3/x:0': test_features[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)],
                    'GPU_3/y:0': test_labels[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)],
                    "GPU_3/mask:0": (
                            test_labels[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)] != -5),
                    'GPU_3/filename:0': filenames[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)],

                    'epoch:0': 0, 'is_training:0': False})

            tl = 0
            for ld in tb_l: tl += ld
            tl = tl / D.num_gpus

            t_xx, t_yy, t_yy_ = np.concatenate(t_xx, axis=0), np.concatenate(t_yy, axis=0), np.concatenate(t_yy_, axis=0)
            t_batch_filenames = np.concatenate(t_batch_filenames, axis=0)
            epoch_filename.append(t_batch_filenames)

            epoch_l += tl
            epoch_xx.append(t_xx)
            epoch_yy.append(t_yy)
            epoch_yy_.append(t_yy_)

        test_end = time.time()
        epoch_xx, epoch_yy, epoch_yy_ = np.concatenate(epoch_xx, axis=0), np.concatenate(epoch_yy, axis=0), \
                                        np.concatenate(epoch_yy_, axis=0)
        epoch_l = epoch_l/test_batchs_perE
        epoch_a = compute_verification(epoch_yy_.copy(), epoch_yy.copy())

        logger.info("test: testtime = %.4fsec/sam\n mean_loss=\n  %s\nmean_acc=\n  %s",
                    (test_end - test_begin) / test_batchs_perE / D.batch_size, str(epoch_l), str(epoch_a))


if __name__ == '__main__':
    round = 7
    logger.info("################################# round %d begin ###########################", round)

    test_features, test_labels, test_batchs_perE, filenames = read_data(round)
    inner_dir = '128261 ml_mu_nolocal2d_add_wotDS7'
    print_result_graph_and_para(test_features, test_labels, test_batchs_perE, filenames, inner_dir)

    logger.info("################################# round %d end ###########################", round)
