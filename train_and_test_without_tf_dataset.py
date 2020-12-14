# -*- coding: utf-8 -*-
import time
import copy
from model.utils import *
from config import D
from model.model_multistage import MultiStageModel
from verification import compute_verification
from model.log_configure import logger

tf.logging.set_verbosity(tf.logging.ERROR)
# --------------------------------- print hyperparameters --------------------------------------
logger.info(str(D))


def _average_gradients(grads_list):
    """Calculate the average gradient for each shared variable across all towers.
      Note that this function provides a synchronization point across all towers.
      Args:
        grads_list: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
      """
    average_grads = []
    for grad_and_vars in zip(*grads_list):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(values=grads, axis=0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train_gpu(h, w, model_dir, round):
    # ------------------------------ graph --------------------------------------
    with tf.Graph().as_default() as g, tf.device('/cpu:0'):
        names = locals()
        model = MultiStageModel(dict(D))
        epo = tf.placeholder(tf.float32, shape=[], name="epoch")
        is_training = tf.placeholder(tf.bool, shape=[], name="is_training")

        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        decayed_lr = tf.train.exponential_decay(D.learning_rate, epo, D.decay_epochs, D.decay_rate, staircase=True)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        trian_optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr)
        grad_list = []

        xlist, ylist, y_list, losslist, filename_list = [], [], [], [], []
        layers_sorted_and_feature, layerRes, layerout = [], [], []

        with tf.variable_scope(tf.get_variable_scope()):
            reuse = False
            for i in range(D.num_gpus):
                if i > 0: reuse = True
                with tf.device("/gpu:%d" % i), tf.name_scope('GPU_%d' % i) as scope:
                    names['xs%d' % i] = tf.placeholder(tf.float32, shape=[None, h, w, D.splited_channel], name="x")
                    names['y%d' % i] = tf.placeholder(tf.float32, shape=[None, h, w, D.out_channel], name="y")
                    names['mask%d' % i] = tf.placeholder(tf.float32, shape=[None, h, w, D.out_channel], name="mask")

                    names['y_%d' % i], names['layers_sorted_and_feature%d' % i], \
                    names['layerRes%d' % i], names['layerout%d' % i] = model.multilayer_redense(names['xs%d' % i],
                                                                                                training=is_training,
                                                                                                reuse=reuse)
                    loss_dic = model.compute_loss(names['y%d' % i], names['y_%d' % i],
                                                  names['mask%d' % i], names['layerout%d' % i])

                    logger.debug("gpu %d, y_ shape: %s", i, names['y_%d' % i].get_shape().as_list())
                    logger.debug("gpu %d, variables_list len: %s" % (i, str(len(model.train_variables))))

                    names['filename%d' % i] = tf.placeholder(tf.string, shape=[None, ], name="filename")

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    grad = trian_optimizer.compute_gradients(loss_dic["loss"])
                    grad_list.append(grad)

                    xlist.append(names['xs%d' % i])
                    ylist.append(names['y%d' % i])
                    filename_list.append(names['filename%d' % i])

                    y_list.append(names['y_%d' % i])
                    losslist.append(loss_dic)

                    layers_sorted_and_feature.append(names['layers_sorted_and_feature%d' % i])
                    layerRes.append(names['layerRes%d' % i])
                    layerout.append(names['layerout%d' % i])

        ave_grad = _average_gradients(grad_list)
        train_op = trian_optimizer.apply_gradients(ave_grad, global_step=global_step)

        # # Add histograms for gradients.
        # for grad, var in ave_grad:
        #     if grad is not None:
        #         summaries.append(tf.summary.histogram('gradients/' + var.op.name, grad))
        # kernel_store = {}
        # # Add histograms for trainable variables.
        # for var in tf.trainable_variables():
        #     summaries.append(tf.summary.histogram('variables/' + var.op.name, var))
        #     logger.info("Var %s\nvar op %s", str(var), str(var.op).replace('\n', ''))
        #     if 'conv2d' and 'kernel' in var.op.name:  # (k_s, k_s, inc, num_features)
        #         kernel_store[var.op.name] = var

        summary_op = tf.summary.merge(summaries)
        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.global_variables_initializer()

    # ----------- count parameters -------------------
    logger.debug(str(xlist) + '\n' + str(ylist) + '\n' + str(y_list) + '\n' + str(losslist))
    logger.debug(str([names['mask0'], names['mask1'], names['mask2'], names['mask3']]) + '\n' + str(filename_list))
    logger.debug(str(epo) + '\n' + str(is_training) + '\n' + str(global_step) + '\n' + str(train_op).replace('\n', ''))

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

    if os.path.exists(os.path.join(D.data_dir, 'features_' + D.input_dataset + '-' + str(round) + '.npy')):
        features = np.load(os.path.join(D.data_dir, 'features_' + D.input_dataset + '-' + str(round) + '.npy'))
        labels = np.load(os.path.join(D.data_dir, 'labels_' + D.input_dataset + '-' + str(round) + '.npy'))
        eval_features = np.load(os.path.join(D.data_dir, 'eval_features_' + D.input_dataset + '-' +
                                             str(round) + '.npy'))
        eval_labels = np.load(os.path.join(D.data_dir, 'eval_labels_' + D.input_dataset + '-' +
                                           str(round) + '.npy'))
        test_features = np.load(os.path.join(D.data_dir, 'test_features_' + D.input_dataset + '-' +
                                             str(round) + '.npy'))
        test_labels = np.load(os.path.join(D.data_dir, 'test_labels_' + D.input_dataset + '-' +
                                           str(round) + '.npy'))
    else:
        features, labels = read_files(train_sample_filepaths, train_labels_filepaths, prior_dic, -5)
        # (?, h, w, sC)  (?, h, w, outC)
        eval_features, eval_labels = read_files(eval_sample_filepaths, eval_labels_filepaths, prior_dic, -5)
        test_features, test_labels = read_files(test_sample_filepaths, test_labels_filepaths, prior_dic, -5)

        np.save(os.path.join(D.data_dir, 'features_' + D.input_dataset + '-' + str(round)), features)
        np.save(os.path.join(D.data_dir, 'labels_' + D.input_dataset + '-' + str(round)), labels)
        np.save(os.path.join(D.data_dir, 'eval_features_' + D.input_dataset + '-' + str(round)), eval_features)
        np.save(os.path.join(D.data_dir, 'eval_labels_' + D.input_dataset + '-' + str(round)), eval_labels)
        np.save(os.path.join(D.data_dir, 'test_features_' + D.input_dataset + '-' + str(round)), test_features)
        np.save(os.path.join(D.data_dir, 'test_labels_' + D.input_dataset + '-' + str(round)), test_labels)

    read_elapse = time.time() - load_data_begin
    logger.info("read all train data end, read_elapse=%.4fs", read_elapse)

    # ------------------------------train session--------------------------------------
    config = tf.ConfigProto(allow_soft_placement=True)  # , log_device_placement=True
    config.gpu_options.allow_growth = True
    with tf.Session(graph=g, config=config) as sess:
        sess.run(init_op)

        graph_def = sess.graph.as_graph_def(add_shapes=True)
        train_summary_writer = tf.summary.FileWriter(model_dir + "train")  # , graph_def=graph_def
        valid_summary_writer = tf.summary.FileWriter(model_dir + "valid")
        test_summary_writer = tf.summary.FileWriter(model_dir + "test")

        # run op
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        logger.info("training begin")
        total_step = 0
        indices = np.arange(0, np.shape(train_sample_filepaths)[0])
        epoch_result = open(os.path.join(model_dir, 'valid_log.txt'), 'a')
        total_step_loss = open(os.path.join(model_dir, 'step_loss.txt'), 'a')

        for epoch in range(D.epochs):
            np.random.shuffle(indices)
            features = np.take(features, indices, axis=0)
            labels = np.take(labels, indices, axis=0)

            filepaths = np.array(np.char.split(np.take(train_sample_filepaths, indices, axis=0), '.').tolist())
            filenames = [os.path.split(filepath)[-1] for filepath in filepaths[:, -2]]

            epoch_xx, epoch_yy, epoch_yy_ = [], [], []
            epoch_l = {}

            epoch_begin = time.time()
            tbp = copy.copy(train_batchs_perE)
            for b in range(train_batchs_perE):  # train_batchs_perE
                batch_begin = b * D.batch_size

                # avoid empty batch
                tower_size = math.floor(D.batch_size / D.num_gpus)
                if (b == train_batchs_perE - 1):
                    final_b_sample = np.shape(features[b * D.batch_size:])[0]
                    if final_b_sample < D.num_gpus:
                        tbp -= 1
                        break
                    else:
                        tower_size = math.floor(final_b_sample / D.num_gpus)

                step_begin = time.time()
                _, ldic, xx, yy, yy_, batch_filenames, train_summary_str, \
                layer_sd_and_f, lmhsRes, out_lmhs = sess.run(
                    [train_op, losslist, xlist, ylist, y_list, filename_list, summary_op,
                     layers_sorted_and_feature, layerRes, layerout],
                    feed_dict={
                        names['xs0']: features[batch_begin:(batch_begin + tower_size)],
                        names['y0']: labels[batch_begin:(batch_begin + tower_size)],
                        names['mask0']: (labels[batch_begin:(batch_begin + tower_size)] != -5),
                        names['filename0']: filenames[batch_begin:(batch_begin + tower_size)],

                        names['xs1']: features[(batch_begin + tower_size):(batch_begin + 2 * tower_size)],
                        names['y1']: labels[(batch_begin + tower_size):(batch_begin + 2 * tower_size)],
                        names['mask1']: (labels[(batch_begin + tower_size):(batch_begin + 2 * tower_size)] != -5),
                        names['filename1']: filenames[(batch_begin + tower_size):(batch_begin + 2 * tower_size)],

                        names['xs2']: features[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)],
                        names['y2']: labels[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)],
                        names['mask2']: (labels[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)] != -5),
                        names['filename2']: filenames[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)],

                        names['xs3']: features[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)],
                        names['y3']: labels[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)],
                        names['mask3']: (labels[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)] != -5),
                        names['filename3']: filenames[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)],

                        epo: epoch, is_training: True},
                    options=run_options, run_metadata=run_metadata)
                step_end = time.time()

                ####### B=bs/gpunum
                ## xx           list(len=gpunum) element:numpy (B, h, w, sC)
                ## yy, yy_      list(len=gpunum) element:numpy (B, h, w, outC)
                ## layer_sd_and_f, ,
                ##              list(len=gpunum) element:list(len=2(sd&f)) element:numpy(B, h, w, sC*4)
                ##                                                                 numpy(B, h, w, c*4)
                ## lmhsRes      list(len=gpunum) element:list(len=4(l&m&h&st)) element:numpy(B, h, w, outC)
                ## out_lmhs     list(len=gpunum) element:list(len=4(l&m&h&st)) element:numpy(B, h, w, outC)
                #######

                xx, yy, yy_ = np.concatenate(xx, axis=0), np.concatenate(yy, axis=0), np.concatenate(yy_, axis=0)

                l_dic = {}
                for ld in ldic:
                    l_dic = dic_add(l_dic, ld)
                l_dic = dic_div_constant(l_dic, D.num_gpus)

                if (b == 0 or b == 40) and (np.nansum(layer_sd_and_f[0][0][:, :, :, D.splited_channel * 3:]) > 100):
                    acc_dic = compute_verification(yy_.copy(), yy.copy())
                    logger.debug("train_epoch=%d, mini_batch=%d, total_step=%d, train_time=%.4fs\n"
                                 "loss:\n  %s\n acc:\n  %s",
                                 epoch, b, total_step, step_end - step_begin, str(l_dic), str(acc_dic))

                    visualize_inputs(xx, yy, outpath=os.path.join(model_dir, 'trainresult'),
                                     filenames=np.char.add(np.char.add("epoch%d-step%d-" % (epoch, b),
                                                                       list(batch_filenames)),
                                                           "-inputs_"))

                    batch_filenames = np.concatenate(batch_filenames, axis=0)
                    tower_size = math.ceil(D.batch_size / D.num_gpus)
                    visualize_ensembles(layer_sd_and_f[0][0], outpath=os.path.join(model_dir, 'trainresult'), dpi=300,
                                        filenames=np.char.add(
                                            np.char.add("epoch%d-step%d-" % (epoch, b), list(batch_filenames[:tower_size])),
                                            "-layerinput"))

                assert not np.isnan(l_dic["loss"]), 'Model diverged with loss = NaN'

                total_step_loss.write("%.10f\n" % l_dic['loss'])

                epoch_l = dic_add(epoch_l, l_dic)
                epoch_xx.append(xx)
                epoch_yy.append(yy)
                epoch_yy_.append(yy_)
                total_step += 1

            train_summary_writer.add_summary(train_summary_str, epoch)
            epoch_xx, epoch_yy, epoch_yy_ = np.concatenate(epoch_xx, axis=0), np.concatenate(epoch_yy, axis=0), \
                                            np.concatenate(epoch_yy_, axis=0)
            epoch_a = compute_verification(epoch_yy_.copy(), epoch_yy.copy())
            epoch_l = dic_div_constant(epoch_l, tbp)
            logger.info("train_epoch=%d, epoch_train_time=%.4fs\n epoch_loss:\n  %s\n epoch_acc:\n  %s",
                        epoch, time.time() - epoch_begin, str(epoch_l), str(epoch_a))

            train_summary = tf.Summary()
            for key, value in epoch_l.items():
                train_summary.value.add(tag='mean_loss/' + key, simple_value=epoch_l[key])
            for key, value in epoch_a.items():
                train_summary.value.add(tag='mean_acc/' + key, simple_value=epoch_a[key])
            train_summary_writer.add_summary(train_summary, epoch)

            if epoch % 1 == 0 or epoch == D.epochs - 1:
                train_summary_writer.add_run_metadata(run_metadata, 'epoch%05d' % epoch)

            if epoch % (D.decay_epochs - 1) == 0 or epoch == D.epochs - 1:
                checkpoint_path = os.path.join(model_dir, D.model_name_reg)
                saver.save(sess, checkpoint_path, global_step=epoch)
                logger.info("saved to %s\n total_epoch=%d", checkpoint_path, epoch)

            # ---------------------------------- valid -----------------------------------
            if (epoch % 1 == 0 or epoch == D.epochs - 1):
                valid_begin = time.time()
                eval_indeces = np.arange(0, np.shape(eval_sample_filepaths)[0])
                np.random.shuffle(eval_indeces)
                eval_features = np.take(eval_features, eval_indeces, axis=0)
                eval_labels = np.take(eval_labels, eval_indeces, axis=0)
                filepaths = np.array(
                    np.char.split(np.take(eval_sample_filepaths, eval_indeces, axis=0), '.').tolist())
                filenames = [os.path.split(filepath)[-1] for filepath in filepaths[:, -2]]

                epoch_xx, epoch_yy, epoch_yy_ = [], [], []
                epoch_filename = []
                epoch_l = {}

                ebp = eval_batchs_perE
                for b in range(eval_batchs_perE):
                    batch_begin = b * D.batch_size

                    # avoid empty batch
                    tower_size = math.floor(D.batch_size / D.num_gpus)
                    if (b == eval_batchs_perE - 1):
                        final_b_samples = np.shape(eval_features[b * D.batch_size:])[0]
                        if final_b_samples < D.num_gpus:
                            ebp -= 1
                            break
                        else:
                            tower_size = math.floor(final_b_samples / D.num_gpus)

                    v_xx, v_yy, v_yy_, v_batch_filenames, eva_loss_d, valid_summary_str = sess.run(
                        [xlist, ylist, y_list, filename_list, losslist, summary_op],
                        feed_dict={
                            names['xs0']: eval_features[batch_begin:(batch_begin + tower_size)],
                            names['y0']: eval_labels[batch_begin:(batch_begin + tower_size)],
                            names['mask0']: (eval_labels[batch_begin:(batch_begin + tower_size)] != -5),
                            names['filename0']: filenames[batch_begin:(batch_begin + tower_size)],

                            names['xs1']: eval_features[(batch_begin + tower_size):(batch_begin + 2 * tower_size)],
                            names['y1']: eval_labels[(batch_begin + tower_size):(batch_begin + 2 * tower_size)],
                            names['mask1']: (
                                    eval_labels[(batch_begin + tower_size):(batch_begin + 2 * tower_size)] != -5),
                            names['filename1']: filenames[(batch_begin + tower_size):(batch_begin + 2 * tower_size)],

                            names['xs2']: eval_features[
                                          (batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)],
                            names['y2']: eval_labels[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)],
                            names['mask2']: (
                                    eval_labels[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)] != -5),
                            names['filename2']: filenames[
                                                (batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)],

                            names['xs3']: eval_features[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)],
                            names['y3']: eval_labels[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)],
                            names['mask3']: (
                                    eval_labels[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)] != -5),
                            names['filename3']: filenames[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)],

                            epo: epoch, is_training: False})

                    vl_dic = {}
                    for ld in eva_loss_d:
                        vl_dic = dic_add(vl_dic, ld)
                    vl_dic = dic_div_constant(vl_dic, D.num_gpus)
                    v_xx, v_yy, v_yy_ = np.concatenate(v_xx, axis=0), np.concatenate(v_yy, axis=0), \
                                        np.concatenate(v_yy_, axis=0)
                    v_batch_filenames = np.concatenate(v_batch_filenames, axis=0)
                    epoch_filename.append(v_batch_filenames)

                    epoch_l = dic_add(epoch_l, vl_dic)
                    epoch_xx.append(v_xx)
                    epoch_yy.append(v_yy)
                    epoch_yy_.append(v_yy_)

                valid_summary_writer.add_summary(valid_summary_str, epoch)
                epoch_xx, epoch_yy, epoch_yy_ = np.concatenate(epoch_xx, axis=0), np.concatenate(epoch_yy, axis=0), \
                                                np.concatenate(epoch_yy_, axis=0)
                epoch_l = dic_div_constant(epoch_l, ebp)
                epoch_a = compute_verification(epoch_yy_.copy(), epoch_yy.copy())
                logger.info("valid_epoch=%d, epoch_valid_time=%.4fs, mean_loss:\n  %s\nmean_acc:\n  %s",
                            epoch, time.time() - valid_begin, str(epoch_l), str(epoch_a))

                if epoch % (D.decay_epochs - 1) == 0 or epoch == D.epochs - 1:
                    np.save(os.path.join(model_dir, 'valid_name%d' % epoch), np.concatenate(epoch_filename, axis=0))
                    np.save(os.path.join(model_dir, 'valid_label%d' % epoch), epoch_yy)
                    np.save(os.path.join(model_dir, 'valid_predict%d' % epoch), epoch_yy_)

                for key in epoch_a.keys():
                    epoch_result.write('%.10f ' % epoch_a[key])
                epoch_result.write('\n')

                vaan_summary = tf.Summary()
                for key, value in epoch_l.items():
                    vaan_summary.value.add(tag='mean_loss/' + key, simple_value=epoch_l[key])
                for key, value in epoch_a.items():
                    vaan_summary.value.add(tag='mean_acc/' + key, simple_value=epoch_a[key])
                valid_summary_writer.add_summary(vaan_summary, epoch)

        logger.info("training finished")

        if D.is_test:
            logger.info("testing begin")
            test_begin = time.time()

            filepaths = np.array(np.char.split(test_sample_filepaths, '.').tolist())
            filenames = [os.path.split(filepath)[-1] for filepath in filepaths[:, -2]]

            epoch_xx, epoch_yy, epoch_yy_ = [], [], []
            epoch_filename = []
            epoch_l = {}

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
                    [xlist, ylist, y_list, filename_list, losslist],
                    feed_dict={
                        names['xs0']: test_features[batch_begin:(batch_begin + tower_size)],
                        names['y0']: test_labels[batch_begin:(batch_begin + tower_size)],
                        names['mask0']: (test_labels[batch_begin:(batch_begin + tower_size)] != -5),
                        names['filename0']: filenames[batch_begin:(batch_begin + tower_size)],

                        names['xs1']: test_features[(batch_begin + tower_size):(batch_begin + 2 * tower_size)],
                        names['y1']: test_labels[(batch_begin + tower_size):(batch_begin + 2 * tower_size)],
                        names['mask1']: (test_labels[(batch_begin + tower_size):(batch_begin + 2 * tower_size)] != -5),
                        names['filename1']: filenames[(batch_begin + tower_size):(batch_begin + 2 * tower_size)],

                        names['xs2']: test_features[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)],
                        names['y2']: test_labels[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)],
                        names['mask2']: (
                                test_labels[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)] != -5),
                        names['filename2']: filenames[(batch_begin + 2 * tower_size):(batch_begin + 3 * tower_size)],

                        names['xs3']: test_features[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)],
                        names['y3']: test_labels[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)],
                        names['mask3']: (
                                test_labels[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)] != -5),
                        names['filename3']: filenames[(batch_begin + 3 * tower_size):(batch_begin + D.batch_size)],

                        epo: epoch, is_training: False})

                tl_dic = {}
                for ld in tb_l:
                    tl_dic = dic_add(tl_dic, ld)
                tl_dic = dic_div_constant(tl_dic, D.num_gpus)

                t_xx, t_yy, t_yy_ = np.concatenate(t_xx, axis=0), np.concatenate(t_yy, axis=0), np.concatenate(t_yy_, axis=0)
                t_batch_filenames = np.concatenate(t_batch_filenames, axis=0)
                epoch_filename.append(t_batch_filenames)

                epoch_l = dic_add(epoch_l, tl_dic)
                epoch_xx.append(t_xx)
                epoch_yy.append(t_yy)
                epoch_yy_.append(t_yy_)

            test_end = time.time()
            epoch_xx, epoch_yy, epoch_yy_ = np.concatenate(epoch_xx, axis=0), np.concatenate(epoch_yy, axis=0), \
                                            np.concatenate(epoch_yy_, axis=0)
            epoch_l = dic_div_constant(epoch_l, test_batchs_perE)
            epoch_a = compute_verification(epoch_yy_.copy(), epoch_yy.copy())

            logger.info("test: testtime = %.4fsec/sam\n mean_loss=\n  %s\nmean_acc=\n  %s",
                        (test_end - test_begin) / test_batchs_perE / D.batch_size, str(epoch_l), str(epoch_a))

            with open(os.path.join('test_result.txt'), 'a') as f:
                f.write("round %d\n" % round)
                for key in epoch_a.keys():
                    f.write("%s:%s " % (key, str(epoch_a[key])))
                f.write("\n")

            logger.info("testing finished")
            return epoch_a


def train_and_test(round, model_dir):
    h = D.input_h
    w = D.input_w
    if D.num_gpus > 0:
        return train_gpu(h, w, model_dir, round)


if __name__ == '__main__':
    if D.is_cross:
        mean_acc = {}
        outerdir = D.model_dir + 'ml_' + D.input_dataset[:2] + '_' + D.block_type + '_' + D.merge_type + '_wotDS/'

        for i in range(0, 10, 1):
            logger.info("################################# round %d begin ###########################", i)
            model_dir = outerdir + D.sub_dir[:-1] + str(i) + "/"

            acc_dic = train_and_test(i, model_dir)
            mean_acc = dic_add(mean_acc, acc_dic)
            logger.info("################################# round %d end ###########################", i)
        mean_acc = dic_div_constant(mean_acc, 10)
        logger.info("AFT\nMLNLNet : %s", str(mean_acc))

        valid_log = []
        for dir_or_files in os.listdir(outerdir):
            if os.path.isdir(dir_or_files):
                valid_log.append(np.loadtxt(os.path.join(outerdir, dir_or_files, 'valid_log.txt')))
        valid_log = np.mean(np.array(valid_log), axis=0)
        np.savetxt(os.path.join(outerdir, 'valid_log.txt'), valid_log)

    else:
        round = 7
        logger.info("################################# round %d begin ###########################", round)
        model_dir = D.model_dir + \
                    'ml_' + D.input_dataset[:2] + '_' + D.block_type + '_' + D.merge_type + '_wotDS' + \
                    str(round) + "/"

        train_and_test(round, model_dir)
        logger.info("################################# round %d end ###########################", round)
