from easydict import EasyDict

D = EasyDict()
D.num_gpus = 4
D.batch_size = 24
D.epochs = 80
D.decay_epochs = 20
D.decay_rate = 0.5
D.learning_rate = 1e-3

D.input_dataset = 'ec_pf_tp_AT24_33x33_025'  #'multiorigin_cf_tp_AT24_33x33_025'
D.block_type = 'nolocal2d'  # nolocal2d  conv2d
D.merge_type = 'add'  # concat add
D.model_dir = './summary_and_ckpt/'

D.is_test = False
D.is_cross = False
D.sub_dir = 'cross/'
D.data_dir = './datasets/'

D.num_filters = 64
D.cut_dim = 16


D.input_h = 33
D.input_w = 33
D.splited_channel = 50
D.input_channel = 50

D.out_channel = 1
D.res_dense_block = 4
D.dense_block = 3
D.in_dense_layers = 4


D.enable_function = False
D.model_name_reg = "model.ckpt"

