# -*- coding: utf-8 -*-
# -*- coding: euc-kr -*-
import os
from general_utils import get_logger
import glob

class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # create instance of logger
        self.logger = get_logger(self.log_path)
    d = 50
    # general config
    N = len(glob.glob("results/tb/*"))
    tensorboard_log_path = "results/tb/log" + str(N)
    output_path = "results/crf/"
    model_output = output_path + "model/model2.ckpt"  #50 : model2.ckpt / model25 / model75...
    log_path = output_path + "log.txt"

    # embeddings
    dim = 50 # 50
    t_dim = 61 # 61
    pos_dim = 50  #50
    dim_char = 50  #50
    dic_dim = 4

    glove_filename = "rsc/{}/w2v_{}_add_dic.txt".format(dim, t_dim)
    glove_uni_filename = "rsc/{}/word_unigram_{}.txt".format(dim, dim)

    glove_feature = "rsc/{}/f2v_{}.txt".format(dim, pos_dim)

    NEdic_filename = "rsc/dic/NE_dic.txt"

    # trimmed embeddings (created from glove_filename with build_data.py)
    trimmed_filename = "rsc/word2vec/w2v.{}.trimmed.npz".format(t_dim)
    uni_trimmed_filename = "rsc/word2vec/uni_w2v.{}.trimmed.npz".format(dim)

    # trimmed dic feature
    trimmed_dic = "rsc/dic/NE_dic.trimmed.npz"
    feature_trimmed_filename = "rsc/word2vec/pos2vec.{}.trimmed.npz".format(pos_dim)

    # dataset
    dev_filename = "data_format/result_evaluate.txt"
    test_filename = "data_format/test.txt"
    train_filename = "data_format/result_train.txt"
    max_iter = None  # if not None, max number of examples

    # vocab (created from dataset with build_data.py)
    words_filename = "rsc/words.txt"
    uni_words_filename = "rsc/uni_words.txt"
    tags_filename = "rsc/tags.txt"
    chars_filename = "rsc/chars.txt"
    pos_filename = "rsc/pos.txt"

    # training
    train_embeddings = False
    uni_train_embeddings = False
    pos_train_embeddings = False
    dic_train_embeddings = False

    nepochs = 30
    dropout = 0.5
    batch_size = 20
    lr_method = "adam"
    lr = 0.008
    lr_decay = 0.9
    clip = -1  # if negative, no clipping
    nepoch_no_imprv = 10
    reload = False

    # model hyperparameters
    hidden_size = 300  #300
    char_hidden_size = 300  #300

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    crf = True  # if crf, training is 1.7x slower on CPU
    chars = True  # if char embedding, training is 3.5x slower on CPU
    bi_rnn = True

