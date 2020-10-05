import platform
import os
from Utility.util import Utility
from CmdHelp.CmdHelp import CmdHelper
from logger import Logger, OMLLogger
from oml import OML
import pandas as pd
import numpy as np
import torch
import getpass
from sklearn.metrics import f1_score


if __name__ == '__main__':
    os_name = platform.system()

    # path to dataset
    data_path = 'Downloads/emnist.csv'
    # experiment configuration name
    data_name = 'emnist'

    args = Utility.check_args(CmdHelper.get_parser())

    """ preprocessing """
    username = getpass.getuser()
    data_path = os.path.join('/home/'+username, data_path)
    log_path = os.path.join(args.base_dir, args.log_dir, data_name)
    checkpoint_dir = os.path.join(args.base_dir, args.checkpoint_dir, data_name)
    args.result_dir = os.path.join(args.base_dir, args.result_dir, data_name)

    Utility.setup(args)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    metric_logger = Logger(log_path)

    log_path = os.path.join(log_path, 'oml_train.log')

    oml_logger = OMLLogger.getLogger(log_path)

    """read data and split"""
    data = pd.read_csv(data_path, header=None).values
    print("Data Reading Done")
    n, _ = data.shape

    # randomly split
    idx = np.arange(len(data))
    np.random.shuffle(idx)

    train_num = int(args.train_ratio * n)
    valid_num = int(args.valid_ratio * n)

    train_idx = idx[:train_num]
    valid_idx = idx[train_num:train_num+valid_num]
    test_idx = idx[train_num+valid_num:]

    train_data = data[train_idx]
    valid_data = data[valid_idx]
    test_data = data[test_idx]

    # get features and labels
    train_feature = train_data[:, :-1]
    train_label = train_data[:, -1]

    valid_feature = valid_data[:, :-1]
    valid_label = valid_data[:, -1]

    test_feature = test_data[:, :-1]
    test_label = test_data[:, -1]

    """create OML object and start"""
    oml = OML(args, train_feature, train_label, metric_logger, oml_logger, checkpoint_dir)

    # start training
    oml.start(valid_feature=valid_feature, valid_label=valid_label, evaluate_valid=False)

    # build knn classifier
    oml.build_knn_classifier(train_feature, train_label, k=5, cuda=torch.cuda.is_available(), args=args)
    # predict on test dataset
    p_label, p_prob = oml.predict(test_feature)

    """analysis"""
    acc = np.sum(p_label == test_label)/len(test_label)
    print("Accuracy: ", acc, "Macro F1: ", f1_score(test_label, p_label, average='macro'))

    """save prediction result"""
    result_data = np.concatenate((test_label.reshape(-1, 1), p_label.reshape(-1, 1), p_prob.reshape(-1, 1)), axis=1)
    np.savetxt(os.path.join(args.result_dir, args.result_file), result_data, delimiter=',')
