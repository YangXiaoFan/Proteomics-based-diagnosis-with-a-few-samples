# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif


if __name__ == '__main__':

    for _ in range(10):
        model_name = "ESCC_MAE"  

        x = import_module('models.' + model_name)
        config = x.Config()
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

        start_time = time.time()
        print("Loading data...")
        set_1_train, set_1_dev, set_2_train, set_2_test = build_dataset(config)
        set_1_train_iter = build_iterator(set_1_train, config, aug=True)
        set_1_dev_iter = build_iterator(set_1_dev, config, aug=True)
        config.batch_size = 1
        set_2_train_iter = build_iterator(set_2_train, config)
        set_2_test_iter = build_iterator(set_2_test, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        # train
        model = x.Model(config).to(config.device)
        model.initialize()
        train(config, model, set_1_train_iter, set_1_dev_iter, set_2_train_iter, set_2_test_iter)
