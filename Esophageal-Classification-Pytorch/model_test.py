# coding: UTF-8
import random
import torch
import numpy as np
import pandas as pd
from train_eval import evaluate
from importlib import import_module
from utils import build_iterator


def load_dataset(path):
    contents = []
    data = pd.read_excel(path, sheet_name = 0)
    all_label = data.iloc[0]
    all_label = all_label.tolist()
    all_content = data.iloc[1:5446]
    all_content = all_content.values

    for col in range(all_content.shape[1]):
        content = all_content[:, col]
        content = np.append(content, np.array([0, 0, 0]))
        label = all_label[col]
        label = 1 if label == "T" else 0
        content_pad = (content != 0).astype(int)
        contents.append((content, label, content_pad))
    
    order = np.array(range(0, len(contents)), dtype=int)

    return [contents[order[i]] for i in range(0, 10)]


def build_dataset(config):
    all_data = load_dataset(config.data_path)
    return all_data


if __name__ == '__main__':

    x = import_module('models.ESCC_CNN')
    config = x.Config()
    config.batch_size = 1

    all_data = build_dataset(config)
    all_data_iter = build_iterator(all_data, config)

    # test
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    with torch.no_grad():
        for texts, labels in all_data_iter:
            outputs = model(texts)
