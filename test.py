"""Evaluation script."""

import argparse
import json
import os

import torch

import models
from datasets.kg_dataset import KGDataset
from utils.train import avg_both, format_metrics

parser = argparse.ArgumentParser(description="Test")
parser.add_argument(
    '--model_dir', default="log/WN18RR",
    help="Model path"
)


def test(model_dir):
    # load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)

    # create dataset
    args.dataset = 'WN18RR_n-n//1-1'
    print(args.dataset)
    dataset_path = os.path.join("src_data", args.dataset)
    dataset = KGDataset(dataset_path, False)
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()
    # test_relation_examples = []
    # for i in range(0, 11):
    #     test_relation_examples.append(dataset.get_examples("test", rel_idx=i))

    # load pretrained model weights
    model = getattr(models, args.model)(args)
    device = 'cuda'
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))

    # eval
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    print(format_metrics(test_metrics, split='test'))
    # for i in range(0, 11):
    #     print("\t Results for relation {}".format(i))
    #     test_metrics = avg_both(*model.compute_metrics(test_relation_examples[i], filters))
    #     print(format_metrics(test_metrics, split="test"))
    return test_metrics


if __name__ == "__main__":
    args = parser.parse_args()
    test_metrics = test(args.model_dir)
    # print(format_metrics(test_metrics, split='test'))
