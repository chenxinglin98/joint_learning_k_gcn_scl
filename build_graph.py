# =================================
# File : build_graph.py
# Description :
# Author : ChenX
# CREATE TIME : 2024/3/14 13:39
# =================================
import jsonlines
import numpy as np
from typing import List
from transformers import BertTokenizer


class BuildGraph(object):
    def __init__(self, tokenizer_path: str):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path)
        pass

    def create_graph(self, files: List[str]):
        """
        PMI(A,B) = LOG P(A,B) / P(A)P(B)
        :return:
        """
        # 计算词共享矩阵
        M = np.zeros(shape=(len(self.tokenizer), len(self.tokenizer)))

        for file in files:
            with jsonlines.open(file, "r") as reader:
                for row in reader:
                    text = list(row["kg_text"])
                    input_ids = self.tokenizer.encode(text)[1:-1]
                    for idx in input_ids:
                        for idy in input_ids:
                            if idx == idy:
                                continue

                            M[idx][idy] += 1

                    if "kg_augment" in row:
                        text = list(row["kg_augment"])
                        input_ids = self.tokenizer.encode(text)[1:-1]
                        for idx in input_ids:
                            for idy in input_ids:
                                if idx == idy:
                                    continue

                                M[idx][idy] += 1

        # 计算点互信息
        col_totals = M.sum(axis=0)
        row_totals = M.sum(axis=1)
        expected = np.outer(col_totals, row_totals) / M.sum()
        with np.errstate(divide="ignore"):
            res_log = np.log(M / expected)
        # 将inf转换为0.0
        res_log[np.isinf(res_log)] = 0.0
        res_log[res_log < 0] = 0.0

        return res_log


if __name__ == '__main__':
    g = BuildGraph(tokenizer_path=r"/mnt/storage/shared/model/chinese-roberta-wwm-ext/")
    files_ = [
        "datasets/train_kg.jsonl",
        "datasets/test_kg.jsonl"
    ]

    adj = g.create_graph(files=files_)
    np.save(file="datasets/adj.npy", arr=adj)
