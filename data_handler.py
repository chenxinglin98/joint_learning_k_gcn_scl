# =================================
# File : data_handler.py
# Description :
# Author : ChenX
# CREATE TIME : 2024/3/15 14:25
# =================================
import json
import jsonlines
import pandas as pd
from copy import deepcopy
from brain.knowgraph import KnowledgeGraph

frame = pd.read_csv("datasource/test.tsv")
frame.head()


def get_corpus():
    """
    整理数据格式
    :return:
    """
    data_list = []
    with open("datasource/val.tsv", "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.replace("\n", "")
            parts = line.split("\t")
            data_list.append({"text": parts[1], "label": parts[0]})

    print(len(data_list))

    with jsonlines.open("datasource/val.jsonl", "w") as writer:
        for row in data_list:
            writer.write(row)


def sample_couple():
    data_list = {}
    with open("datasource/1_train_data_augmented.json", "r", encoding="utf-8") as reader:
        info_list = json.load(reader)
        info_list = info_list["questions"]

        i = 0
        while i < len(info_list) - 2:
            data_list[info_list[i]] = [info_list[i + 1], info_list[i + 2]]
            i += 3

    train = []
    with jsonlines.open("datasource/train.jsonl", "r") as reader:
        for row in reader:
            train.append(row)

    data_info = []
    for item in train:
        try:
            values = data_list[item["text"]]
            for v in values:
                item["text_a"] = v
                data_info.append(deepcopy(item))
        except Exception as e:
            print(item)

    with jsonlines.open("datasource/train_point.jsonl", "w") as writer:
        for row in data_info:
            writer.write(row)


def handle_couple():
    data_list = []
    with open("datasource/1_train_data_augmented.json", "r", encoding="utf-8") as reader:
        info_list = json.load(reader)
        info_list = info_list["questions"]

        for row in info_list:
            query = row["question"]
            label = row["label"]

            for q in row["augmented_questions"]:
                data_list.append({"text": query, "augment_query": q, "label": label})

    with jsonlines.open("datasource/train.jsonl", "w") as writer:
        for row in data_list:
            writer.write(row)


def handle_test():
    data_list = []
    with open("datasource/test_data_raw.json", "r", encoding="utf-8") as reader:
        info_list = json.load(reader)

        questions = info_list["questions"]
        labels = info_list["labels"]

        for q, l in zip(questions, labels):
            data_list.append({"text": q, "label": l})

    with jsonlines.open("datasets/test.jsonl", "w") as writer:
        for row in data_list:
            writer.write(row)


def assemble_kg():
    kgm = KnowledgeGraph(spo_files=['HowNet', 'Medical'])

    train = []
    with jsonlines.open("datasets/train.jsonl", "r") as reader:
        for row in reader:
            # {"text": "印度洋周边有哪些国家？", "augment_query": "哪些国家周围环绕着印度洋？", "label": 0}
            kg_text, _, _, _ = kgm.add_knowledge_with_vm(sent_batch=[row["text"], row["augment_query"]])

            kg_text_1 = "".join([x for x in kg_text[0] if x != "[PAD]"])
            kg_text_2 = "".join([x for x in kg_text[1] if x != "[PAD]"])
            row["kg_text"] = kg_text_1
            row["kg_augment"] = kg_text_2

            train.append(row)

    test = []
    with jsonlines.open("datasets/test.jsonl", "r") as reader:
        for row in reader:
            # {"text": "印度洋周边有哪些国家？", "augment_query": "哪些国家周围环绕着印度洋？", "label": 0}
            kg_text, _, _, _ = kgm.add_knowledge_with_vm(sent_batch=[row["text"]])
            kg_text_1 = "".join([x for x in kg_text[0] if x != "[PAD]"])
            row["kg_text"] = kg_text_1

            test.append(row)

    with jsonlines.open("datasets/train_kg.jsonl", "w") as writer:
        for row in train:
            writer.write(row)

    with jsonlines.open("datasets/test_kg.jsonl", "w") as writer:
        for row in test:
            writer.write(row)


if __name__ == '__main__':
    assemble_kg()
