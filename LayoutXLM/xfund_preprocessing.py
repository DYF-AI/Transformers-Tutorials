import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
import tqdm

print(torch.cuda.is_available())

import os
import json
# datasets == 2.5.1
import datasets
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from datasets.arrow_writer import ArrowWriter

# huggingface-hub-0.10.1
from transformers import AutoTokenizer

logger = datasets.logging.get_logger(__name__)

# DP = r"J:\dataset\document-intelligence\XFUND\zh"
# OP = r"J:\dataset\document-intelligence\XFUND\output"
# MP = r"J:\model\pretrained-model\bert_torch"


DP = "/mnt/j/dataset/document-intelligence/XFUND/zh"
OP = "/mnt/j/dataset/document-intelligence/XFUND/output"
MP = "/mnt/j/model/pretrained-model/torch"

def load_image(image_path, size=None):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if size is not None:
        # resize image
        image = image.resize((size, size))
        image = np.asarray(image)
        image = image[:, :, ::-1]  # flip color channels from RGB to BGR
        image = image.transpose(2, 0, 1)  # move channels to first dimension
    return image, (w, h)


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]


def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


# bbox标准化
# def normalize_bbox(bbox, size):
#     width, height = size
#
#     def clip(num, min_num, max_num):
#         return min(max(num, min_num), max_num)
#
#     (x0, y0), (x1, y1) = bbox
#     x0 = clip(0, int((x0 / width) * 1000), 1000)
#     y0 = clip(0, int((y0 / height) * 1000), 1000)
#     x1 = clip(0, int((x1 / width) * 1000), 1000)
#     y1 = clip(0, int((y1 / height) * 1000), 1000)
#
#     if x1 < x0:
#         x1 = x0
#     if y0 > y1:
#         y1 = y0
#
#     assert x1 >= x0
#     assert y1 >= y0
#     return [x0, y0, x1, y1]

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


# xlm_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
# tokenizer = AutoTokenizer.from_pretrained(r"J:\model\pretrained-model\bert_torch\xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained(os.path.join(MP, "xlm-roberta-base"))

# 关系抽取数据处理
def generate_example_relation_extraction(image_path, annotation_path):
    logger.info("Generating examples from = %s %s", image_path, annotation_path)
    with open(annotation_path, encoding="utf-8", mode="r") as fi:
        ann_infos = json.load(fi)
        document_list = ann_infos["documents"]

    for guid, doc, in enumerate(document_list):
        image_file = os.path.join(image_path, doc["img"]["fname"])
        print(image_file)
        size = [doc["img"]["width"], doc["img"]["height"]]
        image, size = load_image(image_file, size=224)
        original_image, _ = load_image(image_file)
        document = doc["document"]
        tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
        entities = []
        relations = []
        id2label = {}
        entity_id_to_index_map = {}
        empty_entity = set()
        for line in document:
            if len(line["text"]) == 0:
                empty_entity.add(line["id"])
                continue
            id2label[line["id"]] = line["label"]
            relations.extend([tuple(sorted(l)) for l in line["linking"]])
            # 将文本转为id
            # offset_mapping：每一个token具体占了几个char，中文都是一个char，英文就看单词的长度， [PAD]全都是0
            tokenized_inputs = tokenizer(
                line["text"],
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
            text_length = 0
            ocr_length = 0
            bbox = []
            for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                if token_id == 6:
                    bbox.append(None)
                    continue
                text_length += offset[1] - offset[0]
                tmp_box = []
                while ocr_length < text_length:
                    ocr_word = line["words"].pop(0)
                    ocr_length += len(
                        tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                    )
                    tmp_box.append(simplify_bbox(ocr_word["box"]))
                if len(tmp_box) == 0:
                    tmp_box = last_box
                bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                last_box = tmp_box  # noqa
            # 计算每个字|词的坐标
            bbox = [
                [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
                for i, b in enumerate(bbox)
            ]
            # 生成每个字或者词的标签
            if line["label"] == "other":
                label = ["O"] * len(bbox)
            else:
                label = [f"I-{line['label'].upper()}"] * len(bbox)
                label[0] = f"B-{line['label'].upper()}"
            tokenized_inputs.update({"bbox": bbox, "labels": label})
            if label[0] != "O":
                entity_id_to_index_map[line["id"]] = len(entities)
                entities.append(
                    {
                        "start": len(tokenized_doc["input_ids"]),
                        "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                        "label": line["label"].upper(),
                    }
                )
            for i in tokenized_doc:
                tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]
        relations = list(set(relations))
        relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
        kvrelations = []
        for rel in relations:
            pair = [id2label[rel[0]], id2label[rel[1]]]
            if pair == ["question", "answer"]:
                kvrelations.append(
                    {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                )
            elif pair == ["answer", "question"]:
                kvrelations.append(
                    {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                )
            else:
                continue

        def get_relation_span(rel):
            bound = []
            for entity_index in [rel["head"], rel["tail"]]:
                bound.append(entities[entity_index]["start"])
                bound.append(entities[entity_index]["end"])
            return min(bound), max(bound)

        relations = sorted(
            [
                {
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "start_index": get_relation_span(rel)[0],
                    "end_index": get_relation_span(rel)[1],
                }
                for rel in kvrelations
            ],
            key=lambda x: x["head"],
        )
        chunk_size = 512
        for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
            item = {}
            for k in tokenized_doc:
                item[k] = tokenized_doc[k][index: index + chunk_size]
            entities_in_this_span = []  # 把相同的实体合并在一起?
            global_to_local_map = {}
            for entity_id, entity in enumerate(entities):
                if (
                        index <= entity["start"] < index + chunk_size
                        and index <= entity["end"] < index + chunk_size
                ):
                    entity["start"] = entity["start"] - index
                    entity["end"] = entity["end"] - index
                    global_to_local_map[entity_id] = len(entities_in_this_span)
                    entities_in_this_span.append(entity)
            relations_in_this_span = []
            for relation in relations:
                if (
                        index <= relation["start_index"] < index + chunk_size
                        and index <= relation["end_index"] < index + chunk_size
                ):
                    relations_in_this_span.append(
                        {
                            "head": global_to_local_map[relation["head"]],
                            "tail": global_to_local_map[relation["tail"]],
                            "start_index": relation["start_index"] - index,
                            "end_index": relation["end_index"] - index,
                        }
                    )
            item.update(
                {
                    "id": f"{doc['id']}_{chunk_id}",
                    "image": image,
                    "original_image": original_image,
                    "entities": entities_in_this_span,
                    "relations": relations_in_this_span,
                }
            )
            yield f"{doc['id']}_{chunk_id}", item


# 关系抽取所需的特征
ds_features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "input_ids": datasets.Sequence(datasets.Value("int64")),
        "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
        "labels": datasets.Sequence(
            datasets.ClassLabel(
                names=["O", "B-QUESTION", "B-ANSWER", "B-HEADER", "I-ANSWER", "I-QUESTION", "I-HEADER"]
            )
        ),
        "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
        "original_image": datasets.features.Image(),
        "entities": datasets.Sequence(
            {
                "start": datasets.Value("int64"),
                "end": datasets.Value("int64"),
                "label": datasets.ClassLabel(names=["HEADER", "QUESTION", "ANSWER"]),
            }
        ),
        "relations": datasets.Sequence(
            {
                "head": datasets.Value("int64"),
                "tail": datasets.Value("int64"),
                "start_index": datasets.Value("int64"),
                "end_index": datasets.Value("int64"),
            }
        ),
    }
)


def build_data(image_path, annotation_path, output_path):
    writer = ArrowWriter(features=ds_features,
                         path=output_path,
                         # hash_salt="zh"
                         )
    it = generate_example_relation_extraction(image_path, annotation_path)
    try:
        for key, record in it:
            example = ds_features.encode_example(record)
            writer.write(example, key)
    finally:
        num_examples, num_bytes = writer.finalize()
        writer.close()


# # 生成中间格式文件
# build_data(os.path.join(DP, "zh.train"),
#            os.path.join(DP, "zh.train.json"),
#            os.path.join(DP, "zh.train.arrow"))
#
build_data(os.path.join(DP, "zh.val"),
           os.path.join(DP, "zh.val.json"),
           os.path.join(DP, "zh.val.arrow"))

train_dataset = Dataset.from_file(os.path.join(DP, "zh.train.arrow"))
test_dataset = Dataset.from_file(os.path.join(DP, "zh.val.arrow"))
ds = DatasetDict({"train": train_dataset, "test": test_dataset})

print("dataset:", ds)

# dataset = load_dataset("nielsr/XFUN", "xfun.fr")

print("preprocessing end")

example = ds['train'][0]
print(example.keys())

for id, box in zip(example['input_ids'], example['bbox']):
    print(tokenizer.decode([id]), box)

image = example['original_image']
width, height = image.size
image.resize((int(width * 0.3), int(height * 0.3)))

# image.show()

entities = example['entities']

id2label = {0: "HEADER", 1: "QUESTION", 2: "ANSWER"}

for start, end, label in zip(entities['start'], entities['end'], entities['label']):
    print(start, end, id2label[label])

entities = example['entities']
entities_names = []
entities_with_boxes = []
for start, end, label in zip(entities['start'], entities['end'], entities['label']):
    print(tokenizer.decode(example['input_ids'][start:end]), id2label[label])
    # entity can consist of multiple boxes
    for box in example['bbox'][start:end]:
        entities_with_boxes.append((box, label))
    entities_names.append(tokenizer.decode(example['input_ids'][start:end]))

from PIL import ImageDraw


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


draw = ImageDraw.Draw(image)

width, height = image.size

label2color = {'HEADER': 'blue', 'QUESTION': 'green', 'ANSWER': 'orange'}

for box, label_id in entities_with_boxes:
    predicted_label = id2label[label_id]
    box = unnormalize_box(box, width, height)
    draw.rectangle(box, outline=label2color[predicted_label], width=2)
    draw.text((box[0], box[1]), text=predicted_label, fill=label2color[predicted_label], width=2)

# image.show()

relations = []
for head, tail in zip(example['relations']['head'], example['relations']['tail']):
    print(f"Question: {entities_names[head]}", f"Answer: {entities_names[tail]}")


from utils.evaluation import re_score