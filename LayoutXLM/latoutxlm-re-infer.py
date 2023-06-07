# layoutxlm-re推理
# @date：20221226
# #Author: DYF-AI
import os
import copy
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import datasets
from datasets import load_dataset, load_from_disk
from datasets import Dataset, DatasetDict

from utils.image_utils import *

from transformers import AutoTokenizer
from modeling_layoutxlm_re import LayoutLMv2ForRelationExtraction
from utils.visual import draw_re_results

DP = "/mnt/j/dataset/document-intelligence/XFUND/fr"
OP = "/mnt/j/dataset/document-intelligence/XFUND/output"
MP = "/mnt/j/model/pretrained-model/bert_torch"

TRAINED_MP = "./output/layoutxlm-finetuned-xfund-re/"

use_visual_backbone = False

# tokenizer = AutoTokenizer.from_pretrained(os.path.join(TRAINED_MP, "checkpoint-5000"))
tokenizer = AutoTokenizer.from_pretrained(os.path.join(MP, "xlm-roberta-base"))

f_w = open( os.path.join("./output/pred", 'infer.txt'), mode='w', encoding='utf-8')
# re训练的模型
model = LayoutLMv2ForRelationExtraction.from_pretrained(
    os.path.join(TRAINED_MP, "checkpoint-5000"), use_visual_backbone=use_visual_backbone)


def build_tokenizer_data(json_data):
    # 返回切分的模型输入
    ser_inputs = []
    entity_idx_dict_batch = []
    size = [json_data["img"]["width"], json_data["img"]["height"]]
    # 读取整张图的信息
    tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
    # 实体、关系、实体id及label, 实体id对应index
    entities, relations, id2label, entity_id_to_index_map = [], [], {}, {}
    empty_entity = set()
    # 遍历每个json的每个box(实体)
    for line in json_data['document']:
        print(line)
        if len(line['text']) == 0:
            empty_entity.add(line['id'])
            continue
        id2label[line["id"]] = line["label"]
        # 推理时可以不管relation
        relations.extend([tuple(sorted(l)) for l in line["linking"]])
        # 切词, offset_mapping: 每个词的起始结束位置
        tokenized_inputs = tokenizer(line["text"],
                                     add_special_tokens=False,
                                     return_offsets_mapping=True,
                                     return_attention_mask=False,
                                     )
        # bbox是切词后词的box
        text_length, ocr_length, bbox = 0, 0, []
        # 计算每个词的box
        for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
            if token_id == 6:
                bbox.append(None)
                continue
            text_length += offset[1] - offset[0]
            tmp_box = []
            # 这里统一json的切词和tokenizer的切词
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
        # 当box为None时,替换
        print("before:", bbox)
        bbox = [
            [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
            for i, b in enumerate(bbox)
        ]
        print("after:", bbox)
        # 生成每个字或者词的标签
        if line["label"] == "other":
            label = ["O"] * len(bbox)
        else:
            label = [f"I-{line['label'].upper()}"] * len(bbox)
            label[0] = f"B-{line['label'].upper()}"
        tokenized_inputs.update({"bbox": bbox, "labels": label})
        if label[0] != "O":
            #entity_id_to_index_map[line["id"]] = len(entities)
            entities.append(
                {
                    "start": len(tokenized_doc["input_ids"]),
                    "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                    "label": line["label"].upper(),
                    # 保留其他的实体信息
                    "transcription" : line['text'],
                    "bbox" : line["box"],
                    "id": line["id"]
                }
            )
        for i in tokenized_doc:
            tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]

    # 输入词的长度是512
    chunk_size = 512
    for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
        print("chunk_id:", chunk_id)
        if chunk_id == 2:
            print("chunk_id == 2: ", json_data["img"]["fname"])
        item = {}
        # k: input_ids, bbox, labels
        for k in tokenized_doc:
            item[k] = tokenized_doc[k][index: index + chunk_size]
        entities_in_this_span = []
        global_to_local_map = {}
        for entity_id, entity in enumerate(entities):
            # 确保实体是512以内
            if (
                    index <= entity["start"] < index + chunk_size
                    and index <= entity["end"] < index + chunk_size
            ):
                entity["start"] = entity["start"] - index
                entity["end"] = entity["end"] - index
                global_to_local_map[entity_id] = len(entities_in_this_span)
                entities_in_this_span.append(entity)
        item.update(
            {
                "id": f"{json_data['id']}_{chunk_id}",
                "entities": entities_in_this_span,
            }
        )


        ser_inputs.append(item)
        entity_idx_dict = {}

        # 记录删除O的实体id，对应的原始包含O的实体id
        for idx, entity in enumerate(entities):
            entity_idx_dict[idx] = entity["id"]

        entity_idx_dict_batch.append(entity_idx_dict)

        print("item:", item)
    return ser_inputs, entity_idx_dict_batch

# 模型输入
def make_input(input_datas, entity_idx_dict_batch, batch_size = 2):
    label2id = {"HEADER": 0, "QUESTION": 1, "ANSWER": 2}
    model_input = {}
    num = 0
    input_ids, bbox, attention_mask, entity_dict_list, relations = [], [], [], [], []
    id = []
    ser_results = []
    for index, input_data in enumerate(input_datas):
        # padding
        # 1. input_ids
        tmp_input_ids = [0] * 512
        tmp_input_ids[0:len(input_data["input_ids"])] = input_data["input_ids"]
        input_ids.append(tmp_input_ids)
        # 2. bbox
        tmp_bbox = []
        for i in range(512):
            tmp_bbox.append([0, 0, 0, 0])
        tmp_bbox[0:len(input_data["bbox"])] = input_data["bbox"]
        bbox.append(tmp_bbox)
        # 3. attention_mask
        tmp_attention_mask = [0] * 512
        tmp_attention_mask[0:len(input_data["input_ids"])] = [1] * len(input_data["input_ids"])
        attention_mask.append(tmp_attention_mask)
        # 4. entity_dict
        tmp_entity_dict = {'start': [entity["start"] for entity in input_data["entities"]],
                        'end': [entity["end"] for entity in input_data["entities"]],
                        'label': [label2id[entity["label"]] for entity in input_data["entities"]]}
        entity_dict_list.append(tmp_entity_dict)

        # ser结果：包含text,box
        ser_result = [
            {
                "transcription": entity["transcription"],
                "bbox": entity["bbox"],
                "id" : entity["id"]
            }
            for entity in input_data["entities"]
        ]
        ser_results.append(ser_result)


        relations.append({'start_index': [], 'end_index': [], 'head': [], 'tail': []})

        id.append(input_data["id"])

        if num == batch_size or len(input_datas) < batch_size:
            # 模型推理
            model_input["input_ids"] = torch.LongTensor(input_ids)
            model_input["bbox"] = torch.LongTensor(bbox)
            model_input["attention_mask"] = torch.FloatTensor(attention_mask)
            model_input["entities"] = entity_dict_list
            model_input["relations"] = relations

            with torch.no_grad():
                # inputs: ocr处理侯的输入，ids（SER和RE的inputs是一的）
                outputs = model(**model_input)
                print("output:", outputs)

                pred_relations = outputs["pred_relations"]

                print("pred_relations:",pred_relations)

                post_result = _infer(pred_relations, ser_results, entity_idx_dict_batch)

                for chunk_id_name, re_res in zip(id, post_result):
                    #re_res = post_result[0]
                    image_file_name = chunk_id_name.rsplit("_", 1)[0]
                    chunk_id_index =  chunk_id_name.rsplit("_", 1)[1]
                    image_file = os.path.join(DP, f"fr.val/{image_file_name}.jpg")
                    res_str = '{}\t{}\n'.format(
                        image_file,
                        json.dumps(
                            {
                                "ocr_info": re_res,
                            }, ensure_ascii=False))

                    # 可视化结果
                    img_res = draw_re_results(
                        image_file, re_res, font_path="./fonts/simfang.ttf")
                    # 保存可视化结果
                    img_save_path = os.path.join(
                        "./output/pred",
                        os.path.splitext(os.path.basename(image_file))[0] +
                        f"_{chunk_id_index}_ser_re.jpg")
                    cv2.imwrite(img_save_path, img_res)
                    print("save vis result to {}".format(img_save_path))

                    f_w.write(res_str)
            input_ids, bbox, attention_mask, entity_dict_list = [], [], [], []

        num += 1

def _infer(pred_relations, ser_results, entity_idx_dict_batch):
    results = []

    # id entity
    for pred_relation, ser_result, entity_idx_dict in zip(
            pred_relations, ser_results, entity_idx_dict_batch):
        ori_id_entity = {}
        for ser_res in ser_result:
            ori_id_entity[ser_res["id"]] = ser_res

        result = []
        used_tail_id = []
        for relation in pred_relation:
            if relation['tail_id'] in used_tail_id:
                continue
            used_tail_id.append(relation['tail_id'])
            # ocr_info_head = ser_result[entity_idx_dict[relation['head_id']]]
            # ocr_info_tail = ser_result[entity_idx_dict[relation['tail_id']]]
            print("*"*20)
            print("relation['head_id']:", relation['head_id'])
            print("entity_idx_dict:",entity_idx_dict)
            print("ori_id_entity:",ori_id_entity)
            ocr_info_head = ori_id_entity[entity_idx_dict[relation['head_id']]] if entity_idx_dict[relation['head_id']] in ori_id_entity else None
            ocr_info_tail = ori_id_entity[entity_idx_dict[relation['tail_id']]] if entity_idx_dict[relation['tail_id']] in ori_id_entity else None
            if (ocr_info_head == None or ocr_info_tail == None) :
                continue
            result.append((ocr_info_head, ocr_info_tail))
        results.append(result)
    print("results:", results)
    return results


documents_dict = {}
with open(os.path.join(DP, "fr.val.json"), "r", encoding="utf-8") as fi:
    json_val_data = json.load(fi)
    documents = json_val_data["documents"]
    for document in documents:
        print("document:", document)
        documents_dict[document["id"]] = document


for key in documents_dict.keys():
    print("key:", key)
    input_datas, entity_idx_dict_batch = build_tokenizer_data(documents_dict[key])
    print("inputs:", input_datas)
    make_input(input_datas, entity_idx_dict_batch)

#
# input_datas, entity_idx_dict_batch = build_tokenizer_data(documents_dict["fr_val_32.jpg"])
# print("inputs:", input_datas)
# make_input(input_datas, entity_idx_dict_batch)