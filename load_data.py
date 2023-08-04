from collections import Counter
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Callable
import json
import os
import random
import spacy

data_dir = 'data/'
CT_json_dir = 'CT json/'

entail_label_set = ['Entailment', 'Contradiction']
entail_label_map = {key: val for val, key in enumerate(entail_label_set)}

med7 = spacy.load("en_core_med7_trf")
global total_entity_tagged
total_entity_tagged = Counter()

global MAX_LEN
MAX_LEN = 512


def tag_text(statement):
    """
    Sequence labelling
    """
    offset = 0
    doc = med7(statement)
    for ent in doc.ents:
        # text[tagged_label], e.g. oral[ROUTE] capecitabine[DRUG]
        statement = f'{statement[:ent.start_char + offset]}{statement[ent.start_char + offset:ent.end_char + offset]}' \
                    f'[{ent.label_}]{statement[ent.end_char + offset:]}'
        offset += (len(ent.label_) + 2)
        total_entity_tagged[ent.label_] += 1
    return statement


def get_evidence_list(ctr_id, section_id):
    """
    Returns the evidence list given the param CT and its section id.
    """
    with open(os.path.join(data_dir, CT_json_dir, f'{ctr_id}.json')) as f:
        js_obj = json.load(f)
        yield js_obj[section_id]


def get_evidence(json_obj, _id_num):
    """
    Used for task 1, returns a string of concatenated evidences.
    """
    evidence_list = [*get_evidence_list(json_obj[f'{_id_num}_id'], json_obj['Section_id'])][0]
    evidence_list = [evidence_list[idx] for idx in json_obj[f'{_id_num}_evidence_index']]
    return ' '.join(evidence_list)


def get_full_evidence(json_obj, _id_num, statement):
    """
    Used for task 2, returns two lists, one of evidence, and one for non-evidence.
    """
    full_evidence_list = [*get_evidence_list(json_obj[f'{_id_num}_id'], json_obj['Section_id'])][0]
    # concat statement & evidence
    full_evidence_list = [f"Statement: {statement} Evidence: {evidence}" for evidence in full_evidence_list]
    evidence_list = [full_evidence_list[idx] for idx in json_obj[f'{_id_num}_evidence_index']]
    non_evidence_list = list(set(full_evidence_list) - set(evidence_list))
    return evidence_list, non_evidence_list


def preprocess_text(obj, mode, ner=False):
    """
    Processes features according to the param mode and returns the processed feature statement
    """
    statement = obj['Statement']
    section = obj['Section_id'].upper()
    first_evidence = get_evidence(obj, 'Primary')
    second_evidence = get_evidence(obj, 'Secondary') if obj['Type'] == 'Comparison' else 'N/A'
    if ner:
        statement = tag_text(statement)
        first_evidence = tag_text(first_evidence)
        second_evidence = tag_text(second_evidence)
    if mode == 0:
        # statement only
        statement = statement
    elif mode == 1:
        # concat statement & evidence
        statement = f'{statement} {first_evidence} {second_evidence}'
    elif mode == 2:
        # concat statement & section & evidence
        statement = f'{statement} {section} {first_evidence} {second_evidence}'
    elif mode == 3:
        # concat statement & section & evidence with captions
        statement = f'Statement: {statement} Section: {section} Primary_Evidence: {first_evidence} Secondary_Evidence: {second_evidence}'
    elif mode == 4:
        # concat statement & section & evidence with punc
        statement = f'#{statement}# *{section}* ${first_evidence}$ @{second_evidence}@'
    elif mode == 5:
        statement = f'{statement} Section:{section}'
    return statement


def load_data(json_l_file, mode, ner):
    """
    Used in Entailment Classification(task1)
    Reads in json list files containing the dataset, returning 2 respective lists of labels, data.
    """
    data = []
    labels = []

    with open(os.path.join(data_dir, json_l_file)) as f:
        js_obj = json.load(f)
        for obj in js_obj:
            statement = preprocess_text(js_obj[obj], mode, ner)
            data.append(statement)
            label = js_obj[obj]['Label'] if 'Label' in js_obj[obj] else ''  # for test-set
            labels.append(entail_label_map.get(label, 0))  # for test-set
    if ner:
        print(total_entity_tagged)
    return labels, data


def compile_data(json_l_file, evidence_model, train=False):
    """
    Used for Evidence Classification(task2), returns 2 respective lists of balanced labels and data.
    """
    total_evidence = []
    total_labels = []
    with open(os.path.join(data_dir, json_l_file)) as f:
        js_obj = json.load(f)
        for obj in js_obj:
            evidence = []
            non_evidence = []
            entailment_label = evidence_model.inference(js_obj[obj]['Statement'] + ' ' + js_obj[obj]['Section_id'])
            statement = f"{entailment_label}, {js_obj[obj]['Statement']} {js_obj[obj]['Section_id']}"
            evidence_sections = ['Primary', 'Secondary'] if js_obj[obj]['Type'] == 'Comparison' else ['Primary']
            for ord in evidence_sections:
                evidence_list, non_evidence_list = get_full_evidence(js_obj[obj], ord, statement)
                if train:
                    # sample non-evidence in Train set
                    non_evidence_list = random.sample(non_evidence_list, int(len(evidence_list) * 1.5)) if len(
                        non_evidence_list) > int(len(evidence_list) * 1.5) else non_evidence_list
                evidence.extend(evidence_list)
                non_evidence.extend(non_evidence_list)
            # {1: 'relevant evidence', 0: 'non-relevant evidence'}
            total_labels.extend([1] * len(evidence))
            total_labels.extend([0] * len(non_evidence))
            total_evidence.extend(evidence_list)
            total_evidence.extend(non_evidence_list)
    return total_labels, total_evidence


def check_evidence(evidence):
    """
    Used in run.py, checks if an evidence ID exists
    """
    evidence = evidence.upper() if evidence.endswith('.json') else f'{evidence.upper()}.json'
    return os.path.exists(os.path.join(data_dir, CT_json_dir, evidence))


def parse_input(statement, entail_label, section, evidence1, evidence2):
    """
    Used in run.py, takes the param input and returns the task-2 inference
    """
    statement = statement.strip()
    evidence1 = evidence1 if evidence1.endswith('.json') else f'{evidence1.upper()}.json'
    evidence2 = evidence2 if evidence2.endswith('.json') or len(evidence2) == 0 else f'{evidence2.upper()}.json'
    evidence1 = os.path.join(data_dir, CT_json_dir, evidence1)
    evidence1_list, evidence2_list = None, None
    with open(evidence1) as f:
        evidence1_list = json.load(f)
        evidence1_list = evidence1_list[section]
        if len(evidence2) > 0:
            evidence2 = os.path.join(data_dir, CT_json_dir, evidence2)
            with open(evidence2) as f:
                evidence2_list = json.load(f)
                evidence2_list = evidence2_list[section]
    statement = f"{entail_label}, {statement} {section}"
    processed_evidence1_list = [f"Statement: {statement} Evidence: {evidence}" for evidence in evidence1_list]
    processed_evidence2_list = [f"Statement: {statement} Evidence: {evidence}" for evidence in
                                evidence2_list] if evidence2_list is not None else evidence2_list
    return evidence1_list, evidence2_list, processed_evidence1_list, processed_evidence2_list


def get_label_num():
    """
    Returns numbers of Entailment(task1) labels
    """
    return len(entail_label_set)


def get_inv_label():
    """
    Used in Entailment(task1) inference, returns {label idx: label name}
    """
    return {val: key for key, val in entail_label_map.items()}


def get_dataloader(
        tokenizer: Callable,
        train_filename: str,
        dev_filename: str,
        device: Callable,
        batch_size: int,
        mode: int = 2,
        evidence_model: bool = None,
        ner: bool = False,
):
    """
    Pre-processes datasets, and returns Dataloaders
    """
    if evidence_model is not None:
        train_labels, train_data = compile_data(train_filename, evidence_model, True)
        dev_labels, dev_data = compile_data(dev_filename, evidence_model)
    else:
        train_labels, train_data = load_data(train_filename, mode, ner)
        dev_labels, dev_data = load_data(dev_filename, mode, ner)

    class TextDataset(Dataset):
        def __init__(self, texts, labels=None):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            tokens = tokenizer.encode_plus(
                self.texts[index],
                add_special_tokens=True,
                max_length=MAX_LEN,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids = tokens['input_ids'].squeeze()
            attention_mask = tokens['attention_mask'].squeeze()

            if self.labels is not None:
                label = torch.tensor(self.labels[index], dtype=torch.long)
                return input_ids.to(device), attention_mask.to(device), label.to(device)
            else:
                return input_ids.to(device), attention_mask.to(device)

    train_dataset = TextDataset(train_data, labels=train_labels)
    dev_dataset = TextDataset(dev_data, labels=dev_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, dev_dataloader
