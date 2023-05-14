import re
import json
import numpy as np
from transformers import PreTrainedTokenizer
from tqdm import tqdm

class DSTDataset():

    def __init__(
        self,
        data_dir: str = None,
        ontology_dir: str = None,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512,
    ):
        self.data = json.load(open(data_dir, 'r'))
        self.ontology = json.load(open(ontology_dir, 'r'))
        self.domain_slots = list(self.ontology.keys())
        self.slot_types = ["none", "dontcare", "span", "yes", "no"]
        self.samples = list()

        for sample in tqdm(self.data, total=len(self.data)):

            prev_utterance = str()
            state = set()

            for utterance in sample["dialogue"]:

                text = preprocess(utterance["text"])

                if utterance["role"] == "user":
                    
                    current_state = set(utterance["state"])

                    self.samples.append({
                        "prev_utterance": prev_utterance,
                        "current_utterance": text,
                        "current_state": list(set(current_state) - state),
                    })

                    for cs in current_state:
                        state.add(cs)
                else:
                    prev_utterance = text

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.span_labeling_errors = list()

    def __len__(self):
        return len(self.samples)
    
    def get_vocab(self):
        return self.domain_slots, self.slot_types
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        prev_utterance = sample["prev_utterance"]
        current_utterance = sample["current_utterance"]
        current_state = sample["current_state"]

        prev_utterance_ids = self.tokenizer(prev_utterance).input_ids
        token_type_ids = [0] * len(prev_utterance_ids)

        current_utterance_ids = self.tokenizer(current_utterance).input_ids[1:]
        token_type_ids += [1] * len(current_utterance_ids)

        slot_type_labels = [self.slot_types.index("none") for _ in self.domain_slots]
        span_labels = [[0, 0] for _ in self.domain_slots]

        input_ids = prev_utterance_ids + current_utterance_ids

        for state in current_state:
            domain, slot, value = state.split("-")
            slot_idx = self.domain_slots.index("-".join([domain, slot]))

            if value in ["dontcare", "yes", "no"]:
                slot_type_labels[slot_idx] = self.slot_types.index(value)
            else:
                slot_type_labels[slot_idx] = self.slot_types.index("span")

                value_ids = self.tokenizer(value, add_special_tokens=False).input_ids
                value_len = len(value_ids)

                for i in range(min(len(input_ids), self.max_length) - value_len):
                    if input_ids[i:i+value_len] == value_ids:
                        span_labels[slot_idx][0] = i
                        span_labels[slot_idx][1] = i + value_len

        attention_mask = [1] * len(input_ids)

        gap = max(0, self.max_length - len(input_ids))

        input_ids += [self.tokenizer.pad_token_id] * gap
        attention_mask += [0] * gap
        token_type_ids += [0] * gap

        slot_type_labels = np.array(slot_type_labels)
        multi_labels = np.array(slot_type_labels != 0, dtype=int)
        
        return {
            "input_ids": np.array(input_ids[:self.max_length], dtype=int),
            "attention_mask": np.array(attention_mask[:self.max_length], dtype=int),
            "token_type_ids": np.array(token_type_ids[:self.max_length], dtype=int),
            "slot_type_labels": np.array(slot_type_labels, dtype=int),
            "span_labels": np.array(span_labels, dtype=int),
            "multi_labels": multi_labels
        }


def preprocess(string):

    days = {
        "하루": "1일",
        "이틀": "2일",
        "사흘": "3일",
        "나흘": "4일",
        "닷새": "5일",
        "엿새": "6일",
        "이레": "7일",
        "이흐레": "7일",
        "일주일": "7일",
        "여흐레": "8일",
        "아흐레": "9일",
        "열흘": "10일",
    }

    for day in days.keys():
        if day in string:
            string = string.replace(day, days[day])
    
    numerals = {
        "한": "1",
        "두": "2",
        "세": "3",
        "네": "4",
        "다섯": "5",
        "여섯": "6",
        "일곱": "7",
        "여덟": "8",
        "아홉": "9",
        "열": "10",
        "열한": "11",
        "열두": "12",
    }

    for numeral in numerals.keys():
        match = re.findall("(({})[ ]*([명분시]))".format(numeral), string=string)
        if match:
            for m in match:
                if "오후" in match:
                    string = str(int(numerals[m[1]]) + 12) + m[2]
                else:
                    string = string.replace(m[0], numerals[m[1]] + m[2])
                
    string = string.replace("혼자", "1명")

    match = re.findall("(([0-9]+)시[ ]*([0-9]+)분)", string=string)
    for m in match:
        m = list(m)
        if len(m[1]) == 1:
            m[1] = "0" + m[1]
        if len(m[2]) == 1:
            m[2] = "0" + m[2]
        string = string.replace(m[0], m[1] + " : " + m[2])

    match = re.findall("(([0-9]+)시[ 반]*)", string=string)
    for m in match:
        m = list(m)
        if len(m[1]) == 1:
            m[1] = "0" + m[1]

        if "반" in m[0]:
            string = string.replace(m[0], m[1] + " : 30")
        else:
            string = string.replace(m[0], m[1] + " : 00")

    match = re.findall("([서울 ]*([동서남북]+쪽))", string=string)
    for m in match:
        string = string.replace(m[0], " 서울 " + m[1]).replace("  ", " ").rstrip().strip()
        
    return string


def model_input_change(dataset):
    dataset_dic = {'input_ids':[], 'attention_mask':[], 'token_type_ids':[], 'slot_type_labels':[], 'start_index':[], 'end_index':[]}


    for i in tqdm(range(len(dataset)), total=len(dataset)):
        dataset_dic['input_ids'].append(np.array(dataset[i]['input_ids']))
        dataset_dic['attention_mask'].append(np.array(dataset[i]['attention_mask']))
        dataset_dic['token_type_ids'].append(np.array(dataset[i]['token_type_ids']))
        dataset_dic['slot_type_labels'].append(np.array(dataset[i]['slot_type_labels']))
        start_etc = []
        end_etc = []
        for start, end in dataset[i]['span_labels']:
            if start == 0 or end == 0:
                start = -100
                end = -100
            start_etc.append(start)
            end_etc.append(end)
        dataset_dic['start_index'].append(np.array(start_etc, dtype=int))
        dataset_dic['end_index'].append(np.array(end_etc, dtype=int))

    input_ids = np.array(dataset_dic['input_ids'], dtype=int)
    token_type_ids = np.array(dataset_dic['token_type_ids'], dtype=int)
    attention_mask = np.array(dataset_dic['attention_mask'], dtype=int)
    slot_type_labels = np.array(dataset_dic['slot_type_labels'], dtype=int)
    start_index = np.array(dataset_dic['start_index'], dtype=int)
    end_index = np.array(dataset_dic['end_index'], dtype=int)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids':token_type_ids, 'slot_type_labels':slot_type_labels, 'start_index':start_index, 'end_index':end_index}

def model_output_change(idx_to_slot_type, idx_to_slot_key, dev_inputs, slot_type_pred, start_pred, end_pred, test_dir, tokenizer):
    slot_type_straing = []
    slot_key_string = []
    slot_value_string = []

    for i in range(len(slot_type_pred)):
        slot_type_line = []
        slot_key_line = []
        slot_value_line = []
        for j in range(len(slot_type_pred[i])):
            if (np.argmax(slot_type_pred[i][j])) != 0: # None이 아니면
                slot_type_line.append(idx_to_slot_type[np.argmax(slot_type_pred[i][j])])
                slot_key_line.append(idx_to_slot_key[j])
                start_idx_pred = np.argmax(start_pred[i][j])
                end_idx_pred = np.argmax(end_pred[i][j])
                slot_value_line.append(tokenizer.decode(dev_inputs['input_ids'][i][start_idx_pred:end_idx_pred]))

        slot_type_straing.append(slot_type_line)
        slot_key_string.append(slot_key_line)
        slot_value_string.append(slot_value_line)

    predict_key_value = []
    for t, k, v in zip(slot_type_straing, slot_key_string, slot_value_string):
        predict_k_v = []
        for i in range(len(t)):
            if t[i] == 'span':
                predict_k_v.append(k[i] + '-' + v[i].replace(' ', '', 100))
            else:
                predict_k_v.append(k[i] + '-' + t[i])
        predict_key_value.append(predict_k_v)

    with open(test_dir, 'r') as file:
        dev = json.load(file)

    true_states = []

    for doc in dev:
        texts = doc['dialogue']
        true_state = []
        for idx in range(len(texts)//2):
            slots = doc['dialogue'][idx*2]['state']
            state = {}
            for slot in slots:
                key = '-'.join(slot.split('-')[:2])
                value = slot.split('-')[-1]
                state[key] = value.replace(' ', '', 100)
            true_state.append(state)
        true_states.append(true_state)

    dialogue_unit_state_length = [len(i) for i in true_states]

    pred_states = []

    dialogue_unit_predict = []
    c = 0
    for i in dialogue_unit_state_length:
        dialogue_unit_predict.append(predict_key_value[c:c+i])
        c += i

    for idx, dialog in enumerate(dialogue_unit_predict):
        state_dict = {}
        result = []
        for utterance in dialog:
            for state in utterance:
                try:
                    domain, slot, value = state.split("-")
                except:
                    print(state, idx)
                state_dict[domain + '-' + slot] = value
            result.append(state_dict.copy())
        pred_states.append(result)

    return true_states, pred_states