import json
import csv
import re
import argparse
import os


def get_ontologies(parsed_string):
    ontologies = {}
    text = re.sub(r"\s*\[\s*", " [ ", parsed_string)
    text = re.sub(r"\s*\]\s*", " ] ", text)
    left_sq_brackets = []
    intents = []
    for token in text.strip().split():
        if token == "[":
            if len(left_sq_brackets) > 0:
                left_sq_brackets[-1] += 1
        elif token == "]":
            left_sq_brackets[-1] -= 1
            if left_sq_brackets[-1] == 0:
                left_sq_brackets.pop()
                intents.pop()
        elif token.startswith("IN:"):
            if len(left_sq_brackets) > 0:
                left_sq_brackets[-1] -= 1
            left_sq_brackets.append(1)
            intent = token[len("IN:"):]
            intents.append(intent)
            ontologies[intent] = ontologies[intent] if intent in ontologies else []
        elif token.startswith("SL:"):
            slot = token[len("SL:"):]
            if slot not in ontologies[intents[-1]]:
                ontologies[intents[-1]].append(slot)
    return ontologies


def parse_string_to_tree(input_string):
    tokens = input_string.split()
    stack = []
    for token in tokens:
        if token.startswith('['):
            intent = token[1:].split(':')[1]
            node = {'intent': intent, 'slots': []}
            if stack:
                stack[-1]['slots'].append(node)
            stack.append(node)
        elif token.startswith('SL:'):
            slot = token[3:]
            if stack:
                stack[-1]['slots'].append(slot)
        elif token == ']':
            if len(stack) > 1:
                top = stack.pop()
                if len(stack) > 0:
                    stack[-1]['slots'].append(top)
    return stack[0]


def extract_nested_slots(input_string):
    matches = []
    while True:
        match = re.search(r'\[SL:([^][]+)\s+([^][]+)\]', input_string)
        if match:
            slot_name = match.group(1)
            slot_text = match.group(2)
            matches.append((slot_name, slot_text.strip()))
            input_string = input_string[:match.start()] + input_string[match.end():]
        else:
            break
    return matches


def extract_slots(input_string):
    matches = []
    stack = []
    for i, char in enumerate(input_string):
        if char == '[':
            stack.append(i)
        elif char == ']':
            if stack:
                start = stack.pop()
                if input_string[start:start + 4] == '[SL:':
                    slot_info = input_string[start:i + 1]
                    matches.append(slot_info)

    return matches

def curate_seqtopv2(data_dir, save_dir):
    num_intents_total = {"1": 0, "2": 0, "3": 0, '4': 0, '5': 0}
    api_catalog = {}
    num_intents_train = {"1": 0, "2": 0, "3": 0, '4': 0, '5': 0}
    num_intents_eval = {"1": 0, "2": 0, "3": 0, '4': 0, '5': 0}
    num_intents_test = {"1": 0, "2": 0, "3": 0, '4': 0, '5': 0}
    for domain in ['navigation', 'alarm', 'event', 'messaging', 'music', 'reminder', 'timer', 'weather']:
        for split in ['train', 'eval', 'test']:
            json_data = []
            num_intents = {"1": 0, "2": 0, "3": 0, '4': 0, '5': 0}
            tsv_file_path = os.path.join(data_dir, f'{domain}_{split}.tsv')
            # Open and read the TSV file
            with open(tsv_file_path, "r", newline="") as file:
                # Create a CSV reader with tab as the delimiter
                reader = csv.reader(file, delimiter="\t")

                # Iterate through each row in the TSV file
                for idx, row in enumerate(reader):
                    if idx == 0:
                        continue
                    question = row[1]
                    input_string = row[2]
                    utterance_ontologies = get_ontologies(input_string)
                    ontologies = {}
                    ontologies = {
                        key: list(
                            set(
                                ontologies.get(key, [])
                                + utterance_ontologies.get(key, [])
                            )
                        )
                        for key in (ontologies.keys() | utterance_ontologies.keys())
                    }

                    matches_2 = extract_slots(input_string)
                    # Print the extracted slot information
                    slot_info = {}
                    for match in matches_2:
                        parts = match.split(' ', maxsplit=1)
                        slot_name = parts[0].replace('[', '').strip()
                        slot_text = parts[1].replace(']', '').strip()
                        if '[IN:' in slot_text:
                            pattern_intent = r'\[IN:([^]][^\s]*)'
                            matches_intent = re.findall(pattern_intent, slot_text)
                            for in_match in matches_intent:
                                slot_text = in_match
                                break
                        if slot_name in slot_info:
                            slot_info[slot_name].append(slot_text)
                        else:
                            slot_info[slot_name] = [slot_text]
                    apis_seq = []

                    for intent, slots in ontologies.items():
                        if 'UNSUPPORTED_' in intent:
                            continue
                        api_slots = {}
                        for slot in slots:
                            slt_val = slot_info['SL:' + slot][0]
                            slot_info['SL:' + slot] = slot_info['SL:' + slot][1:]
                            api_slots[slot] = slt_val
                        api_slots_arr = [f'{slot} = "{val}"' for slot, val in api_slots.items()]
                        api = f'{intent}({", ".join(api_slots_arr)})'
                        apis_seq.append((api, input_string.index('IN:' + intent)))
                        if intent not in api_catalog:
                            api_catalog[intent] = {
                                "description": "",
                                "parameters": []
                            }
                        else:
                            for slot_name, val in api_slots.items():
                                if slot_name not in api_catalog[intent]["parameters"]:
                                    api_catalog[intent]["parameters"].append(slot_name)

                    # Use re.search to find the pattern in the input string
                    if len(apis_seq) > 0:
                        ordered_seq = sorted(apis_seq, key=lambda tup: tup[1], reverse=True)
                        only_apis = []
                        for api in ordered_seq:
                            only_apis.append(api[0])
                        json_data.append(
                            {
                                "input": question,
                                'apis': only_apis,
                            }
                        )
                        num_intents[str(len(only_apis))] += 1
                        num_intents_total[str(len(only_apis))] += 1
                save_path = os.path.join(save_dir, f"{domain}_{split}.json")
                with open(save_path, 'w+') as file:
                    json.dump(json_data, file, indent=4)

                print(tsv_file_path)
                print('Num of examples: ', len(json_data))
                for key, val in num_intents.items():
                    print(f'\tNumber of examples with {key} APIs: {val}')
                print('-' * 20)
                if split == 'train':
                    for key, val in num_intents.items():
                        num_intents_train[key] += val
                elif split == 'eval':
                    for key, val in num_intents.items():
                        num_intents_eval[key] += val
                elif split == 'test':
                    for key, val in num_intents.items():
                        num_intents_test[key] += val
    print('Overall stats: ')
    print(num_intents_total)
    for key, val in num_intents_total.items():
        print(f'Number of examples with {key} APIs: {val}')
    with open(os.path.join(save_dir, 'api_spec.json'), 'w+') as file:
        json.dump(api_catalog, file, indent=4)

    print('TRAIN:')
    num_train = 0
    for key, val in num_intents_train.items():
        print(f'Number of examples with {key} APIs: {val}')
        num_train += val
    print('EVAL:')
    num_eval = 0
    for key, val in num_intents_eval.items():
        print(f'Number of examples with {key} APIs: {val}')
        num_eval += val
    print('TEST:')
    num_test = 0
    for key, val in num_intents_test.items():
        print(f'Number of examples with {key} APIs: {val}')
        num_test += val

    print('num_train: ', num_train)
    print('num_eval: ', num_eval)
    print('num_test: ', num_test)
    print('total: ', str(num_train + num_eval + num_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    curate_seqtopv2(args.data_dir, args.save_dir)



