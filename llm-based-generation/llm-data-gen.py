import argparse
import json
import jsonlines
import os
import time

import requests
from dotenv import load_dotenv
from tqdm import tqdm


class GENAI:
    def __init__(self, model) -> None:
        self.model = model
        env_path = "../.env"
        load_dotenv(env_path)
        self.API_KEY = os.getenv("GENAI_KEY", None)
        self.URL = os.getenv("GENAI_API", None)

    def ask_batch(self, prompt, temperature=0.7, max_new_tokens=128, greedy=True):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.API_KEY}'
        }
        decoding_method = 'greedy' if greedy else 'sample'
        data = {
            "model_id": self.model,
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "decoding_method": decoding_method
            }
        }
        response = requests.post(self.URL, headers=headers, data=json.dumps(data))
        output_list = [x['generated_text'] for x in response.json()['results']]
        return output_list


def generate_llm_paraphrase(api_str_list, save_path, model):
    def chunk_list(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    single_api_str_list = [api.split('[SEP]') for api in api_str_list]
    single_api_str_list = [item.strip() for sublist in single_api_str_list for item in sublist]  # flatten
    genai_obj = GENAI(model=model)  # model can be any generative model
    api_to_str = {}
    chunked_list = chunk_list(single_api_str_list, 5)  # chucked api lists with batch size 5
    for idx, batch in enumerate(chunked_list):
        print(f'Processing batch {idx} out of {len(chunked_list)}')
        prompts = []
        for api in batch:
            output_string = f'intent: {api}'
            prompt = f'Convert the following intent and its parameteres into an imperative sentence. Do not copy the API or its parameters as is in the output sentence.\n\nInput:\n' + output_string + '\nOutput:\n'
            prompts.append(prompt)
        try:
            outputs = genai_obj.ask_batch(prompts)
        except:
            print('Connection error at ask_batch')
            time.sleep(5)
            outputs = genai_obj.ask_batch(prompts)
        for txt, api in zip(outputs, batch):
            api_to_str[api] = txt
        # uncomment to save llm phrases with interval of 100 batches
        # if len(api_to_str) > 0 and len(api_to_str) % 100 == 0:
        #     with open(save_path, 'w+') as file:
        #         json.dump(api_to_str, file, indent=4)
    # uncomment to save all llm phrases
    # with open(save_path, 'w+') as file:
    #     json.dump(api_to_str, file, indent=4)
    return api_to_str


def extract_raw_data(raw_data_dir):
    data_files = [item for item in os.listdir(raw_data_dir) if
                  os.path.isfile(os.path.join(raw_data_dir, item)) and not item == 'schema.json']
    processed_data = []
    for file in tqdm(data_files):
        print(file)
        data = json.load(open(os.path.join(raw_data_dir, file)))
        print(len(data))
        for d in tqdm(data):  # each dialog
            for t in d['turns']:  # each turns
                if t['speaker'] == 'USER':
                    turn_intent_slots = []
                    for f in t['frames']:
                        if f['state']['slot_values'] and not f['state']['active_intent'] == 'NONE':
                            turn_slots = []
                            for slot, values in f['state']['slot_values'].items():
                                turn_slots.append(f'{slot} = {values[0]}')
                            slot_str = ' ; '.join(turn_slots)
                            turn_intent_slots.append(f"{f['state']['active_intent']}({slot_str})")
                    api_str = ' [SEP] '.join(turn_intent_slots)
                    processed_data.append({
                        'dialog_id': d['dialogue_id'],
                        'speaker': 'USER',
                        'input': t['utterance'],
                        'output': api_str
                    })
                else:
                    processed_data.append({
                        'dialog_id': d['dialogue_id'],
                        'speaker': 'BOT',
                        'input': t['utterance'],
                        'output': ''
                    })
    return processed_data


def curate_llm_based_data(data_dir_root, save_dir, dataset_name, model):
    os.makedirs(save_dir, exist_ok=True)
    splits = ['train', 'test', 'dev']
    for split in splits:
        print(f'======= {split} =======')
        data_dir = os.path.join(data_dir_root, split)
        raw_data = extract_raw_data(data_dir)  # combine multiple dialog files

        # uncomment to save intermediate data (raw data)
        # raw_save_path = os.path.join(save_dir, f'{dataset_name}-raw-{split}.jsonl')
        # with jsonlines.open(raw_save_path, "w") as writer:
        #     writer.write_all(raw_data)

        api_str_list, api_str_dialog_map = [], {}
        for d in raw_data:
            if d['output'] and 'NONE(' not in d['output']:
                api_str_list.append(d['output'])
                api_str_dialog_map.setdefault(d['dialog_id'], []).append(d['output'])

        llm_paraphrase_save_path = os.path.join(save_dir, f'{dataset_name}-llm-{split}.jsonl')
        api_to_str = generate_llm_paraphrase(api_str_list, llm_paraphrase_save_path, model)

        # reconstruct the data with llm-paraphrases
        processed_data_dict_list = []
        for _, conv in api_str_dialog_map.items():
            input_list, output_list, api_list, intents = [], [], [], []
            for apis in conv:
                api_list.extend(apis.split('[SEP]'))
            for api in api_list:
                intent = api[:api.index('(')]
                if intent not in intents:
                    intents.append(intent)
            api_list = [max([api for api in api_list if api.startswith(intent)], key=len) for intent in
                        intents]  # take the longest string of intents (more slots).
            for api in api_list:
                if api in api_to_str.keys():
                    output_list.append(api)
                    api_str = api_to_str[api].lower()
                    api_str = api_str + '.' if not api_str.endswith('.') else api_str
                    input_list.append(api_str)
            if input_list and output_list:
                processed_data_dict_list.append(
                    {
                        'input': ' '.join(input_list),
                        'output': ' [SEP] '.join(output_list)
                    }
                )

        # save processed outputs
        processed_data_save_path = os.path.join(save_dir, f'{dataset_name}-processed-{split}.jsonl')
        with jsonlines.open(processed_data_save_path, "w") as writer:
            writer.write_all(processed_data_dict_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    curate_llm_based_data(args.data_dir, args.save_dir, args.dataset_name, args.model)
