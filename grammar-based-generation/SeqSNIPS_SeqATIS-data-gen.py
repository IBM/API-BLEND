import os, json, re
from tqdm import tqdm
import argparse
import spacy
en = spacy.load('en_core_web_sm')


def read_file(file_path):
    """ Read data file of given path.
    :param file_path: path of data file.
    :return: list of sentence, list of slot and list of intent.
    """
    texts, slots, intents = [], [], []
    text, slot = [], []
    with open(file_path, 'r', encoding="utf8") as fr:
        for line in fr.readlines():
            items = line.strip().split()
            if len(items) == 1:
                texts.append(text)
                slots.append(slot)
                if "/" not in items[0]:
                    intents.append(items)
                else:
                    new = items[0].split("/")
                    intents.append([new[1]])
                # clear buffer lists.
                text, slot = [], []
            elif len(items) == 2:
                text.append(items[0].strip())
                slot.append(items[1].strip())
    return texts, slots, intents


def split_string_on_delimiters(string, delimiters, max_splits=None):
    # Create a pattern for all delimiters
    pattern = '|'.join(map(re.escape, delimiters))

    # Split the string using the pattern and max_splits
    return re.split(pattern, string, maxsplit=max_splits)


def clause_parse(text):
    doc = en(text)
    seen = set()
    chunks = []
    for sent in doc.sents:
        heads = [cc for cc in sent.root.children if cc.dep_ == 'conj']

        for head in heads:
            words = [ww for ww in head.subtree]
            for word in words:
                seen.add(word)
            chunk = (' '.join([ww.text for ww in words]))
            chunks.append((head.i, chunk))

        unseen = [ww for ww in sent if ww not in seen]
        chunk = ' '.join([ww.text for ww in unseen])
        chunks.append((sent.root.i, chunk))

    chunks = sorted(chunks, key=lambda x: x[0])
    return chunks


def parse_IOB(tokens, tags):
    from nltk import pos_tag
    from nltk.tree import Tree
    from nltk.chunk import conlltags2tree
    # tag each token with pos
    pos_tags = [pos for token, pos in pos_tag(tokens)] #nltk.download('averaged_perceptron_tagger')
    # convert the BIO / IOB tags to tree
    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)  # parse the tree to get our original text
    original_text = []
    for subtree in ne_tree:
        # checking for 'O' tags
        if type(subtree) == Tree:
            original_label = subtree.label()
            original_string = " ".join([token for token, pos in subtree.leaves()])
            original_text.append((original_string, original_label))
    return original_text


def create_dataset(data_dir, save_dir):
    for dataset in ['ATIS', 'SNIPS']:
        print(dataset)
        num_failed_exs = 0
        for split in ['train.txt', 'dev.txt', 'test.txt']:
            print(split)
            fpath = data_dir + f'/{dataset}/{split}'
            raw_data = []
            texts, slots, intents = read_file(fpath)
            for idx in tqdm(range(len(texts))):
                sentence = ' '.join(texts[idx]).strip()
                all_intents = intents[idx][0].split('#')
                if len(all_intents) == 1:
                    clauses = [sentence]
                else:
                    clauses = split_string_on_delimiters(sentence, ['and also', 'and then', ',', 'and', 'also'], max_splits=len(all_intents))
                    if len(clauses) != len(all_intents):
                        clauses = split_string_on_delimiters(sentence, [',', 'and then'], max_splits=len(all_intents))
                    if len(clauses) != len(all_intents):
                        if len(clauses) != len(all_intents):
                            clauses = clause_parse(sentence)
                        if len(clauses) != len(all_intents):
                            new_clauses = []
                            for idx, c in enumerate(clauses):
                                id, clause = c
                                if len(new_clauses) == 0:
                                    new_clauses.append(clause)
                                elif new_clauses[-1].strip().endswith('and and'):
                                    new_clauses[-1] = new_clauses[-1].replace('and and', 'and').strip() + ' '+ clause
                                else:
                                    new_clauses.append(clause)
                            clauses = new_clauses
                if len(clauses) > len(all_intents):
                    new_clauses = []
                    for c in clauses:
                        if len(new_clauses) > 0:
                            if len(new_clauses[-1].split(' ')) < 3:
                                new_clauses[-1] += ' '+ c
                                continue
                            elif len(c.split(' ')) < 5:
                                new_clauses[-1] += ' ' + c
                                continue
                        new_clauses.append(c)
                    clauses = new_clauses
                start = 0
                apis = []
                for i in range(len(clauses)):
                    if type(clauses[i]) == tuple:
                        num, clause = clauses[i]
                    else:
                        clause = clauses[i]
                    num_words = len(clause.split(' '))
                    slots_arr = slots[idx][start:start+num_words]
                    tokens = texts[idx][start:start+num_words]
                    params = parse_IOB(tokens, slots_arr)
                    start += num_words
                    params_dic = {}
                    for (val, name) in params:
                        if name not in params_dic:
                            params_dic[name] = []
                        params_dic[name].append(val)
                    apis.append(
                        {
                            'API': all_intents[i],
                            'Parameters': params_dic
                        }
                    )
                raw_data.append({
                    'text': sentence,
                    'APIs': apis,
                })
            directory = f"{save_dir}/Seq{dataset}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            out_fname = f"{directory}/{split.replace('.txt', '')}.json"
            with open(out_fname, 'w') as file:
                json.dump(raw_data, file, indent=4)
            print('Number of examples: ', len(raw_data))
            print('num_failed_exs: ', num_failed_exs)
            print('*'*20)
        print('-'*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    create_dataset(args.data_dir, args.save_dir)