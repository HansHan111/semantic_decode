from utils import *
from typing import List, Tuple
import random
from collections import Counter
import torch
import torch.utils.data as data

class GeoqueryDataset(data.Dataset):
    """
    Wraps the dataset dicts so that HuggingFace can use them
    """
    def __init__(self, inputs, labels, data_dict):
        self.inputs = inputs
        self.labels = labels
        self.data_dict = data_dict

    def copy_no_label(self):
        new_labels = [[-100] * len(labels) for labels in self.data_dict['labels']]
        new_data_dict = {
            'input_ids': self.data_dict['input_ids'],
            'attention_mask': self.data_dict['attention_mask'],
            'labels': new_labels
        }
        return GeoqueryDataset(self.inputs, new_labels, new_data_dict)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data_dict['input_ids'][idx],
            'attention_mask': self.data_dict['attention_mask'][idx],
            'labels': self.data_dict['labels'][idx]
        }

    def __len__(self):
        return len(self.labels)

    def __dict__(self):
        return self.data_dict


def convert_to_hf_dataset(data_indexed, inp_pad_length=23, out_pad_length=65, quiet=False):
    """
    Converts the raw loaded data into a format appropriate for Huggingface
    :param data_indexed: the loaded Example objects
    :param inp_pad_length: length to pad inputs to (23 is the longest)
    :param out_pad_length: length to pad outputs to (65 is the longest)
    :param quiet: True if we should suppress output, false if we print an example
    :return:
    """
    # input_ids, token_type_ids, attention_mask   https://huggingface.co/docs/transformers/preprocessing#build-tensors
    inputs = [ex.x_indexed + [0] * (inp_pad_length - len(ex.x_indexed)) for ex in data_indexed]
    # -100 is the token to be ignored for BART outputs
    # https://huggingface.co/transformers/v4.0.1/model_doc/bart.html#transformers.models.bart.modeling_bart._prepare_bart_decoder_inputs
    labels = [ex.y_indexed + [-100] * (out_pad_length - len(ex.y_indexed)) for ex in data_indexed]
    # 1 is index of SOS
    attention_mask = [[1] * len(ex.x_indexed) + [0] * (inp_pad_length - len(ex.x_indexed)) for ex in data_indexed]
    encodings = {
        'input_ids': inputs,
        'attention_mask': attention_mask,
        'labels': labels
    }
    if not quiet:
        print("Here are some examples of the dataset as Huggingface will see it:")
        print("Inputs: " + repr(len(inputs)) + "x" + repr(len(inputs[0])) + " " + repr(inputs))
        print("Attention mask: " + repr(len(attention_mask)) + "x" + repr(len(attention_mask[0])) + " " +repr(attention_mask))
        print("Labels: " + repr(len(labels)) + "x" + repr(len(labels[0])) + " " + repr(labels))
    return GeoqueryDataset(encodings['input_ids'], encodings['labels'], encodings)


class Example(object):
    """
    Wrapper class for a single (natural language, logical form) input/output (x/y) pair
    Attributes:
        x: the natural language as one string
        x_tok: tokenized natural language as a list of strings
        x_indexed: indexed tokens, a list of ints
        y: the raw logical form as a string
        y_tok: tokenized logical form, a list of strings
        y_indexed: indexed logical form, a list of ints
    """
    def __init__(self, x: str, x_tok: List[str], x_indexed: List[int], y, y_tok, y_indexed):
        self.x = x
        self.x_tok = x_tok
        self.x_indexed = x_indexed
        self.y = y
        self.y_tok = y_tok
        self.y_indexed = y_indexed

    def __repr__(self):
        return " ".join(self.x_tok) + " => " + " ".join(self.y_tok) + "\n   indexed as: " + repr(self.x_indexed) + " => " + repr(self.y_indexed)

    def __str__(self):
        return self.__repr__()


PAD_SYMBOL = "<PAD>"
UNK_SYMBOL = "<UNK>"
SOS_SYMBOL = "<SOS>"
EOS_SYMBOL = "<EOS>"


def load_datasets(train_path: str, dev_path: str, test_path: str, domain=None) -> (List[Tuple[str,str]], List[Tuple[str,str]], List[Tuple[str,str]]):
    """
    Reads the training, dev, and test data from the corresponding files.
    :param train_path:
    :param dev_path:
    :param test_path:
    :param domain: Ignore this parameter
    :return:
    """
    train_raw = load_dataset(train_path, domain=domain)
    dev_raw = load_dataset(dev_path, domain=domain)
    test_raw = load_dataset(test_path, domain=domain)
    return train_raw, dev_raw, test_raw


def load_dataset(filename: str, domain="geo") -> List[Tuple[str,str]]:
    """
    Reads a dataset in from the given file.
    :param filename:
    :param domain: Ignore this parameter
    :return: a list of untokenized, unindexed (natural language, logical form) pairs
    """
    dataset = []
    with open(filename) as f:
        for line in f:
            x, y = line.rstrip('\n').split('\t')
            # Geoquery features some additional preprocessing of the logical form
            if domain == "geo":
                y = geoquery_preprocess_lf(y)
            dataset.append((x, y))
    print("Loaded %i exs from file %s" % (len(dataset), filename))
    return dataset


def tokenize(x) -> List[str]:
    """
    :param x: string to tokenize
    :return: x tokenized with whitespace tokenization
    """
    return x.split()


def index(x_tok: List[str], indexer: Indexer) -> List[int]:
    return [indexer.index_of(xi) if indexer.index_of(xi) >= 0 else indexer.index_of(UNK_SYMBOL) for xi in x_tok]


def index_data(data, input_indexer: Indexer, output_indexer: Indexer, example_len_limit):
    """
    Indexes the given data
    :param data:
    :param input_indexer:
    :param output_indexer:
    :param example_len_limit:
    :return:
    """
    data_indexed = []
    for (x, y) in data:
        x_tok = tokenize(x)
        y_tok = tokenize(y)[0:example_len_limit]
        data_indexed.append(Example(x, x_tok, index(x_tok, input_indexer), y, y_tok,
                                          index(y_tok, output_indexer) + [output_indexer.index_of(EOS_SYMBOL)]))
    return data_indexed


def index_datasets_one_indexer(train_data, dev_data, test_data, example_len_limit, unk_threshold=0.0) -> (List[Example], List[Example], List[Example], Indexer):
    """
    Indexes train and test datasets where all words occurring less than or equal to unk_threshold times are
    replaced by UNK tokens.
    :param train_data:
    :param dev_data:
    :param test_data:
    :param example_len_limit:
    :param unk_threshold: threshold below which words are replaced with unks. If 0.0, the model doesn't see any
    UNKs at train time
    :return:
    """
    input_word_counts = Counter()
    # Count words and build the indexers
    for (x, y) in train_data:
        for word in tokenize(x):
            input_word_counts[word] += 1.0
    indexer = Indexer()
    # Reserve 0 for the pad symbol for convenience
    indexer.add_and_get_index(PAD_SYMBOL)
    indexer.add_and_get_index(UNK_SYMBOL)
    indexer.add_and_get_index(PAD_SYMBOL)
    indexer.add_and_get_index(SOS_SYMBOL)
    indexer.add_and_get_index(EOS_SYMBOL)
    # Index all input words above the UNK threshold
    for word in input_word_counts.keys():
        if input_word_counts[word] > unk_threshold + 0.5:
            indexer.add_and_get_index(word)
    # Index all output tokens in train
    for (x, y) in train_data:
        for y_tok in tokenize(y):
            indexer.add_and_get_index(y_tok)
    # Index things
    train_data_indexed = index_data(train_data, indexer, indexer, example_len_limit)
    dev_data_indexed = index_data(dev_data, indexer, indexer, example_len_limit)
    test_data_indexed = index_data(test_data, indexer, indexer, example_len_limit)
    return train_data_indexed, dev_data_indexed, test_data_indexed, indexer


##################################################
# YOU SHOULD NOT NEED TO LOOK AT THESE FUNCTIONS #
##################################################
def print_evaluation_results(test_data, selected_derivs, denotation_correct, example_freq=50, print_output=True):
    """
    Prints output and accuracy. YOU SHOULD NOT NEED TO CALL THIS DIRECTLY
    :param test_data:
    :param selected_derivs:
    :param denotation_correct:
    :param example_freq: How often to print output
    :param print_output: True if we should print the scores, false otherwise (you should never need to set this False)
    :return: List[float] which is [exact matches, token level accuracy, denotation matches]
    """
    num_exact_match = 0
    num_tokens_correct = 0
    num_denotation_match = 0
    total_tokens = 0
    for i, ex in enumerate(test_data):
        pred_y_toks = selected_derivs[i].y_toks if i < len(selected_derivs) else [""]
        if print_output and i % example_freq == example_freq - 1:
            print('Example %d' % i)
            print('  x      = "%s"' % ex.x)
            print('  y_tok  = "%s"' % ex.y_tok)
            print('  y_pred = "%s"' % pred_y_toks)
        # Compute accuracy metrics
        y_pred = ' '.join(pred_y_toks)
        # Check exact match
        if y_pred == ' '.join(ex.y_tok):
            num_exact_match += 1
        # Check position-by-position token correctness
        num_tokens_correct += sum(a == b for a, b in zip(pred_y_toks, ex.y_tok))
        total_tokens += len(ex.y_tok)
        # Check correctness of the denotation
        if denotation_correct[i]:
            num_denotation_match += 1
    if print_output:
        print("Exact logical form matches: %s" % (render_ratio(num_exact_match, len(test_data))))
        print("Token-level accuracy: %s" % (render_ratio(num_tokens_correct, total_tokens)))
        print("Denotation matches: %s" % (render_ratio(num_denotation_match, len(test_data))))
    return [num_exact_match / len(test_data), num_tokens_correct / total_tokens, num_denotation_match / len(test_data)]


def render_ratio(numer, denom):
    return "%i / %i = %.3f" % (numer, denom, float(numer) / denom)


def geoquery_preprocess_lf(lf):
    """
    Geoquery preprocessing adapted from Jia and Liang. Standardizes variable names with De Brujin indices -- just a
    smarter way of indexing variables in statements to make parsing easier.
    :param lf:
    :return:
    """
    cur_vars = []
    toks = lf.split(' ')
    new_toks = []
    for w in toks:
        if w.isalpha() and len(w) == 1:
            if w in cur_vars:
                ind_from_end = len(cur_vars) - cur_vars.index(w) - 1
                new_toks.append('V%d' % ind_from_end)
            else:
                cur_vars.append(w)
                new_toks.append('NV')
        else:
            new_toks.append(w)
    return ' '.join(new_toks)
