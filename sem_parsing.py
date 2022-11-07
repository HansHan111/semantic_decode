import argparse
import random
import numpy as np
from data import *
from models import *
from utils import *
from typing import List

from transformers import pipeline, AutoTokenizer, BertConfig, BertModel, BartConfig, BartForConditionalGeneration, BartModel, TrainingArguments, Trainer


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='sem_parser.py')

    # General system running and configuration options
    parser.add_argument('--decode_type', type=str, default='BASIC',
                        help='what type of decoding to use (BASIC, ORACLE, or FANCY)')

    parser.add_argument('--train_path', type=str,
                        default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str,
                        default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str,
                        default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str,
                        default='geo_test_output.tsv', help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo',
                        help='domain (geo for geoquery)')
    parser.add_argument('--no_java_eval', dest='perform_java_eval', default=True,
                        action='store_false', help='run evaluation of constructed query against java backend')
    parser.add_argument('--quiet', dest='quiet', default=False,
                        action='store_true', help="Suppress dataset printing")
    parser.add_argument('--eval_from_checkpoint', default=False,
                        action='store_true', help="Evaluate model from checkpoint")
    parser.add_argument('--model_save_path', type=str,
                        default='./models/bart-summarizer/', help='path to save models (directory)')
    parser.add_argument('--model_load_path', type=str, default='./models/bart-summarizer/checkpoint-1500',
                        help='path to load model from (specific checkpoint)')

    parser.add_argument('--epochs', type=int, default=30,
                        help='epochs for training')
    # Dev performance:
    # Recall: 3249/4028 = 0.8066037735849056
    # Exact Match: 48/120 = 0.4

    parser.add_argument('--decoder_len_limit', type=int,
                        default=65, help='output length limit of the decoder')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    # Load the training and test data using the same indexer for both
    train, dev, test = load_datasets(
        args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, indexer = index_datasets_one_indexer(
        train, dev, test, args.decoder_len_limit)

    # Print some examples from the dataset
    print("%i train exs, %i dev exs, %i input types, %i output types" % (
        len(train_data_indexed), len(dev_data_indexed), len(indexer), len(indexer)))
    print("Max input length: " + repr(max([len(ex.x_indexed)
          for ex in train_data_indexed + dev_data_indexed + test_data_indexed])))
    if not args.quiet:
        print("Indexer (%i tokens): %s" % (len(indexer), indexer))
        print("Here are some examples post tokenization and indexing:")
        for i in range(0, min(len(train_data_indexed), 10)):
            print(train_data_indexed[i])

    # We create train and validation datasets that will be used
    # to train and evaluate the model during training
    train_dataset = convert_to_hf_dataset(train_data_indexed, quiet=args.quiet)
    dev_dataset = convert_to_hf_dataset(dev_data_indexed, quiet=args.quiet)

    if args.eval_from_checkpoint:
        config, _ = initialize_seq2seq_model(len(indexer))
        model = BartForConditionalGeneration.from_pretrained(
            args.model_load_path)
        model = model.to("cpu")
    else:
        config, model = initialize_seq2seq_model(len(indexer))
        # just to make sure no crashes
        decode_basic(model, indexer, train_dataset, num_exs=10)
        train_seq2seq_model(model, train_dataset, dev_dataset, args)
        # if training on the GPU, moves to the CPU for evaluation
        model = model.to("cpu")

    if args.decode_type == "BASIC":
        # Removes the labels to ensure that you don't simply use the oracle method here
        preds = decode_basic(
            model, indexer, dev_dataset.copy_no_label(), num_exs=-1)
        score_decoded_outputs(preds, indexer, dev_dataset)
    elif args.decode_type == "ORACLE":
        preds = decode_oracle(model, indexer, dev_dataset, num_exs=-1)
        score_decoded_outputs(preds, indexer, dev_dataset)
    elif args.decode_type == "FANCY":
        # Removes the labels to ensure that you don't simply use the oracle method here
        preds = decode_fancy(
            model, indexer, dev_dataset.copy_no_label(), num_exs=-1)
        score_decoded_outputs(preds, indexer, dev_dataset)
