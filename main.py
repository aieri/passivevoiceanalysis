#!/usr/bin/env python3

import argparse
import spacy
import pandas as pd
from spacy.matcher import Matcher


def main(args):
    '''Analyze a csv dataset and add n_active/n_passive columns'''
    df = pd.read_excel(args.dataset, index_col=args.index_column)
    analyzer = Analyzer(args.model)
    df['n_sentences'] = df[args.text_column].apply(analyzer.count_sentences)
    df['n_passive'] = df[args.text_column].apply(analyzer.count_passive)
    df['n_active'] = df.n_sentences - df.n_passive
    print(df)
    if args.output:
        df.to_excel(args.output)


class Analyzer():
    '''A passive voice analyzer class'''
    def __init__(self, model):
        self.nlp = spacy.load(model)
        # Borrowed straight out of
        # https://github.com/armsp/active_or_passive.git
        passive_rule = [{'DEP': 'nsubjpass'},
                        {'DEP': 'aux', 'OP': '*'},
                        {'DEP': 'auxpass'},
                        {'TAG': 'VBN'}]
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add('Passive', [passive_rule])

    def count_passive(self, text):
        '''Return how many passive voice sentences are present in the text'''
        doc = self.nlp(text)
        return len(list(filter(self.matcher, doc.sents)))

    def count_sentences(self, text):
        '''Return how many sentences are present in the text'''
        doc = self.nlp(text)
        return len(list(doc.sents))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output')
    parser.add_argument('--model', default='en_core_web_trf')
    parser.add_argument('--index-column', default='ResponseId')
    parser.add_argument('--text-column', default='Text')
    args = parser.parse_args()
    main(args)
