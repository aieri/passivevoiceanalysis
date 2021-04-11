#!/usr/bin/env python3

import argparse
import spacy
import pandas as pd
from spacy.matcher import Matcher


def main(dataset, output, model):
    '''Analyze a csv dataset and add n_active/n_passive columns'''
    df = pd.read_csv(dataset, index_col='Response#')
    analyzer = Analyzer(model)
    df['n_sentences'] = df['Description'].apply(analyzer.count_sentences)
    df['n_passive'] = df['Description'].apply(analyzer.count_passive)
    df['n_active'] = df.n_sentences - df.n_passive
    print(df)
    if output:
        df.to_csv(output)

class Analyzer():
    '''A passive voice analyzer class'''
    def __init__(self, model):
        self.nlp = spacy.load(model)
        passive_rule = [{'DEP': 'nsubjpass'},
                        {'DEP': 'aux', 'OP': '*'},
                        {'DEP': 'auxpass'},
                        {'TAG': 'VBN'}]
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add('Passive', [passive_rule])

    def count_passive(self, text):
        doc = self.nlp(text)
        return len(list(filter(self.matcher, doc.sents)))

    def count_sentences(self, text):
        doc = self.nlp(text)
        return len(list(doc.sents))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output')
    parser.add_argument('--model', default='en_core_web_sm')
    args = parser.parse_args()
    main(args.dataset, args.output, args.model)
