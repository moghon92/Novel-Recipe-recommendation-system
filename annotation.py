from stanza.server import CoreNLPClient
from dataclasses import dataclass, asdict
from typing import List
import pandas as pd
import json
from fractions import Fraction
import regex as re
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

@dataclass
class NERData:
    ner: List[str]
    tokens: List[str]

    # Let's use Pandas to make it pretty in a notebook
    def _repr_html_(self):
        return pd.DataFrame(asdict(self)).T._repr_html_()


def annotate_ner(ner_model_file: str, texts, df, tokenize_whitespace: bool = True):
    properties = {"ner.model": ner_model_file, "tokenize.whitespace": tokenize_whitespace,
                  "ner.applyNumericClassifiers": False}

    # annotated = {'ingrs':[], 'qty':[], 'units':[]}
    temp = df.copy()
    with CoreNLPClient(
            annotators=['tokenize', 'ssplit', 'ner'],
            properties=properties,
            timeout=30000,
            be_quiet=True,
            memory='6G') as client:
        for k,v in texts.items():

            for sent in v.split('. '):
                print(sent)
                ing = []
                qty = []
                unit = []
                annot = client.annotate(sent)

                for sentence in annot.sentence:
                    for t in sentence.token:
                        if t.coarseNER == 'NAME':
                            ing.append(t.word)
                        elif t.coarseNER == 'QUANTITY':
                            qty.append(t.word)
                        elif t.coarseNER == 'UNIT':
                            unit.append(t.word)

                q = ' '.join([i for i in qty])
                f = float(sum(Fraction(s) for s in q.split() if re.search('[a-zA-Z]+', s) is None))

                i_ns = ' '.join([i for i in ing])
                n = re.sub('[^a-zA-Z ]', '', i_ns.split(',')[0]).strip().lower()
                print(f'{n}:NAME\n{f}:QUANTITY\n{unit[0]}:UNIT\n')
                temp.loc[k, n] = f




    return temp


def extract_ner_data(annotation):
    tokens = [token for sentence in annotation.sentence for token in sentence.token]
    return ([t.word for t in tokens], [t.coarseNER for t in tokens])


def annotate(df):
    items = df.set_index('RecipeID')['Ingredients'].to_dict()
    annotations = annotate_ner('ar.model.ser.gz', items, df.set_index('RecipeID'))

    return annotations






df = pd.read_csv('new_recipes.csv')
annotated = annotate(df.iloc[:2,:])
print(annotated)
#annotated.to_csv('output.csv')

#with open('convert.txt', 'w') as convert_file:
#    convert_file.write(json.dumps(annotated))