
import pandas as pd
import os
import spacy
import stanza
#stanza.install_corenlp()
import os
import re
from pathlib import Path
from typing import List
import subprocess
from typing import Dict


from tqdm.notebook import tqdm
from stanza.server import CoreNLPClient



# Reimplement the logic to find the path where stanza_corenlp is installed.
core_nlp_path = os.getenv('CORENLP_HOME', str(Path.home() / 'stanza_corenlp'))

# A heuristic to find the right jar file
classpath = [str(p) for p in Path(core_nlp_path).iterdir() if re.match(r"stanford-corenlp-[0-9.]+\.jar", p.name)][0]
print(classpath)


def ner_prop_str(train_files: List[str], test_files: List[str], output: str) -> str:
    """Returns configuration string to train NER model"""
    train_file_str = ','.join(train_files)
    test_file_str = ','.join(test_files)
    return f"""
trainFileList = {train_file_str}
testFiles = {test_file_str}
serializeTo = {output}
map = word=0,answer=1

useClassFeature=true
useWord=true
useNGrams=true
noMidNGrams=true
maxNGramLeng=6
usePrev=true
useNext=true
useSequences=true
usePrevSequences=true
maxLeft=1
useTypeSeqs=true
useTypeSeqs2=true
useTypeySequences=true
wordShape=chris2useLC
useDisjunctive=true
"""

def write_ner_prop_file(ner_prop_file: str, train_files: List[str], test_files: List[str], output_file: str) -> None:
    with open(ner_prop_file, 'wt') as f:
        props = ner_prop_str(train_files, test_files, output_file)
        f.write(props)

def train_model(model_name, train_files: List[str], test_files: List[str], print_report=True,
                classpath=classpath) -> str:
    """Trains CRF NER Model using StanfordNLP"""
    model_file = f'{model_name}.model.ser.gz'
    ner_prop_filename = f'{model_name}.model.props'
    write_ner_prop_file(ner_prop_filename, train_files, test_files, model_file)

    result = subprocess.run(
        ['java',
         '-Xmx2g',
         '-cp', classpath,
         'edu.stanford.nlp.ie.crf.CRFClassifier',
         '-prop', ner_prop_filename],
        capture_output=True)

    # If there's an error with invocation better log the stacktrace
    if result.returncode != 0:
        print(result.stderr.decode('utf-8'))
    result.check_returncode()

    if print_report:
        print(*result.stderr.decode('utf-8').split('\n')[-11:], sep='\n')

    return model_file


def data_filename(source, split):
    return f'{source}_{split}.tsv'




#models = {}
#for source in ['ar', 'gk', 'ar_gk']:
#    print(source)
#    train_files = [data_filename(s, 'train') for s in source.split('_')]
#    test_files = [data_filename(s, 'test') for s in source.split('_')]
#    models[source] = train_model(source, train_files, test_files)
#    print()


