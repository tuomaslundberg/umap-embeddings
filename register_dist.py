import sys
import pandas as pd
import os

source = sys.argv[1]
dest = sys.argv[2]

df = pd.read_csv(source, sep='\t')
df['doc_length'] = df['text'].str.split().str.len()
#aggr = df.groupby('preds')['doc_length'].agg(['count', 'median', 'std'])
aggr = df['doc_length'].agg(['count', 'median', 'std'])

aggr.to_csv(f'{dest}/totals_{os.path.basename(source)}', sep='\t')
