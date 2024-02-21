import pandas as pd
import matplotlib.pyplot as plt
from plots import plot_data

def load_data (tsv_file, faa_file):

    df_tsv = pd.read_csv(tsv_file, sep='\t')

    faa_file = 'data_prot/refseq-pharokka-proteins.faa'

    columns_to_exclude = ['contig', 'gene']
    class_counts = df_tsv.drop(columns_to_exclude, axis=1).sum()

    plot_data(class_counts, 'results')

    return df_tsv, class_counts



