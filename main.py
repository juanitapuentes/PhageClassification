from utils import dataset_statistics, joinSequences, createFileSequences, embeddingsGeneration
from dataloader import load_data
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from plots import scatter_embeddings

def main():

    tsv_file = 'data_prot/refseq-wide-phrogn-head-packaging.tsv'
    faa_file = 'data_prot/refseq-pharokka-proteins.faa'
    directory = 'results'

    if not os.path.exists(directory):
        os.makedirs(directory)

    df_tsv, class_counts = load_data(tsv_file, faa_file)

    dataset_statistics(class_counts)

    sequence_dict = joinSequences(faa_file)

    data = createFileSequences(df_tsv, sequence_dict, directory)
    
    name = "ProtBERT"

    embeddings_flat = embeddingsGeneration(data, name)

    scatter_embeddings (embeddings_flat, name, directory)
    

if __name__ == "__main__":
    main()