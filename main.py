from utils import dataset_statistics, joinSequences, createFileSequences, embeddingsGeneration
from dataloader import load_data
import os
import pandas as pd
import pickle
import torch
import json
import random
import numpy as np
from biotransformers import BioTransformers
import time
import matplotlib.pyplot as plt
from plots import scatter_embeddings
import h5py
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ray

def main():

    ray.init()

    tsv_file = 'output_all_classes.tsv'
    faa_file = 'data_prot/refseq-pharokka-proteins.faa'
    directory = 'results/includeAllClasses'

    if not os.path.exists(directory):
        os.makedirs(directory)

    num = 10000


    df_tsv, class_counts = load_data(tsv_file, faa_file)

    #dataset_statistics(class_counts)

    sequence_dict = joinSequences(faa_file)

    data = createFileSequences(df_tsv, sequence_dict, directory)

    lysis_data = []
    tail_data = []
    head_packaging_data = []
    metabolism_data = []
    connector_data = []
    integration_excision_data = []
    transcription_data = []
    moron_data = []
    other_data = []


    for item in data:
        if item[2] == 'lysis':
            lysis_data.append(item)
        elif item[2] == 'tail':
            tail_data.append(item)
        elif item[2] == 'head and packaging':
            head_packaging_data.append(item)
        elif item[2] == 'DNA, RNA and nucleotide metabolism':
            metabolism_data.append(item)
        elif item[2] == 'connector':
            connector_data.append(item)
        elif item[2] == 'integration and excision':
            integration_excision_data.append(item)
        elif item[2] == 'transcription regulation':
            transcription_data.append(item)
        elif item[2] == 'moron, auxiliary metabolic gene and host takeover':
            moron_data.append(item)
        elif item[2] == 'other':
            other_data.append(item)
    
    numSamples= 500
    # Randomly sample 4,000 items from each list
    lysis_sample = random.sample(lysis_data, numSamples)
    tail_sample = random.sample(tail_data, numSamples)
    head_packaging_sample = random.sample(head_packaging_data, numSamples)
    metabolism_sample = random.sample(metabolism_data, numSamples)
    connector_sample = random.sample(connector_data, numSamples)
    integration_excision_sample = random.sample(integration_excision_data, numSamples)
    transcription_sample = random.sample(transcription_data, numSamples)
    moron_sample = random.sample(moron_data, numSamples)
    other_sample = random.sample(other_data, numSamples)

    # Concatenate the samples to get the final subsample of 12,000 items
    final_subsample = lysis_sample + tail_sample + head_packaging_sample + metabolism_sample + connector_sample + integration_excision_sample + transcription_sample + moron_sample + other_sample

    # Shuffle the final subsample
    random.shuffle(final_subsample)

    with open('complete_balanced_data_350.pkl', 'wb') as picklefile:
        pickle.dump(final_subsample, picklefile)

    #fasta_list = [sublist[1] for sublist in data]
    fasta_list = [sublist[1] for sublist in final_subsample]

    start_time = time.time()

    bio_trans = BioTransformers(backend="protbert_bfd",num_gpus=4)
    embeddings = bio_trans.compute_embeddings(fasta_list, pool_mode=('cls','mean'),batch_size=4)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken for protbert_bfd:", elapsed_time, "seconds")

    cls_emb = embeddings['cls']
    mean_emb = embeddings['mean']

    for key, value in embeddings.items():
        if isinstance(value, np.ndarray):
            embeddings[key] = value.tolist()

    with h5py.File('embeddings_ProtBFD_350_balanced_allclasses.h5', 'w') as hf:
        for key, value in embeddings.items():
            hf.create_dataset(str(key), data=value)

    
    with open('embeddings_ProtBFD_350_balanced_allclasses.json', 'w') as json_file:
        json.dump(embeddings, json_file)


if __name__ == "__main__":
    main()
