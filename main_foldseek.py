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
import pickle




def main():

    phroka = "False"
    ray.init()

    with open('/media/SSD5/jpuentes/EgoPOSE2/prost5_sequences_350.pkl', 'rb') as f:
        # Load the object from the pickle file
        fasta_list = pickle.load(f)

    fasta_list=[x.upper() for x in fasta_list]
    start_time = time.time()

    bio_trans = BioTransformers(backend="protbert_bfd",num_gpus=3)
    embeddings = bio_trans.compute_embeddings(fasta_list, pool_mode=('cls','mean'),batch_size=4)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken for protbert_bfd:", elapsed_time, "seconds")

    cls_emb = embeddings['cls']
    mean_emb = embeddings['mean']

    for key, value in embeddings.items():
        if isinstance(value, np.ndarray):
            embeddings[key] = value.tolist()

    breakpoint()
    with open('embeddings_ProtBFD_350_foldseek.json', 'w') as json_file:
        json.dump(embeddings, json_file)

    with h5py.File('embeddings_ProtBFD_350_foldseek.h5', 'w') as hf:
        for key, value in embeddings.items():
            hf.create_dataset(str(key), data=value)

    
    
if __name__ == "__main__":
    main()