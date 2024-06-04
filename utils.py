import torch
import pandas as pd
from transformers import BertModel,BertTokenizer, BertForMaskedLM
import numpy as np

def tokenize_sequences(sequences, tokenizer):
    tokenized_sequences = []
    for sequence in sequences:
        breakpoint()
        tokens = tokenizer.tokenize(sequence)
        tokenized_sequence = ' '.join(tokens)
        tokenized_sequences.append(tokenized_sequence)
    return tokenized_sequences

def dataset_statistics (class_counts_total):
    breakpoint()
    class_counts = class_counts_total.iloc[2:]

    highest_value = class_counts.max()  # Maximum value in the entire dataframe
    min_value = class_counts.min()  # Minimum value in the entire dataframe
    average = class_counts.mean()  # Average value across all classes
    count_less_than_10 = (class_counts < 10).sum()  # Count of values less than 10
    count_less_than_5 = (class_counts < 5).sum() 
    count_less_than_2 = (class_counts < 2).sum() # Count of values less than 5


    print("Highest Value:", highest_value)
    print("Minimum Value:", min_value)
    print("Average:", average)
    print("Number of samples with less than 10:", count_less_than_10)
    print("Number of samples with less than 5:", count_less_than_5)
    print("Number of samples with less than 1:", count_less_than_2)


def joinSequences (faa_file):

    sequence_dict = {}
    with open(faa_file, 'r') as f:
        current_id = None
        current_sequence = ''
        for line in f:
            if line.startswith('>'):
                if current_id is not None:
                    sequence_dict[current_id] = current_sequence
                current_id = line.strip().split()[0][1:]
                current_sequence = ''
            else:
                current_sequence += line.strip()
        # Add last sequence after loop ends
        if current_id is not None:
            sequence_dict[current_id] = current_sequence

    return sequence_dict

def createFileSequences (df_tsv, sequence_dict, directory):

    data = []
    
    # Step 3: Correlate the information
    for index, row in df_tsv.iterrows():
        gene_id = row['gene']
        cat = row['category']
        #class_annotation = row.drop(['contig', 'gene']).astype(int).idxmax()   # Get the column name with the highest value (class)
        sequence = sequence_dict.get(gene_id, 'Sequence not found')
        #data.append([gene_id, sequence, class_annotation])
        

        for column_name, value in row.items():
            # Check if the value is '1'
            if value == 1:
                phrog = column_name

        data.append([gene_id, sequence, cat, phrog])

    df_final = pd.DataFrame(data, columns=['Gene ID','Sequence', 'Category','PHROG'])

    # Save the DataFrame to a .xlsx file
    output_file = f'{directory}/sequences_with_classes_embeds.xlsx'
    df_final.to_excel(output_file, index=False)

    print(f"Data saved to {output_file}")

    return data


def embeddingsGeneration (data, name):

    if name=="BERT":

        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if name=="ProtBERT":

        model_name = "Rostlab/prot_bert"

        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
        

    model = BertModel.from_pretrained(model_name)
    sequences = [row[1] for row in data]

    encoded_inputs = tokenizer(sequences, padding=True, truncation=True, max_length=512, return_tensors='pt')
    #encoded_inputs = tokenizer(sequences, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        embeddings = outputs.last_hidden_state 

    breakpoint()
    embeddings_np = embeddings.numpy()

    # Flatten embeddings to obtain a single vector for each sequence
    embeddings_flat = np.mean(embeddings_np, axis=1)


    return embeddings_flat
