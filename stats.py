import pandas as pd
import csv

# Read the TSV file into a DataFrame
df = pd.read_csv('./data_prot/refseq-wide-all-phrogn.tsv', delimiter='\t')

# Group by the 'category' column and sum the counts for each gene
category_counts_phrog = df.groupby('category').sum()
category_counts = df.groupby('category').sum().sum(axis=1)

phrog_columns = df.columns[4:]

# Transpose the DataFrame so that phrog values are in rows
transposed_df = df.melt(id_vars=['contig', 'gene', 'annot', 'category'], value_vars=phrog_columns, var_name='phrog', value_name='count')

# Drop rows where count is 0
transposed_df = transposed_df[transposed_df['count'] > 0]

# Count the number of unique phrogs for each category
category_phrog_counts = transposed_df.groupby('category')['phrog'].nunique()

phrog_sums = transposed_df.groupby(['category', 'phrog'])['count'].sum()

# Count the number of PHROGs with less than 10 genes for "head and packaging" and for "tail"
head_and_packaging_phrog_count = ((phrog_sums.loc['head and packaging'] < 10).sum())
tail_phrog_count = ((phrog_sums.loc['tail'] < 10).sum())
lysis_phrog_count = ((phrog_sums.loc['lysis'] < 10).sum())
metabolism_phrog_count = ((phrog_sums.loc['DNA, RNA and nucleotide metabolism'] < 10).sum())
connector_phrog_count = ((phrog_sums.loc['connector'] < 10).sum())
integration_excision_phrog_count = ((phrog_sums.loc['integration and excision'] < 10).sum())
transcription_phrog_count = ((phrog_sums.loc['transcription regulation'] < 10).sum())
moron_phrog_count = ((phrog_sums.loc['moron, auxiliary metabolic gene and host takeover'] < 10).sum())
other_phrog_count = ((phrog_sums.loc['other'] < 10).sum())

#print the data
print(f"Number of PHROGs with less than 10 genes for 'head and packaging': {head_and_packaging_phrog_count}")
print(f"Number of PHROGs with less than 10 genes for 'tail': {tail_phrog_count}")
print(f"Number of PHROGs with less than 10 genes for 'lysis': {lysis_phrog_count}")
print(f"Number of PHROGs with less than 10 genes for 'DNA, RNA and nucleotide metabolism': {metabolism_phrog_count}")
print(f"Number of PHROGs with less than 10 genes for 'connector': {connector_phrog_count}")
print(f"Number of PHROGs with less than 10 genes for 'integration and excision': {integration_excision_phrog_count}")
print(f"Number of PHROGs with less than 10 genes for 'transcription regulation': {transcription_phrog_count}")
print(f"Number of PHROGs with less than 10 genes for 'moron, auxiliary metabolic gene and host takeover': {moron_phrog_count}")
print(f"Number of PHROGs with less than 10 genes for 'other': {other_phrog_count}")



head_and_packaging_phrogs = phrog_sums.loc['head and packaging']
tail_phrogs = phrog_sums.loc['tail']
lysis_phrogs = phrog_sums.loc['lysis']
metabolism_phrogs = phrog_sums.loc['DNA, RNA and nucleotide metabolism']
connector_phrogs = phrog_sums.loc['connector']
integration_excision_phrogs = phrog_sums.loc['integration and excision']
transcription_phrogs = phrog_sums.loc['transcription regulation']
moron_phrogs = phrog_sums.loc['moron, auxiliary metabolic gene and host takeover']
other_phrogs = phrog_sums.loc['other']

# Count the number of genes in these PHROGs
head_and_packaging_genes_count = head_and_packaging_phrogs[head_and_packaging_phrogs < 10].sum()
tail_genes_count = tail_phrogs[tail_phrogs < 10].sum()
lysis_genes_count = lysis_phrogs[lysis_phrogs<10].sum()
metabolism_genes_count = metabolism_phrogs[metabolism_phrogs < 10].sum()
connector_genes_count = connector_phrogs[connector_phrogs < 10].sum()
integration_excision_genes_count = integration_excision_phrogs[integration_excision_phrogs < 10].sum()
transcription_genes_count = transcription_phrogs[transcription_phrogs < 10].sum()
moron_genes_count = moron_phrogs[moron_phrogs < 10].sum()
other_genes_count = other_phrogs[other_phrogs < 10].sum()

# Print the data
print(f"Number of genes in PHROGs with less than 10 genes for 'head and packaging': {head_and_packaging_genes_count}")
print(f"Number of genes in PHROGs with less than 10 genes for 'tail': {tail_genes_count}")
print(f"Number of genes in PHROGs with less than 10 genes for 'lysis': {lysis_genes_count}")
print(f"Number of genes in PHROGs with less than 10 genes for 'DNA, RNA and nucleotide metabolism': {metabolism_genes_count}")
print(f"Number of genes in PHROGs with less than 10 genes for 'connector': {connector_genes_count}")
print(f"Number of genes in PHROGs with less than 10 genes for 'integration and excision': {integration_excision_genes_count}")
print(f"Number of genes in PHROGs with less than 10 genes for 'transcription regulation': {transcription_genes_count}")
print(f"Number of genes in PHROGs with less than 10 genes for 'moron, auxiliary metabolic gene and host takeover': {moron_genes_count}")
print(f"Number of genes in PHROGs with less than 10 genes for 'other': {other_genes_count}")


phrog_sums = transposed_df.groupby(['category', 'phrog'])['count'].sum()

df_copy = df.copy()

# Drop specified columns
df_copy.drop(columns=['contig', 'gene', 'annot', 'category'], inplace=True)

# Sum up the occurrences of positive labels for each phrog
phrog_counts = df_copy.sum(axis=0)
phrog_counts = phrog_counts.astype(int)

# Get the phrogs with less than 10 genes
phrogs_to_remove = phrog_counts[phrog_counts < 10].index

# Filter the DataFrame to remove rows corresponding to these phrogs
filtered_df = df_copy[~df_copy[phrogs_to_remove].any(axis=1)]

# Write the filtered DataFrame back to a TSV file
filtered_df.to_csv('filtered_file.tsv', sep='\t', index=False)

# Now, if you want to keep the dropped columns in the saved file, you can concatenate the original columns with the filtered DataFrame
original_columns = df[['contig', 'gene', 'annot', 'category']]

merged_df = pd.merge(filtered_df, original_columns, left_index=True, right_index=True)
merged_df = merged_df[list(original_columns.columns) + list(filtered_df.columns)]

# Write the DataFrame with dropped columns and filtered rows to a TSV file
merged_df.to_csv('filtered_file_with_columns.tsv', sep='\t', index=False)


def delete_empty_rows(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, \
         open(output_file, 'w', newline='') as outfile:
        
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        
        for row in reader:
            # Check if any values exist from column 4 onwards
            if any(row[3:]):
                writer.writerow(row)

# Provide the input TSV file name and desired output file name
input_file = 'filtered_file_with_columns.tsv'
output_file = 'output_all_classes.tsv'

delete_empty_rows(input_file, output_file)

