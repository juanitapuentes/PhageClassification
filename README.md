# Unveiling the Viral Functional Dark Matter through Artificial Intelligence 

Juanita Puentes Mozo<sup>* 1</sup>, Camilo Garcia Botero<sup>1</sup> & Alejandro Reyes Muñoz<sup>1</sup>

<sup>1</sup> [Computational Biology and Microbial Ecology](https://cienciasbiologicas.uniandes.edu.co/es/investigacion/biologia-computacional-y-ecologia-microbiana), Department of Biological Sciences, Universidad de los Andes. Bogotá, Colombia <br/>

<p align="center">
</p>

___________


<p align="center">
<img src="figures/architectue.png" width="800">
</p>

_______
> Our proposed method leverages text embeddings from Large Language Models (LLMs) and visual features from Visual Transformers to classify viral proteins using 3Di FASTA representations. The model inputs are the FASTA sequences of amino acids from viral proteins, which are processed by the Text Module (TM). Each FASTA sequence is tokenized using BioTransformers and analyzed by ProteinBERT's transformer module. Additionally, the FASTA sequence is converted by ProstT5, a protein language model (pLM) that translates between protein sequences and structures, encoding protein structures as token sequences based on the 3Di-alphabet from the Foldseek 3D-alignment method. This sequence is then converted into a PNG image and processed by a Visual Transformer (ViT). The embeddings from both modules are concatenated and classified using a Feed Forward Network.


### What is a transformer?

Transformers are an AI language model that has significantly impacted speech recognition and automated chatbots. They are now recognized as a revolutionary method for understanding protein languages, providing a novel approach to studying and designing proteins. Transformers were first introduced in the paper "Attention is All You Need." This blog post offers a detailed exploration of the transformer architecture and the underlying mathematics.

### Why transformers for protein ?

Proteins are essential molecules that carry out vital functions in all living organisms. They are composed of one or more chains of amino acids. Despite there being only 20 distinct amino acids, their various combinations have produced thousands of functional proteins in humans. By considering amino acids as words and proteins as sentences, transformers can be utilized to interpret the language of proteins. When trained on billions of protein sequences identified across numerous species, a transformer can discern meaningful amino acid sequences and suggest new combinations from a linguistic standpoint.
 
<p align="center">
  <img width="50%" src="https://raw.githubusercontent.com/DeepChainBio/bio-transformers/main/.source/_static/protein.png">
</p>

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/DeepChainBio/bio-transformers/main/.source/_static/sequence.png">
</p>


Using a transformer model trained in protein language to analyze a specific sequence can reveal extensive details about the protein. As demonstrated in the earlier example, the transformer can identify which amino acids are crucial and must be present in the protein from a linguistic perspective. This insight is especially valuable for determining amino acid regions critical to the protein's function or stability.

## 0. Getting started

## Installation
It is recommended to work with conda environments in order to manage the specific dependencies of this package.
The `bio-transformers` package can be found on [pypi](https://pypi.org/project/bio-transformers/).

Please note that you are suppose to have a correct cuda/torch installation before installing this library.


### Work with conda environment

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual)

2. Create a virtual environment and activate it.

```bash
  conda create --name bio-transformers python=3.7 -y && conda activate bio-transformers
```

3. Install the package in environment.

```bash
  pip install bio-transformers
```

### Environment for developing

Conda:

1. Clone this git repo via HTTPS or SSH:

 ```bash
 git clone https://github.com/DeepChainBio/bio-transformers
 cd bio-transformers
 ```

2. Create developement environment based on the yaml file.

```bash
conda env create -f environment_dev.yaml
conda activate bio-transformers-dev
```

3. Install package and pre-commit hooks.

```
pip install -e .
pre-commit install
```

## Data

To download the dataset, please click the provided [link](https://drive.google.com/drive/folders/1vdIfHs8GdB_JMCP1g_U9gQ9yKoMKsd90?usp=sharing) and ensure that you place the downloaded folder at the same level as all the scripts within this directory. 

The folder contains two types of files: **refseq-pharokka-proteins.faa**, which includes all the FASTA amino acid sequences of the viral proteins, and **refseq-wide-all-phrogn.tsv**, which lists all the sequence IDs along with category-level and family-level annotations.






