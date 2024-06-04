## Unveiling the Viral Functional Dark Matter through Artificial Intelligence 

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



## Getting started

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





To download the dataset, please click the provided [link](https://drive.google.com/drive/folders/1vdIfHs8GdB_JMCP1g_U9gQ9yKoMKsd90?usp=sharing) and ensure that you place the downloaded folder at the same level as all the scripts within this directory. 
