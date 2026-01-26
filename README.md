**English**| [中文](README_zh.md)
# MGM
[![](https://img.shields.io/pypi/v/microformer-mgm)](https://pypi.org/project/microformer-mgm/) 
[![Downloads](https://pepy.tech/badge/microformer-mgm)](https://pepy.tech/project/microformer-mgm)
![](https://img.shields.io/badge/status-beta-yellow?style=flat) 
![](https://img.shields.io/github/license/HUST-NingKang-Lab/MGM?style=flat) 

🎊 MGM has been published in [*Advanced Science* ](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202513333).

**Microbial General Model (MGM)** is a large-scale pretrained language model designed for interpretable microbiome data analysis. MGM supports a variety of tasks, including data preparation, model training, and inference, making it a versatile tool for microbiome research.

![MGM Pipeline](pipeline.png)


## Installation

### From PyPI

Install MGM using pip:

```bash
pip install microformer-mgm
```

### From Source

Install MGM from the source code:

```bash
python setup.py install
```

## MicroCorpus-260K

The **MicroCorpus-260K** dataset includes 263,302 microbiome samples sourced from MGnify, ideal for training your own MGM model. It is available for download on [OneDrive](https://1drv.ms/f/c/18d7815ed643fcef/EhpVq8yLORVImPIqpsFtHa4BrllBymIa-OWc7jS2vyADzw?e=AzZiLf). The dataset includes:

- **`MicroCorpus-260K.pkl`**: Normalized microbiome corpus (mean and standard deviation across all samples).
- **`MicroCorpus-260K_unnorm.pkl`**: Unnormalized microbiome corpus.
- **`mgnify_biomes.csv`**: Metadata for the samples in the dataset.

### Loading MicroCorpus-260K

Load the dataset in Python:

```python
from pickle import load
corpus = load(open('MicroCorpus-260K.pkl', 'rb'))
corpus[0]  # Access the first sample (dict with input_ids and attention_mask)
abundance = corpus.data  # Access the abundance data
```

## Example Notebook

The [MGM_Interpretability_Guideline.ipynb](MGM_Interpretability_Guideline.ipynb) Jupyter Notebook provides a comprehensive guide for using MGM to perform interpretable microbiome analysis. It demonstrates how to:

* Fine-tune MGM on an infant microbiome dataset.
* Extract sample embeddings for visualization.
* Compute attention weights to identify keystone microbial genera.

This notebook is ideal for researchers seeking to understand MGM's interpretability features and apply them to their own datasets.

## CLI Usage

MGM is accessed via a command-line interface (CLI) with various modes. The general syntax is:

```bash
mgm <mode> [options]
```

Below, the modes are grouped into **Data Preparation**, **Model Training**, and **Inference** for better organization.

### Data Preparation

#### `construct`

Converts abundance data into a microbiome corpus, normalized using phylogeny, and ranked from high to low genus abundance.

- **Input:** Abundance data in `hdf5`, `csv`, or `tsv` format (features in rows, samples in columns)
- **Output:** A `.pkl` file containing the microbiome corpus

**Example:**

```bash
mgm construct -i data/abundance.csv -o data/corpus.pkl
```

> **Note:** For `hdf5` files, use `-k` to specify the key (default is `genus`).

### Model Training

#### `pretrain`

Pretrains the MGM model using causal language modeling on a microbiome corpus. Optionally, trains a generator with labeled data.

- **Input:** 
  - Microbiome corpus (`.pkl`)
  - Optional: Label file (`.csv`, two columns: sample ID and label)
- **Output:** Pretrained MGM model

**Examples:**

```bash
mgm pretrain -i data/corpus.pkl -o models/pretrained_model
mgm pretrain -i data/corpus.pkl -l data/labels.csv -o models/generator_model --with-label
```

> **Note:** Use `--from-scratch` to train from scratch instead of loading pretrained weights. If a label file is provided, the tokenizer and model embedding layer are updated.

#### `train`

Trains a supervised MGM model from scratch using labeled data.

- **Input:** 
  - Microbiome corpus (`.pkl`)
  - Label file (`.csv`, two columns: sample ID and label)
- **Output:** Supervised MGM model

**Example:**

```bash
mgm train -i data/corpus.pkl -l data/labels.csv -o models/supervised_model
```

#### `finetune`

Finetunes a pretrained MGM model for a specific task using labeled data.

- **Input:** 
  - Microbiome corpus (`.pkl`)
  - Label file (`.csv`, two columns: sample ID and label)
  - Optional: Pretrained model (defaults to MicroCorpus-260K pretrained model if not specified)
- **Output:** Finetuned MGM model

**Example:**

```bash
mgm finetune -i data/corpus.pkl -l data/labels.csv -m models/pretrained_model -o models/finetuned_model
```

### Inference

#### `predict`

Generates predictions using a finetuned MGM model. Optionally evaluates against ground truth labels.

- **Input:** 
  - Microbiome corpus (`.pkl`)
  - Optional: Label file (`.csv`) for evaluation
  - Supervised MGM model
- **Output:** Prediction results (`.csv`)

**Example:**

```bash
mgm predict -E -i data/corpus.pkl -l data/labels.csv -m models/finetuned_model -o data/predictions.csv
```

> **Note:** Use `-E` with a label file to compare predictions with ground truth.

#### `generate`

Generates synthetic microbiome data using a pretrained MGM model.

- **Input:** 
  - Pretrained MGM model
  - Optional: Prompt file (`.txt`, one label per line) for labeled generation
- **Output:** Synthetic genus tensors (`.pkl`)

**Example:**

```bash
mgm generate -m models/generator_model -p data/prompt.txt -n 100 -o data/synthetic.pkl
```

> **Note:** Use `-n` to specify the number of samples to generate.

#### `reconstruct`

Reconstructs abundance data from a ranked corpus, with optional training of a reconstructor model or label decoding.

- **Input:** 
  - Abundance file (e.g., `csv`) for training the reconstructor, or a trained model checkpoint.
  - Ranked corpus (`.pkl`) for reconstruction
  - Optional: Generator model and prompt file (text, one label per line) for labeled data
- **Output:** 
  - Reconstructed corpus (`.pkl`)
  - Reconstructor model (if training)
  - Decoded labels (if applicable)

**Examples:**

```bash
mgm reconstruct -a data/abundance.csv -i data/synthetic.pkl -g models/generator_model -w True -o data/reconstructor
mgm reconstruct -r data/reconstructor_model.ckpt -i data/synthetic.pkl -g models/generator_model -w True -o data/reconstructed
```

For more details on any mode, run:

```bash
mgm <mode> --help
```

## Maintainers

| Name          | Email                     | Organization                                             |
|---------------|---------------------------|----------------------------------------------------------|
| Haohong Zhang | [haohongzh@gmail.com](mailto:haohongzh@gmail.com)       | PhD Student, School of Life Science and Technology, HUST |
| Zixin Kang    | [29590kang@gmail.com](mailto:29590kang@gmail.com)       | Undergraduate, School of Life Science and Technology, HUST |
| Kang Ning     | [ningkang@hust.edu.cn](mailto:ningkang@hust.edu.cn)      | Professor, School of Life Science and Technology, HUST   |
