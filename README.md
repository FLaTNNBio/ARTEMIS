# ARTEMIS: Causal Inference via Dynamic Contrastive Learning and Mutual Information


## 📌 Overview
This project proposes **ARTEMIS**, a novel architecture for estimating Conditional Average Treatment Effects (CATE). It leverages **Dynamic Contrastive Learning** alongside **Mutual Information (MI) bounds** to learn robust, balanced latent representations for causal inference. 

We benchmark ARTEMIS on standard causal inference datasets:
- **IHDP**
- **JOBS**
- **TCGA** (The Cancer Genome Atlas)

## 🚀 Key Features
- **Dynamic Contrastive Pairing:** Dynamically constructs positive and negative pairs based on Individual Treatment Effect (ITE) estimates.
- **Mutual Information Penalty:** Enforces independence between the learned latent space and the treatment assignment to reduce selection bias.
- **Ablation Studies:** Built-in experiment modes to dissect the contribution of MI and Contrastive Learning.
- **Hyperparameter Optimization:** Native integration with `Optuna` for extensive hyperparameter searches.

---

## 🛠 Prerequisites

### Environment Setup
We recommend using a `conda` or `venv` environment to manage dependencies.

```bash
conda create -n artemis python=3.9 -y
conda activate artemis
```

### Install Dependencies
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

---

## 📂 Data Preparation

Datasets must be placed in the respective `datasets/` directories. 
Update the paths in the configuration sections of the scripts (e.g., `JOBS_PATH` in `scripts/jobs/train_jobs.py`) to point to your local dataset files.

- **JOBS:** `datasets/jobs/jobs_DW_bin.new.10.train.npz`
- **TCGA:** Ensure the TCGA dataset is formatted appropriately for the benchmark in `datasets/tcga/`.

---

## 💻 Running the Code

The main entry points are the dataset-specific scripts. For example, to run the JOBS benchmark and ablation studies:

### 1. Execution Modes
In the main scripts, you can configure the `EXPERIMENT_MODE` variable:
- `"optuna"`: Runs hyperparameter search using Optuna to find the optimal configuration.
- `"final"`: Trains and evaluates the full ARTEMIS model using the best hyperparameters.
- `"ablation"`: Runs ablation studies (e.g., turning off MI, removing Contrastive Learning) using the best parameters.

### 2. Training & Evaluation
To train the model or run ablation studies, execute the target script:

```bash
python scripts/jobs/train_jobs.py
```
*Note: For the TCGA dataset, use `scripts/tcga/train_tcga.py`.*

---

## 📊 Results and Outputs
By default, results and model artifacts are saved in the output directories (e.g., `ablation_outputs_jobs_gmi/`).
- Best hyperparameter configurations are saved as `.json` and `.csv`.
- Training logs, evaluation metrics (e.g., Policy Risk, ATT error, PEHE), and ablation comparisons will be printed to the console and saved in the output folder.

---

## 🧠 Code Structure

- `scripts/jobs/train_jobs.py`: Main script for the JOBS dataset. Contains the Data Loader, CATE Encoder, Outcome Heads, and the training loop.
- `scripts/tcga/train_baselines.py`: Contains the baseline and comparative algorithms for the TCGA dataset.
- `scripts/tcga/train_tcga.py`: Main script for the TCGA dataset benchmark.
- `artemis/utils/losses.py`: Shared loss functions.

### Core Modules:
- **`CATEEncoder` / `Encoder`**: Encodes the input covariates into a latent representation using spectral normalization.
- **`OutcomeHead` / `DoseAwareNet`**: Multi-headed outcome network for each treatment branch.
- **`TreatmentClassifier`**: Classifier used to compute the Variational MI Lower Bound.
- **`DynamicContrastiveCausalDS`**: Custom PyTorch `Dataset` that dynamically pairs samples based on their predicted ITE.

---
