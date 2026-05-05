
import os
import io
import sqlite3
import datetime
import logging
import copy
from itertools import cycle
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("TCGA_ARTEMIS_TRUE_MISE")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# USER CONFIG
# ==============================================================================
NPZ_PATH = "../../datasets/tcga/tcga.npz"
TCGA_DATA_DIR = "../../datasets/tcga"  # must contain tcga.db, min_val.npy, max_val.npy
N_RUNS = 5
DOSE_GRID_SIZE = 65
BENCHMARK_BATCH_SIZE = 32
BENCHMARK_SEED = 909
BENCHMARK_ASSIGNMENT_BIAS = 10.0

SAVE_RESULTS = True
OUT_DIR = "ablation_outputs_tcga_true_mise"
os.makedirs(OUT_DIR, exist_ok=True)

BEST_PARAMS = {
    "lr": 1e-3,
    "batch_size": 256,
    "latent_dim": 128,
    "hidden_dim": 256,
    "epochs": 150,
    "patience": 25,
    "main_weight_decay": 1e-4,
    "clf_weight_decay": 1e-5,
    "lr_treat_clf": 3e-4,
    "treat_clf_steps": 5,
    "clip_norm": 2.0,

    "dose_loss_weight": 0.05,

    "alpha": 0.10,
    "margin": 0.60,
    "warmup_epochs": 10,
    "pair_update_freq": 5,
    "pair_threshold_percentile": 30.0,

    "lambda_mi_pos": 0.05,
    "mi_start_epoch": 20,
    "mi_pos_min_count": 12,

    "encoder_dropout": 0.15,
    "clf_dropout": 0.15,

    "use_contrastive": True,
    "use_local_mi": True,
    "use_dynamic_update": True,
    "pair_mode": "dynamic_ite",
    "feature_k": 20,

    # MI ablation mode:
    # - "local": MI is estimated only on positive/effect-homogeneous pairs.
    # - "global": MI is estimated on all latent representations in the batch.
    "mi_mode": "local",
}

# ==============================================================================
# Minimal benchmark utilities (standalone DRNet-like TCGA reconstruction)
# ==============================================================================
LAST_ROW_ID = None
LAST_ID_SET = None

def clip_percentage(x):
    return np.clip(x, 0.0, 1.0)

def stable_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def gaussian(x, mean, std):
    std = max(float(std), 1e-8)
    return np.exp(-0.5 * ((x - mean) / std) ** 2)

def random_cycle_generator(items):
    rng = np.random.RandomState(0)
    items = np.asarray(items)
    while True:
        perm = rng.permutation(items)
        for x in perm:
            yield x

def resample_with_replacement_generator(items):
    rng = np.random.RandomState(0)
    items = np.asarray(items)
    while True:
        yield items[rng.randint(0, len(items))]

def get_last_row_id():
    return LAST_ROW_ID

def get_last_id_set():
    return LAST_ID_SET

def report_distribution(data, labels, num_classes, set_name):
    counts = np.zeros((num_classes,))
    for i in range(num_classes):
        counts[i] = np.sum(labels == i) / float(len(labels))
    print("INFO: Using", set_name, "set (n=", len(data), ") with distribution", counts)

class BaseDataAccess:
    def get_split_indices(self):
        return (None, None)
    def make_propensity_lists(self, *args, **kwargs):
        return None

    def create_temporary_table(self, table_name, values):
        self.db.execute(f"CREATE TEMP TABLE {table_name} (id INT);")
        if len(values) != 0:
            self.db.executemany(f"INSERT INTO {table_name} VALUES (?);", values)
        return table_name

    def drop_temporary_table(self, table_name):
        self.db.execute(f"DROP TABLE {table_name};")

    def get_rows(self, train_ids, columns=""):
        tmp_name = "tmp_data"
        self.create_temporary_table(tmp_name, [(int(x),) for x in train_ids])

        patients = self.db.execute(
            f"SELECT rowid, * FROM {DataAccess.TABLE_CLINICAL} "
            f"WHERE rowid IN (SELECT id FROM {tmp_name});"
        ).fetchall()

        self.drop_temporary_table(tmp_name)

        patient_rowids = [x[0] for x in patients]
        patient_ids = [x[1] for x in patients]

        tmp_name = "patient_ids"
        self.create_temporary_table(tmp_name, [(x,) for x in patient_ids])

        rnaseq_rows = self.db.execute(
            f"SELECT * FROM {DataAccess.TABLE_RNASEQ} "
            f"WHERE clinical_id IN (SELECT id FROM {tmp_name});"
        ).fetchall()

        self.drop_temporary_table(tmp_name)

        id_seq_map = {}
        for sample in rnaseq_rows:
            clinical_id = sample[-1]
            id_seq_map[clinical_id] = sample

        rnaseq_data = np.array([id_seq_map[pid][1] for pid in patient_ids], dtype=np.float32)
        rnaseq_data = (rnaseq_data - self.min_val) / (self.max_val - self.min_val + 1e-5)

        return rnaseq_data, patient_rowids, rnaseq_data

class Benchmark:
    def __init__(self, data_access, num_treatments=2, **kwargs):
        self.data_access = data_access
        self.num_treatments = num_treatments
        self.assign_counterfactuals = True
    def get_data_access(self):
        return self.data_access
    def get_num_treatments(self):
        return self.num_treatments
    def filter(self, patients):
        return patients

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle=True)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)
sqlite3.register_converter("DATE", lambda x: datetime.datetime.fromtimestamp(float(x) / 1000))

class DataAccess(BaseDataAccess):
    DB_FILE_NAME = "tcga.db"
    MIN_FILE_NAME = "min_val.npy"
    MAX_FILE_NAME = "max_val.npy"

    TABLE_CLINICAL = "clinical"
    TABLE_RNASEQ = "rnaseq"

    def __init__(self, data_dir, **kwargs):
        self.data_dir = data_dir
        self.tcga_num_features = int(np.rint(kwargs["tcga_num_features"]))
        self.db = None

        min_path = os.path.join(self.data_dir, DataAccess.MIN_FILE_NAME)
        max_path = os.path.join(self.data_dir, DataAccess.MAX_FILE_NAME)
        if os.path.exists(min_path) and os.path.exists(max_path):
            self.min_val, self.max_val = np.load(min_path)[:-1], np.load(max_path)[:-1]
        else:
            raise FileNotFoundError(
                f"Expected {DataAccess.MIN_FILE_NAME} and {DataAccess.MAX_FILE_NAME} inside {self.data_dir}"
            )

        self.connect()
        self.setup_schema()

    def connect(self):
        self.db = sqlite3.connect(
            os.path.join(self.data_dir, DataAccess.DB_FILE_NAME),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self.db.execute("PRAGMA journal_mode = OFF;")
        self.db.execute("PRAGMA page_size = 16384;")

    def setup_schema(self):
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS clinical ("
            "id TEXT NOT NULL PRIMARY KEY, age INT, gender INT, icd10_diagnosis TEXT, dataset_name TEXT, "
            "days_to_death INT, days_to_recurrence INT, days_to_surgery INT, did_radiation_therapy INT);"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS rnaseq ("
            "id TEXT NOT NULL PRIMARY KEY, data ARRAY, clinical_id TEXT NOT NULL, "
            "FOREIGN KEY(clinical_id) REFERENCES clinical(id));"
        )
        self.db.commit()

    def get_row(self, table_name, id, with_rowid=False):
        columns = "*"
        if with_rowid:
            columns = "rowid, " + columns
        if isinstance(id, tuple):
            id = id[0]
        id = int(id)
        query = f"SELECT {columns} FROM {table_name} WHERE rowid = ?;"
        return self.db.execute(query, (id,)).fetchone()

    def get_rows_by_clinical_id(self, table_name, id, with_rowid=False):
        columns = "*"
        if with_rowid:
            columns = "rowid, " + columns
        query = f"SELECT {columns} FROM {table_name} WHERE clinical_id = ?;"
        return self.db.execute(query, (id,)).fetchone()

    def get_entry_with_id(self, id, args=None):
        if args is None:
            args = {}

        with_rnaseq = args.get("with_rnaseq", True)

        if isinstance(id, tuple):
            id = id[0]
        id = int(id)

        patient = self.get_row(DataAccess.TABLE_CLINICAL, id, with_rowid=True)
        if patient is None:
            raise ValueError(f"No clinical row found for rowid={id}")

        patient_rowid = patient[0]
        patient_id = patient[1]

        result = {"clinical": patient}

        if with_rnaseq:
            result["rnaseq"] = self.get_rows_by_clinical_id(DataAccess.TABLE_RNASEQ, patient_id)

        return patient_rowid, result

    def get_labelled_patients(self):
        return_value = self.db.execute(
            "SELECT clinical.rowid FROM clinical WHERE clinical.id IN "
            "(SELECT clinical_id FROM rnaseq) ORDER BY clinical.rowid;"
        ).fetchall()
        return np.asarray(np.squeeze(return_value)).astype(np.int64)

    def get_rnaseq_dimension(self):
        rnaseq = self.db.execute("SELECT data FROM rnaseq WHERE rowid = 1;").fetchone()[0]
        return rnaseq.shape[0]

    def get_labels(self, args, patients, benchmark):
        tcga_num_features = int(np.rint(args["tcga_num_features"]))
        assignments = []
        for id in patients:
            pid = int(id[0]) if isinstance(id, tuple) else int(id)
            entry = self.get_entry_with_id(pid, {"with_rnaseq": True})[1]
            rnaseq_data = np.array(entry["rnaseq"][1], dtype=np.float32)
            rnaseq_data = (rnaseq_data - self.min_val) / (self.max_val - self.min_val + 1e-5)
            if tcga_num_features > 0:
                rnaseq_data = rnaseq_data[:tcga_num_features]
            assignment = benchmark.get_assignment(pid, rnaseq_data)[0]
            assignments.append(assignment)
        assignments = np.array(assignments)
        num_labels = benchmark.get_num_treatments()
        return assignments, num_labels

    def prepare_batch(self, args, batch_data, benchmark, is_train=False):
        with_exposure = args["with_exposure"]
        tcga_num_features = self.tcga_num_features

        patient_ids = np.array([x["clinical"][0] for x in batch_data], dtype=np.int64)
        rnaseq_data = np.array([x["rnaseq"][1] for x in batch_data], dtype=np.float32)
        rnaseq_data = (rnaseq_data - self.min_val) / (self.max_val - self.min_val + 1e-5)

        assignments = list(map(benchmark.get_assignment, patient_ids, rnaseq_data))

        if with_exposure:
            treatment_data, batch_y, treatment_strength = zip(*assignments)
        else:
            treatment_data, batch_y = zip(*assignments)
            treatment_strength = None

        treatment_data = np.array(treatment_data)

        if tcga_num_features > 0:
            rnaseq_data = benchmark.select_features(rnaseq_data)

        batch_y = np.array(batch_y)
        batch_x = [rnaseq_data, treatment_data]

        if with_exposure:
            batch_x += [np.array(treatment_strength)]

        return batch_x, batch_y

class TCGABenchmark(Benchmark):
    def __init__(self, data_dir, num_treatments=4, num_centroids_mean=7, num_centroids_std=2,
                 num_relevant_gene_loci_mean=10, num_relevant_gene_loci_std=3, response_mean_of_mean=0.45,
                 response_std_of_mean=0.15, response_mean_of_std=0.1, response_std_of_std=0.05,
                 strength_of_assignment_bias=10, epsilon_std=0.15, with_exposure=True, **kwargs):
        super().__init__(DataAccess(data_dir, **kwargs), num_treatments, **kwargs)
        self.centroids = None
        self.dosage_centroids = None
        self.with_exposure = with_exposure
        self.assignment_cache = {}
        self.num_centroids_mean = num_centroids_mean
        self.num_centroids_std = num_centroids_std
        self.num_relevant_gene_loci_mean = num_relevant_gene_loci_mean
        self.num_relevant_gene_loci_std = num_relevant_gene_loci_std
        self.response_mean_of_mean = response_mean_of_mean
        self.response_std_of_mean = response_std_of_mean
        self.response_mean_of_std = response_mean_of_std
        self.response_std_of_std = response_std_of_std
        self.strength_of_assignment_bias = strength_of_assignment_bias
        self.epsilon_std = epsilon_std
        self.seed = kwargs["seed"]
        self.random_generator = None
        self.num_features = int(np.rint(kwargs["tcga_num_features"]))
        self.num_archetypes_per_treatment = 2
        self.scaling_constant = 50

    def get_scaling_constant(self):
        return self.scaling_constant

    def initialise(self, args):
        self.random_generator = np.random.RandomState(909)
        self.centroids = None
        all_features = self.data_access.get_rnaseq_dimension()
        if self.num_features > 0 and self.num_features != all_features:
            self.selected_features = self.random_generator.choice(
                self.data_access.get_rnaseq_dimension(), self.num_features, replace=False
            )
        else:
            self.selected_features = np.arange(all_features)

    def select_features(self, x):
        return x[:, self.selected_features]

    def get_from_generator_with_offsets(self, generator, centroid_indices, adjust_last=False):
        centroids_tmp, current_idx = [], 0
        centroid_indices = list(centroid_indices)

        while len(centroid_indices) != 0:
            x, _ = next(generator)
            ids = get_last_id_set()
            batch_size = len(x[0])

            while len(centroid_indices) != 0 and centroid_indices[0] < current_idx + batch_size:
                next_index_global = int(centroid_indices[0])
                del centroid_indices[0]
                local_idx = next_index_global - current_idx

                is_last_treatment = len(centroid_indices) == 0
                if is_last_treatment and adjust_last:
                    response_mean_of_mean = 1 - self.response_mean_of_mean
                else:
                    response_mean_of_mean = self.response_mean_of_mean

                response_mean = clip_percentage(
                    self.random_generator.normal(response_mean_of_mean, self.response_std_of_mean)
                )
                response_std = clip_percentage(
                    self.random_generator.normal(self.response_mean_of_std, self.response_std_of_std)
                ) + 0.025

                gene_loci_indices = np.arange(len(x[0][local_idx]))
                rnaseq_data = self.data_access.get_entry_with_id(int(ids[local_idx]), {"with_rnaseq": True})[1]["rnaseq"][1]
                centroid_data = (
                    gene_loci_indices,
                    rnaseq_data[gene_loci_indices],
                    response_mean,
                    response_std,
                )
                centroids_tmp.append(centroid_data)

            current_idx += batch_size
        return centroids_tmp

    def fit(self, generator, steps, batch_size):
        num_samples = steps * batch_size
        centroid_indices = sorted(self.random_generator.permutation(num_samples)[: self.num_treatments + 1])

        if self.with_exposure:
            self.dosage_centroids = []
            for treatment_idx in range(self.num_treatments):
                dosage_centroid_indices = sorted(
                    self.random_generator.permutation(num_samples)[: self.num_archetypes_per_treatment]
                )
                self.dosage_centroids.append(
                    self.get_from_generator_with_offsets(generator, dosage_centroid_indices)
                )
                for dosage_idx in range(self.num_archetypes_per_treatment):
                    min_response = self.random_generator.normal(0.0, 0.1)
                    self.dosage_centroids[treatment_idx][dosage_idx] += (min_response,)

        self.centroids = self.get_from_generator_with_offsets(generator, centroid_indices, adjust_last=True)
        self.assignment_cache = {}

    def get_centroid_weights(self, x, centroids=None):
        if centroids is None:
            centroids = self.centroids
        similarities = list(map(
            lambda indices, centroid: cosine_similarity(
                x[indices].reshape(1, -1), centroid.reshape(1, -1)
            ),
            map(lambda x: x[0], centroids),
            map(lambda x: x[1], centroids),
        ))
        return np.squeeze(similarities)

    def get_dose_response_curve(self, z, treatment_idx, return_all=False):
        dosage_distances = self.get_centroid_weights(z, centroids=self.dosage_centroids[treatment_idx])
        normalised_distances = stable_softmax(self.strength_of_assignment_bias * dosage_distances)
        d = normalised_distances
        _, _, d0_mean, d0_std, d0_min = self.dosage_centroids[treatment_idx][0]
        _, _, d1_mean, d1_std, d1_min = self.dosage_centroids[treatment_idx][1]

        def dose_response_curve(treatment_strength):
            this_y = d[0] * gaussian(treatment_strength - d0_min, d0_mean, d0_std) + \
                     d[1] * gaussian(treatment_strength - d1_min, d1_mean, d1_std)
            return this_y

        if return_all:
            return dose_response_curve, d, d0_mean, d0_std, d0_min, d1_mean, d1_std, d1_min
        else:
            return dose_response_curve

    def _assign(self, x):
        distances = self.get_centroid_weights(x)
        expected_responses = []
        for treatment in range(self.num_treatments + 1):
            _, _, response_mean, response_std = self.centroids[treatment]
            y_this_treatment = self.random_generator.normal(response_mean, response_std)
            expected_responses.append(
                clip_percentage(y_this_treatment + self.random_generator.normal(0.0, self.epsilon_std))
            )
        expected_responses = np.array(expected_responses)

        y = []
        if self.with_exposure:
            treatment_strengths = []
            for treatment_idx in range(self.num_treatments):
                dose_response_curve, d, d0_mean, d0_std, d0_min, d1_mean, d1_std, d1_min = \
                    self.get_dose_response_curve(x, treatment_idx, return_all=True)
                treatment_strength = clip_percentage(self.random_generator.normal(0.65, 0.1))
                treatment_strengths.append(treatment_strength)
                this_y = dose_response_curve(treatment_strength)
                y.append(this_y * expected_responses[treatment_idx])
            treatment_strengths = np.array(treatment_strengths)
        else:
            raise NotImplementedError
        y = np.array(y)

        treatment_chosen = self.random_generator.choice(
            self.num_treatments,
            p=stable_softmax(self.strength_of_assignment_bias * y)
        )

        return treatment_chosen, self.scaling_constant * y, treatment_strengths

    def get_assignment(self, id, x):
        if self.centroids is None:
            return 0, 0, 0

        if id not in self.assignment_cache:
            rnaseq_data = self.data_access.get_entry_with_id(id, {"with_rnaseq": True})[1]["rnaseq"][1]
            values = self._assign(rnaseq_data)
            self.assignment_cache[id] = values

        assigned_treatment, assigned_y, treatment_strength = self.assignment_cache[id]
        if self.assign_counterfactuals:
            return assigned_treatment, assigned_y, treatment_strength
        else:
            return assigned_treatment, assigned_y[assigned_treatment], treatment_strength[assigned_treatment]

def make_generator(args, benchmark, is_validation=False, is_test=False,
                   validation_fraction=0.2, test_fraction=0.2, seed=909, randomise=True,
                   stratify=True, resample_with_replacement=False):
    fraction_of_data_set = args["fraction_of_data_set"]
    patients = benchmark.get_data_access().get_labelled_patients()
    patients = np.asarray(patients).astype(np.int64)
    patients = benchmark.filter(patients)

    num_patients = len(patients)
    if fraction_of_data_set < 1.0:
        num_patients = int(np.rint(num_patients * fraction_of_data_set))
        patients = np.random.RandomState(0).permutation(patients)[:num_patients]

    num_validation_patients = int(np.floor(num_patients * validation_fraction))
    num_test_patients = int(np.floor(num_patients * test_fraction))

    split_indices = benchmark.get_data_access().get_split_indices()
    if stratify:
        labels, num_labels = benchmark.get_data_access().get_labels(args, map(lambda x: (x,), patients), benchmark)
        if split_indices[0] is None:
            test_sss = StratifiedShuffleSplit(n_splits=1, test_size=num_test_patients, random_state=0)
            rest_indices, test_indices = next(test_sss.split(patients, labels))
        else:
            rest_indices, test_indices = split_indices

        val_sss = StratifiedShuffleSplit(n_splits=1, test_size=num_validation_patients, random_state=0)
        train_indices, val_indices = next(val_sss.split(patients[rest_indices], labels[rest_indices]))

        if is_test:
            report_distribution(patients[rest_indices][train_indices], labels[rest_indices][train_indices], num_labels, "train")
            report_distribution(patients[rest_indices][val_indices], labels[rest_indices][val_indices], num_labels, "validation")
            report_distribution(patients[test_indices], labels[test_indices], num_labels, "test")
    else:
        if split_indices[0] is None:
            indices = np.random.RandomState(0).permutation(num_patients)
            rest_indices, test_indices = indices[num_test_patients:], indices[:num_test_patients]
        else:
            rest_indices, test_indices = split_indices
        remaining_indices = np.random.RandomState(0).permutation(len(rest_indices))
        train_indices, val_indices = remaining_indices[num_validation_patients:], remaining_indices[:num_validation_patients]

    if is_test:
        patients = patients[test_indices]
    elif is_validation:
        patients = patients[rest_indices][val_indices]
    else:
        patients = patients[rest_indices][train_indices]

    num_steps = len(patients)

    def generator():
        global LAST_ROW_ID
        if resample_with_replacement:
            id_generator = resample_with_replacement_generator(patients)
        else:
            id_generator = random_cycle_generator(patients) if randomise else cycle(patients)

        while True:
            next_patient_id = next(id_generator)
            patient_id, result = benchmark.get_data_access().get_entry_with_id(next_patient_id, args)
            LAST_ROW_ID = patient_id
            yield result

    return generator(), num_steps

def to_categorical(x, num_classes):
    x = np.asarray(x, dtype=np.int64)
    out = np.zeros((len(x), num_classes), dtype=np.float32)
    out[np.arange(len(x)), x] = 1.0
    return out

def make_keras_generator(args, wrapped_generator, num_steps, batch_size=1, num_losses=1,
                         benchmark=None, is_train=False):
    method = args["method"]
    with_propensity_dropout = args["with_propensity_dropout"]
    num_steps = num_steps // batch_size

    def generator():
        global LAST_ID_SET
        while True:
            batch_data, ids = zip(*[(next(wrapped_generator), get_last_row_id()) for _ in range(batch_size)])
            LAST_ID_SET = ids
            batch_x, batch_y = benchmark.get_data_access().prepare_batch(args, batch_data, benchmark, is_train)
            if num_losses != 1:
                batch_y = batch_y * num_losses
            if with_propensity_dropout and (method == "nn" or method == "nn+"):
                batch_y = [to_categorical(batch_x[1], num_classes=benchmark.get_num_treatments()), batch_y]
            yield batch_x, batch_y

    return generator(), num_steps

def get_normalised_rnaseq(benchmark: TCGABenchmark, patient_rowid: int) -> np.ndarray:
    entry = benchmark.get_data_access().get_entry_with_id(patient_rowid, {"with_rnaseq": True})[1]
    rnaseq_data = np.array(entry["rnaseq"][1], dtype=np.float32)
    rnaseq_data = (rnaseq_data - benchmark.get_data_access().min_val) / (
        benchmark.get_data_access().max_val - benchmark.get_data_access().min_val + 1e-5
    )
    if benchmark.num_features > 0:
        rnaseq_data = rnaseq_data[:benchmark.num_features]
    return rnaseq_data

def reconstruct_true_curves_for_patient_from_x(benchmark, patient_rowid, x, dose_grid):
    """
    Reconstruct benchmark true curves using an already-normalised RNA-seq vector x.
    """
    # Ensure benchmark cache for this patient exists.
    _ = benchmark.get_assignment(int(patient_rowid), x)

    assigned = benchmark.assignment_cache[int(patient_rowid)]

    # with_exposure=True -> (treatment_chosen, scaled_y, treatment_strengths)
    _, scaled_y, _ = assigned
    scaled_y = np.asarray(scaled_y, dtype=np.float32)

    K = benchmark.num_treatments
    M = len(dose_grid)
    curves = np.zeros((K, M), dtype=np.float32)

    for t in range(K):
        dose_curve = benchmark.get_dose_response_curve(x, t)
        observed_strength = float(scaled_y[t])

        # Recover the multiplicative expected-response factor by dividing
        # observed outcome at assigned dose by the raw dose-curve value there.
        # We use the stored factual dose for treatment t from assignment_cache.
        treatment_strength = float(assigned[2][t])
        factual_curve_val = float(dose_curve(treatment_strength))

        if abs(factual_curve_val) < 1e-8:
            scale_factor = 0.0
        else:
            scale_factor = observed_strength / factual_curve_val

        curves[t, :] = np.array(
            [scale_factor * float(dose_curve(float(s))) for s in dose_grid],
            dtype=np.float32
        )

    return curves

def reconstruct_true_curves_for_patient(benchmark: TCGABenchmark, patient_rowid: int, dose_grid: np.ndarray,
                                        eps: float = 1e-10) -> np.ndarray:
    x = get_normalised_rnaseq(benchmark, patient_rowid)
    _, y_all, treatment_strengths = benchmark.get_assignment(patient_rowid, x)
    K = benchmark.get_num_treatments()
    true_curves = np.zeros((K, len(dose_grid)), dtype=np.float32)
    for t in range(K):
        curve = benchmark.get_dose_response_curve(x, t)
        s_obs = float(treatment_strengths[t])
        curve_at_obs = float(curve(s_obs))
        expected_response_t = float(y_all[t]) / (benchmark.get_scaling_constant() * max(curve_at_obs, eps))
        true_curves[t, :] = benchmark.get_scaling_constant() * expected_response_t * np.array(
            [curve(float(s)) for s in dose_grid], dtype=np.float32
        )
    return true_curves


def build_true_response_grid(benchmark: TCGABenchmark, patient_ids: np.ndarray, dose_grid: np.ndarray) -> np.ndarray:
    """
    Build true response curves for all patients using bulk RNA-seq retrieval,
    preserving the exact order of patient_ids.
    """
    patient_ids = np.asarray(patient_ids).astype(np.int64)

    rnaseq_data, patient_rowids, _ = benchmark.get_data_access().get_rows(patient_ids)
    patient_rowids = np.asarray(list(patient_rowids)).astype(np.int64)

    if benchmark.num_features > 0:
        rnaseq_data = benchmark.select_features(rnaseq_data)

    grids = []
    for pid, x in zip(patient_rowids, rnaseq_data):
        grids.append(reconstruct_true_curves_for_patient_from_x(benchmark, int(pid), x, dose_grid))

    return np.stack(grids, axis=0)  # [N, K, M]

def build_benchmark_and_true_grid_all(data_dir: str, tcga_num_features: int, num_treatments: int,
                                      dose_grid_size: int, batch_size: int, seed: int,
                                      strength_of_assignment_bias: float, total_n: int):
    args = {
        "seed": seed,
        "tcga_num_features": tcga_num_features,
        "with_exposure": True,
        "fraction_of_data_set": 1.0,
        "with_propensity_batch": False,
        "method": "nn",
        "with_propensity_dropout": False,
    }
    benchmark = TCGABenchmark(
        data_dir=data_dir,
        num_treatments=num_treatments,
        strength_of_assignment_bias=strength_of_assignment_bias,
        with_exposure=True,
        seed=seed,
        tcga_num_features=tcga_num_features,
    )
    benchmark.initialise(args)

    wrapped_generator, num_steps = make_generator(
        args=args, benchmark=benchmark, is_validation=False, is_test=False,
        validation_fraction=0.27, test_fraction=0.10, seed=seed,
        randomise=True, stratify=True, resample_with_replacement=False,
    )
    keras_gen, keras_steps = make_keras_generator(
        args=args, wrapped_generator=wrapped_generator, num_steps=num_steps,
        batch_size=batch_size, num_losses=1, benchmark=benchmark, is_train=True,
    )
    LOGGER.info("[TRUE_MISE] Building benchmark state...")
    benchmark.fit(keras_gen, steps=keras_steps, batch_size=batch_size)
    LOGGER.info("[TRUE_MISE] Benchmark fit complete.")

    dose_grid = np.linspace(0.0, 1.0, dose_grid_size, dtype=np.float32)
    # Assumption: NPZ rows follow sqlite clinical rowid order 1..N.
    dose_grid = np.linspace(0.0, 1.0, dose_grid_size, dtype=np.float32)

    patient_ids = benchmark.get_data_access().get_labelled_patients()
    patient_ids = np.asarray(patient_ids).astype(np.int64)
    patient_ids = benchmark.filter(patient_ids)

    LOGGER.info(f"[TRUE_MISE] labelled patients in DB = {len(patient_ids)}")
    LOGGER.info(f"[TRUE_MISE] NPZ rows = {total_n}")

    LOGGER.info("[TRUE_MISE] Building full true response grid ...")
    true_grid_all = build_true_response_grid(benchmark, patient_ids, dose_grid)
    LOGGER.info(f"[TRUE_MISE] true_grid_all shape = {true_grid_all.shape}")

    return benchmark, dose_grid.astype(np.float32), true_grid_all.astype(np.float32)

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_tcga_mitnet_npz(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    required = ["feature", "t", "y", "eval_y", "d", "eval_d"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Missing keys in NPZ: {missing}. Available: {list(data.files)}")

    X = np.asarray(data["feature"], dtype=np.float32)
    T = np.asarray(data["t"], dtype=np.float32).reshape(-1)
    Y = np.asarray(data["y"], dtype=np.float32)
    EVAL_Y = np.asarray(data["eval_y"], dtype=np.float32)
    D = np.asarray(data["d"], dtype=np.float32)
    EVAL_D = np.asarray(data["eval_d"], dtype=np.float32)

    if np.allclose(T, np.round(T)):
        T = np.round(T).astype(np.int64)
    else:
        raise ValueError("t contains non-integer treatment labels.")

    N = X.shape[0]
    num_treatments = Y.shape[1]

    vals, counts = np.unique(T, return_counts=True)
    LOGGER.info(f"[LOAD] X={X.shape}, T={T.shape}, Y={Y.shape}, EVAL_Y={EVAL_Y.shape}, D={D.shape}, EVAL_D={EVAL_D.shape}")
    LOGGER.info(f"[LOAD] treatments: {list(zip(vals.tolist(), counts.tolist()))}")
    LOGGER.warning("[EVAL] This script uses TRUE continuous-dose MISE for early stopping/test, while pairing bootstrap still uses discrete eval_d/eval_y.")

    return {"X": X, "T": T, "Y": Y, "EVAL_Y": EVAL_Y, "D": D, "EVAL_D": EVAL_D, "K": num_treatments}

# ==============================================================================
# UTILS
# ==============================================================================
class EarlyStoppingMetric:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = np.inf
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, value, model):
        if value < self.best_value - self.min_delta:
            self.best_value = float(value)
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

def get_continuous_indices(X):
    return [c for c in range(X.shape[1]) if len(np.unique(X[:, c])) > 2]

def standardize_splits(X_train, X_val, X_test):
    cont_indices = get_continuous_indices(X_train)
    X_train = X_train.copy(); X_val = X_val.copy(); X_test = X_test.copy()
    if cont_indices:
        x_mean = np.mean(X_train[:, cont_indices], axis=0, keepdims=True)
        x_std = np.maximum(np.std(X_train[:, cont_indices], axis=0, keepdims=True), 1e-6)
        X_train[:, cont_indices] = (X_train[:, cont_indices] - x_mean) / x_std
        X_val[:, cont_indices] = (X_val[:, cont_indices] - x_mean) / x_std
        X_test[:, cont_indices] = (X_test[:, cont_indices] - x_mean) / x_std
    return X_train, X_val, X_test

def standardize_y_factual(y_train, y_val, y_test):
    y_mean = float(np.mean(y_train))
    y_std = max(float(np.std(y_train)), 1e-6)
    return (y_train - y_mean) / y_std, (y_val - y_mean) / y_std, (y_test - y_mean) / y_std, y_mean, y_std

def empirical_entropy_from_labels(t_idx: torch.Tensor, num_treatments: int, eps: float = 1e-8) -> torch.Tensor:
    t_idx = t_idx.view(-1).long()
    counts = torch.bincount(t_idx, minlength=num_treatments).float()
    probs = counts / counts.sum().clamp_min(1.0)
    return -(probs * torch.log(probs + eps)).sum()

def treatment_log_prob_mean(classifier: nn.Module, z: torch.Tensor, t_idx: torch.Tensor, num_treatments: int) -> torch.Tensor:
    logits = classifier(z)
    if num_treatments == 2:
        t_float = t_idx.view(-1, 1).float()
        return (t_float * F.logsigmoid(logits) + (1.0 - t_float) * F.logsigmoid(-logits)).mean()
    log_probs = F.log_softmax(logits, dim=1)
    chosen = log_probs.gather(1, t_idx.view(-1, 1).long())
    return chosen.mean()

def treatment_classifier_loss(classifier: nn.Module, z: torch.Tensor, t_idx: torch.Tensor, num_treatments: int) -> torch.Tensor:
    logits = classifier(z)
    if num_treatments == 2:
        return F.binary_cross_entropy_with_logits(logits, t_idx.view(-1, 1).float())
    return F.cross_entropy(logits, t_idx.view(-1).long())

def variational_mi_lower_bound(classifier: nn.Module, z: torch.Tensor, t_idx: torch.Tensor, num_treatments: int) -> torch.Tensor:
    return empirical_entropy_from_labels(t_idx.detach(), num_treatments) + treatment_log_prob_mean(classifier, z, t_idx, num_treatments)

def contrastive_loss(z1, z2, label, margin=1.0):
    label = label.float()
    dist_sq = torch.sum((z1 - z2) ** 2, dim=1)
    loss_sim = label * dist_sq
    loss_dissim = (1.0 - label) * torch.pow(torch.clamp(margin - torch.sqrt(dist_sq + 1e-8), min=0.0), 2)
    return torch.mean(loss_sim + loss_dissim) / 2.0

# ==============================================================================
# SPLIT
# ==============================================================================
def split_tcga_bundle(bundle: Dict[str, np.ndarray], seed: int, true_grid_all: np.ndarray) -> Dict[str, Any]:
    X = bundle["X"]; T = bundle["T"]; Y = bundle["Y"]; EVAL_Y = bundle["EVAL_Y"]; D = bundle["D"]; EVAL_D = bundle["EVAL_D"]; K = bundle["K"]
    idx_all = np.arange(len(X), dtype=np.int64)
    dose_bins = np.digitize(np.mean(EVAL_D, axis=1), np.linspace(0.0, 1.0, 6), right=False)
    strat_joint = T.astype(str) + "_" + dose_bins.astype(str)

    def _safe_stratified_split(X_in, labels, test_size, random_state):
        labels = np.asarray(labels)
        _, counts = np.unique(labels, return_counts=True)
        if np.min(counts) >= 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            idx_a, idx_b = next(sss.split(X_in, labels))
            return idx_a, idx_b, True
        return None, None, False

    tr_idx, tmp_idx, ok_joint = _safe_stratified_split(X, strat_joint, 0.37, seed)
    if not ok_joint:
        LOGGER.warning("[SPLIT] Joint stratification failed. Falling back to T only.")
        tr_idx, tmp_idx, ok_t = _safe_stratified_split(X, T, 0.37, seed)
        if not ok_t:
            rng = np.random.default_rng(seed); idx = np.arange(len(X)); rng.shuffle(idx); cut = int(round(len(X) * 0.63))
            tr_idx, tmp_idx = idx[:cut], idx[cut:]

    X_tr, T_tr, Y_tr, EVAL_Y_tr, D_tr, EVAL_D_tr = X[tr_idx], T[tr_idx], Y[tr_idx], EVAL_Y[tr_idx], D[tr_idx], EVAL_D[tr_idx]
    X_tmp, T_tmp, Y_tmp, EVAL_Y_tmp, D_tmp, EVAL_D_tmp = X[tmp_idx], T[tmp_idx], Y[tmp_idx], EVAL_Y[tmp_idx], D[tmp_idx], EVAL_D[tmp_idx]
    idx_tmp = idx_all[tmp_idx]

    dose_bins_tmp = np.digitize(np.mean(EVAL_D_tmp, axis=1), np.linspace(0.0, 1.0, 6), right=False)
    strat_joint_tmp = T_tmp.astype(str) + "_" + dose_bins_tmp.astype(str)

    val_rel, te_rel, ok_joint_2 = _safe_stratified_split(X_tmp, strat_joint_tmp, round(10 / 37, 6), seed + 1)
    if not ok_joint_2:
        LOGGER.warning("[SPLIT] Second joint stratification failed. Falling back to T only.")
        val_rel, te_rel, ok_t2 = _safe_stratified_split(X_tmp, T_tmp, round(10 / 37, 6), seed + 1)
        if not ok_t2:
            rng = np.random.default_rng(seed + 1); idx = np.arange(len(X_tmp)); rng.shuffle(idx); cut = int(round(len(X_tmp) * (27 / 37)))
            val_rel, te_rel = idx[:cut], idx[cut:]

    val_idx = idx_tmp[val_rel]
    te_idx = idx_tmp[te_rel]

    X_val, T_val, Y_val, EVAL_Y_val, D_val, EVAL_D_val = X_tmp[val_rel], T_tmp[val_rel], Y_tmp[val_rel], EVAL_Y_tmp[val_rel], D_tmp[val_rel], EVAL_D_tmp[val_rel]
    X_te, T_te, Y_te, EVAL_Y_te, D_te, EVAL_D_te = X_tmp[te_rel], T_tmp[te_rel], Y_tmp[te_rel], EVAL_Y_tmp[te_rel], D_tmp[te_rel], EVAL_D_tmp[te_rel]

    yf_tr = Y_tr[np.arange(len(Y_tr)), T_tr]
    yf_val = Y_val[np.arange(len(Y_val)), T_val]
    yf_te = Y_te[np.arange(len(Y_te)), T_te]

    X_tr, X_val, X_te = standardize_splits(X_tr, X_val, X_te)
    yf_tr_norm, yf_val_norm, yf_te_norm, y_mean, y_std = standardize_y_factual(yf_tr, yf_val, yf_te)

    EVAL_Y_tr_norm = (EVAL_Y_tr - y_mean) / y_std

    prop = np.bincount(T, minlength=K).astype(np.float32); prop = prop / prop.sum()

    return {
        "X_tr": X_tr, "T_tr": T_tr, "D_tr": D_tr, "YF_tr": yf_tr_norm, "EVAL_Y_tr": EVAL_Y_tr_norm, "EVAL_D_tr": EVAL_D_tr,
        "X_val": X_val, "T_val": T_val, "D_val": D_val, "YF_val": yf_val_norm, "true_grid_val": true_grid_all[val_idx],
        "X_te": X_te, "T_te": T_te, "D_te": D_te, "YF_te": yf_te_norm, "true_grid_te": true_grid_all[te_idx],
        "y_mean": y_mean, "y_std": y_std, "K": K, "prop": prop, "val_idx": val_idx, "te_idx": te_idx,
    }

# ==============================================================================
# MODEL
# ==============================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)), nn.GELU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.GELU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim, latent_dim)), nn.LayerNorm(latent_dim),
        )
    def forward(self, x):
        return self.net(x)

class DoseAwareNet(nn.Module):
    def __init__(self, input_dim, num_treatments, hidden_dim=256, latent_dim=128, dropout=0.15):
        super().__init__()
        self.num_treatments = num_treatments
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout)
        self.t_embed = nn.Embedding(num_treatments, latent_dim)
        self.dose_net = nn.Sequential(
            spectral_norm(nn.Linear(1, latent_dim // 2)), nn.GELU(),
            spectral_norm(nn.Linear(latent_dim // 2, latent_dim // 2)), nn.GELU(),
        )
        fusion_dim = latent_dim + latent_dim + latent_dim // 2
        self.outcome_net = nn.Sequential(
            spectral_norm(nn.Linear(fusion_dim, hidden_dim)), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)), nn.GELU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim // 2, 1)),
        )
        self.dose_reg_head = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, latent_dim // 2)), nn.GELU(),
            spectral_norm(nn.Linear(latent_dim // 2, 1)), nn.Sigmoid(),
        )

    def forward(self, x, a, d):
        z = self.encoder(x)
        a_emb = self.t_embed(a)
        d_feat = self.dose_net(d.unsqueeze(1))
        y = self.outcome_net(torch.cat([z, a_emb, d_feat], dim=1))
        d_hat = self.dose_reg_head(z)
        return z, y, d_hat

    @torch.no_grad()
    def predict_all_treatments_at_eval_d(self, x: torch.Tensor, eval_d: torch.Tensor) -> torch.Tensor:
        self.eval()
        B = x.shape[0]
        preds = []
        for k in range(self.num_treatments):
            ak = torch.full((B,), k, dtype=torch.long, device=x.device)
            dk = eval_d[:, k] if eval_d.shape[1] == self.num_treatments else eval_d[:, k - 1]
            _, yk, _ = self.forward(x, ak, dk)
            preds.append(yk.squeeze(1))
        return torch.stack(preds, dim=1)

class TreatmentClassifier(nn.Module):
    def __init__(self, latent_dim, num_treatments, dropout=0.15):
        super().__init__()
        out_dim = 1 if num_treatments == 2 else num_treatments
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 128)), nn.GELU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, 64)), nn.GELU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(64, out_dim)),
        )
    def forward(self, z):
        return self.net(z)

# ==============================================================================
# PAIRING
# ==============================================================================
def build_effect_vectors_from_eval_y(eval_y_norm: np.ndarray) -> np.ndarray:
    return eval_y_norm.astype(np.float32)

def compute_effect_threshold(effect_vectors: np.ndarray, perc: float = 30.0, sample: int = 50000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed); N = effect_vectors.shape[0]
    if N < 2:
        return 0.1
    m = min(sample, N)
    idx1 = rng.integers(0, N, size=m); idx2 = rng.integers(0, N, size=m)
    dists = np.linalg.norm(effect_vectors[idx1] - effect_vectors[idx2], axis=1)
    thr = float(np.percentile(dists, perc))
    g = max(float(np.std(effect_vectors)), 1e-6)
    return float(np.clip(thr, max(1e-4, 0.05 * g), max(2.0 * g, 1e-4)))

def _empty_pair_batch(X, T, D, YF):
    empty_shape = (0, X.shape[1])
    return (np.zeros(empty_shape, dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32), np.zeros(empty_shape, dtype=np.float32), np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.array([], dtype=np.int64))

def make_pairs_random(X, T, Df, YF, n_pairs, seed=None):
    rng = np.random.default_rng(seed); N = X.shape[0]
    if N < 2:
        return _empty_pair_batch(X, T, Df, YF)
    idx_a = rng.integers(0, N, size=n_pairs); idx_b = rng.integers(0, N - 1, size=n_pairs); idx_b = np.where(idx_b >= idx_a, idx_b + 1, idx_b)
    labels = rng.integers(0, 2, size=n_pairs).astype(np.int64)
    return (X[idx_a], T[idx_a], Df[idx_a], YF[idx_a], X[idx_b], T[idx_b], Df[idx_b], YF[idx_b], labels)

def make_pairs_from_effect_vectors(X, T, Df, YF, effect_vectors, thr, n_pairs, seed=None):
    rng = np.random.default_rng(seed); N = effect_vectors.shape[0]
    if N < 2:
        return _empty_pair_batch(X, T, Df, YF)
    n_pairs = int(min(max(1, n_pairs), N)); half = n_pairs // 2; used = set(); sim_pairs = []; dis_pairs = []
    def add_pair(i, j, label, container):
        if i == j: return
        key = (min(i, j), max(i, j))
        if key in used: return
        used.add(key); container.append((i, j, label))
    attempts = 0; max_attempts = max(50, n_pairs * 10)
    while len(sim_pairs) < half and attempts < max_attempts:
        i = int(rng.integers(0, N)); dists = np.linalg.norm(effect_vectors - effect_vectors[i], axis=1)
        cand = np.where((dists < thr) & (np.arange(N) != i))[0]
        if cand.size > 0:
            add_pair(i, int(rng.choice(cand)), 1, sim_pairs)
        attempts += 1
    attempts = 0
    while len(dis_pairs) < (n_pairs - len(sim_pairs)) and attempts < max_attempts:
        i = int(rng.integers(0, N)); dists = np.linalg.norm(effect_vectors - effect_vectors[i], axis=1)
        cand = np.where((dists >= thr) & (np.arange(N) != i))[0]
        if cand.size > 0:
            add_pair(i, int(rng.choice(cand)), 0, dis_pairs)
        attempts += 1
    pairs = sim_pairs + dis_pairs
    if len(pairs) < max(1, n_pairs // 2):
        needed = n_pairs - len(pairs)
        for _ in range(needed):
            i = int(rng.integers(0, N)); j = int(rng.integers(0, N - 1))
            if j >= i: j += 1
            label = 1 if np.linalg.norm(effect_vectors[i] - effect_vectors[j]) < thr else 0
            add_pair(i, j, label, pairs)
    if not pairs:
        return _empty_pair_batch(X, T, Df, YF)
    rng.shuffle(pairs); idx_a, idx_b, labels = zip(*pairs)
    idx_a = np.array(idx_a); idx_b = np.array(idx_b); labels = np.array(labels, dtype=np.int64)
    return (X[idx_a], T[idx_a], Df[idx_a], YF[idx_a], X[idx_b], T[idx_b], Df[idx_b], YF[idx_b], labels)

def build_feature_knn_cache(X: np.ndarray, k: int = 20) -> Dict[str, np.ndarray]:
    N = X.shape[0]
    if N < 2:
        return {"knn_idx": np.zeros((N, 0), dtype=np.int64), "far_idx": np.zeros((N, 0), dtype=np.int64)}
    k_eff = int(max(1, min(k, N - 1)))
    nn_model = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean", algorithm="auto", n_jobs=-1)
    nn_model.fit(X); _, inds = nn_model.kneighbors(X, return_distance=True)
    knn_idx = inds[:, 1:].astype(np.int64)
    LOGGER.info(f"[KNN] Built feature KNN cache with k={k_eff} for X={X.shape}")
    return {"knn_idx": knn_idx, "far_idx": np.zeros((N, 0), dtype=np.int64)}

def make_pairs_feature_knn(X, T, Df, YF, n_pairs, k=20, seed=None, knn_cache=None):
    rng = np.random.default_rng(seed); N = X.shape[0]
    if N < 2: return _empty_pair_batch(X, T, Df, YF)
    if knn_cache is None or "knn_idx" not in knn_cache: raise ValueError("feature_knn mode requires a precomputed knn_cache.")
    knn_idx = knn_cache["knn_idx"]; n_pairs = int(min(max(1, n_pairs), N)); idx_a = rng.integers(0, N, size=n_pairs)
    idx_b = np.zeros(n_pairs, dtype=np.int64); labels = np.zeros(n_pairs, dtype=np.int64); half = n_pairs // 2; all_idx = np.arange(N)
    for p in range(n_pairs):
        i = int(idx_a[p]); neigh = knn_idx[i]
        if neigh.size == 0:
            j = int(rng.integers(0, N - 1));
            if j >= i: j += 1
            idx_b[p] = j; labels[p] = int(p < half); continue
        if p < half:
            j = int(rng.choice(neigh)); labels[p] = 1
        else:
            forbid = np.concatenate([neigh, np.array([i], dtype=np.int64)]); allowed_mask = np.ones(N, dtype=bool); allowed_mask[forbid] = False
            far_pool = all_idx[allowed_mask]; j = int(rng.choice(far_pool if far_pool.size > 0 else neigh)); labels[p] = 0
        idx_b[p] = j
    return (X[idx_a], T[idx_a], Df[idx_a], YF[idx_a], X[idx_b], T[idx_b], Df[idx_b], YF[idx_b], labels)

class DynamicTCGAPairDS(Dataset):
    def __init__(self, X_all, T_all, Df_all, YF_all, effect_vectors=None, bs=256, perc=30.0, seed=0,
                 pair_mode="dynamic_ite", feature_k=20, knn_cache=None):
        self.X_all = X_all; self.T_all = T_all; self.Df_all = Df_all; self.YF_all = YF_all; self.bs = int(bs)
        self.perc = float(perc); self.seed = int(seed); self.epoch = 0; self.pair_mode = pair_mode; self.feature_k = feature_k; self.knn_cache = knn_cache
        self.effect_vectors = np.zeros((X_all.shape[0], max(1, T_all.max()+1)), dtype=np.float32) if effect_vectors is None else effect_vectors.astype(np.float32)
        self.update_threshold()
    def set_epoch(self, epoch: int): self.epoch = int(epoch)
    def update_threshold(self): self.thr = compute_effect_threshold(self.effect_vectors, perc=self.perc, seed=self.seed + self.epoch)
    def update_effect_vectors(self, effect_vectors): self.effect_vectors = effect_vectors.astype(np.float32); self.update_threshold()
    def __len__(self): return int(np.ceil(self.X_all.shape[0] / self.bs))
    def __getitem__(self, idx: int):
        seed = (self.seed + 1000003 * self.epoch + 9176 * int(idx)) & 0xFFFFFFFF
        if self.pair_mode == "none":
            return tuple(torch.tensor(v) for v in _empty_pair_batch(self.X_all, self.T_all, self.Df_all, self.YF_all))
        if self.pair_mode in ["dynamic_ite", "static_ite"]:
            arr = make_pairs_from_effect_vectors(self.X_all, self.T_all, self.Df_all, self.YF_all, self.effect_vectors, self.thr, self.bs, seed=seed)
        elif self.pair_mode == "random":
            arr = make_pairs_random(self.X_all, self.T_all, self.Df_all, self.YF_all, self.bs, seed=seed)
        elif self.pair_mode == "feature_knn":
            arr = make_pairs_feature_knn(self.X_all, self.T_all, self.Df_all, self.YF_all, self.bs, k=self.feature_k, seed=seed, knn_cache=self.knn_cache)
        else:
            raise ValueError(f"Unknown pair_mode: {self.pair_mode}")
        x1,t1,d1,y1,x2,t2,d2,y2,lab = arr
        return (torch.tensor(x1, dtype=torch.float32), torch.tensor(t1, dtype=torch.long), torch.tensor(d1, dtype=torch.float32), torch.tensor(y1, dtype=torch.float32),
                torch.tensor(x2, dtype=torch.float32), torch.tensor(t2, dtype=torch.long), torch.tensor(d2, dtype=torch.float32), torch.tensor(y2, dtype=torch.float32),
                torch.tensor(lab, dtype=torch.long))

# ==============================================================================
# METRICS
# ==============================================================================
def predict_all_outcomes_at_eval_d(model: DoseAwareNet, X: np.ndarray, eval_d: np.ndarray, batch_size: int) -> torch.Tensor:
    loader_x = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    eval_d_t = torch.tensor(eval_d, dtype=torch.float32)
    all_preds = []; start = 0
    with torch.no_grad():
        for xb in loader_x:
            B = xb.shape[0]; xb = xb.to(DEVICE); ed_batch = eval_d_t[start:start + B].to(DEVICE)
            preds_b = model.predict_all_treatments_at_eval_d(xb, ed_batch); all_preds.append(preds_b); start += B
    return torch.cat(all_preds, dim=0)

def predict_response_curve_grid(model: DoseAwareNet, X: np.ndarray, dose_grid: np.ndarray, batch_size: int, y_mean: float, y_std: float) -> np.ndarray:
    loader_x = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    G = len(dose_grid); K = model.num_treatments; pred_chunks = []
    dose_grid_t = torch.tensor(dose_grid, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        for xb in loader_x:
            xb = xb.to(DEVICE); B = xb.shape[0]
            pred_b = torch.zeros((B, K, G), dtype=torch.float32, device=DEVICE)
            for k in range(K):
                a = torch.full((B,), k, dtype=torch.long, device=DEVICE)
                for g in range(G):
                    d = torch.full((B,), float(dose_grid_t[g]), dtype=torch.float32, device=DEVICE)
                    _, yhat, _ = model(xb, a, d)
                    pred_b[:, k, g] = yhat.squeeze(1) * y_std + y_mean
            pred_chunks.append(pred_b.cpu().numpy())
    return np.concatenate(pred_chunks, axis=0)

def compute_true_tcga_mise_from_grids(pred_grid: np.ndarray, true_grid: np.ndarray, dose_grid: np.ndarray) -> Dict[str, float]:
    sq_err = (pred_grid - true_grid) ** 2
    int_err = np.trapezoid(sq_err, x=dose_grid, axis=2)
    mise = float(np.mean(int_err))
    return {"mise": mise, "sqrt_mise": float(np.sqrt(mise))}

def compute_tcga_point_proxy_metrics(preds_denorm: torch.Tensor, eval_y_true: torch.Tensor, prop: np.ndarray) -> Dict[str, float]:
    K = preds_denorm.shape[1]
    point_mse = F.mse_loss(preds_denorm, eval_y_true).item()
    point_rmse = float(np.sqrt(point_mse))
    pred_eff = preds_denorm[:, 1:] - preds_denorm[:, [0]] if K > 1 else preds_denorm
    true_eff = eval_y_true[:, 1:] - eval_y_true[:, [0]] if K > 1 else eval_y_true
    if K > 1:
        p0 = float(prop[0])
        prop_new = torch.tensor([float(prop[k]) / (1.0 - p0 + 1e-8) for k in range(1, len(prop))], dtype=torch.float32, device=preds_denorm.device)
        mse_per_t = ((pred_eff - true_eff) ** 2).mean(dim=0)
        pehe_sqrt = float((mse_per_t @ prop_new).sqrt())
    else:
        pehe_sqrt = point_rmse
    mate_errors = []
    for i in range(K):
        for j in range(i + 1, K):
            mate_errors.append(torch.abs((preds_denorm[:, i] - preds_denorm[:, j]) - (eval_y_true[:, i] - eval_y_true[:, j])).mean().item())
    mate = float(np.mean(mate_errors)) if mate_errors else 0.0
    return {"point_mse": point_mse, "point_rmse": point_rmse, "pehe_sqrt": pehe_sqrt, "mate": mate}

# ==============================================================================
# TRAIN
# ==============================================================================
def train_single_run(split: Dict[str, Any], run_seed: int, hyperparams: Dict[str, Any], model_name: str, dose_grid: np.ndarray) -> Dict[str, float]:
    torch.manual_seed(run_seed); np.random.seed(run_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(run_seed)
    LR_MAIN = hyperparams.get("lr", 1e-3); BATCH_SIZE = hyperparams.get("batch_size", 256); LATENT_DIM = hyperparams.get("latent_dim", 128)
    HIDDEN_DIM = hyperparams.get("hidden_dim", 256); EPOCHS = hyperparams.get("epochs", 150); PATIENCE = hyperparams.get("patience", 25)
    MAIN_WD = hyperparams.get("main_weight_decay", 1e-4); CLF_WD = hyperparams.get("clf_weight_decay", 1e-5)
    LR_TREAT_CLF = hyperparams.get("lr_treat_clf", 3e-4); TREAT_CLF_STEPS = hyperparams.get("treat_clf_steps", 5); CLIP_NORM = hyperparams.get("clip_norm", 2.0)
    DOSE_LOSS_WEIGHT = hyperparams.get("dose_loss_weight", 0.05); ALPHA = hyperparams.get("alpha", 0.10); MARGIN = hyperparams.get("margin", 0.60)
    WARMUP_EPOCHS = hyperparams.get("warmup_epochs", 10); PAIR_UPDATE_FREQ = hyperparams.get("pair_update_freq", 5); PERC_THR = hyperparams.get("pair_threshold_percentile", 30.0)
    LAMBDA_MI_POS = hyperparams.get("lambda_mi_pos", 0.05); MI_START_EPOCH = hyperparams.get("mi_start_epoch", 20); POS_MIN_COUNT = hyperparams.get("mi_pos_min_count", 12)
    ENCODER_DROPOUT = hyperparams.get("encoder_dropout", 0.15); CLF_DROPOUT = hyperparams.get("clf_dropout", 0.15)
    USE_CONTRASTIVE = hyperparams.get("use_contrastive", True); USE_LOCAL_MI = hyperparams.get("use_local_mi", True); USE_DYNAMIC_UPDATE = hyperparams.get("use_dynamic_update", True)
    PAIR_MODE = hyperparams.get("pair_mode", "dynamic_ite"); FEATURE_K = hyperparams.get("feature_k", 20)
    MI_MODE = hyperparams.get("mi_mode", "local")
    if MI_MODE not in {"local", "global"}:
        raise ValueError(f"Unknown mi_mode: {MI_MODE}. Expected 'local' or 'global'.")

    X_tr = split["X_tr"]; T_tr = split["T_tr"]; D_tr = split["D_tr"]; YF_tr = split["YF_tr"]
    X_val = split["X_val"]; true_grid_val = split["true_grid_val"]
    X_te = split["X_te"]; T_te = split["T_te"]; D_te = split["D_te"]; YF_te = split["YF_te"]; true_grid_te = split["true_grid_te"]
    y_mean = split["y_mean"]; y_std = split["y_std"]; num_treatments = split["K"]

    Df_tr = np.zeros(len(T_tr), dtype=np.float32)
    active_mask_tr = T_tr > 0
    Df_tr[active_mask_tr] = D_tr[np.arange(len(T_tr))[active_mask_tr], T_tr[active_mask_tr] - 1]

    Df_te = np.zeros(len(T_te), dtype=np.float32)
    active_mask_te = T_te > 0
    Df_te[active_mask_te] = D_te[np.arange(len(T_te))[active_mask_te], T_te[active_mask_te] - 1]

    initial_effect_vectors = np.zeros((X_tr.shape[0], num_treatments), dtype=np.float32)
    feature_knn_cache = build_feature_knn_cache(X_tr, k=FEATURE_K) if PAIR_MODE == "feature_knn" else None

    ds_train = DynamicTCGAPairDS(X_all=X_tr, T_all=T_tr, Df_all=Df_tr, YF_all=YF_tr,
                                 effect_vectors=initial_effect_vectors if PAIR_MODE in ["dynamic_ite","static_ite"] else None,
                                 bs=BATCH_SIZE, perc=PERC_THR, seed=run_seed, pair_mode=PAIR_MODE, feature_k=FEATURE_K, knn_cache=feature_knn_cache)
    dl_train = DataLoader(ds_train, batch_size=None, shuffle=True)

    model = DoseAwareNet(input_dim=X_tr.shape[1], num_treatments=num_treatments, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, dropout=ENCODER_DROPOUT).to(DEVICE)
    treat_clf = TreatmentClassifier(LATENT_DIM, num_treatments=num_treatments, dropout=CLF_DROPOUT).to(DEVICE)
    opt_main = optim.AdamW(model.parameters(), lr=LR_MAIN, weight_decay=MAIN_WD)
    opt_treat_clf = optim.AdamW(treat_clf.parameters(), lr=LR_TREAT_CLF, weight_decay=CLF_WD)
    early_stopper = EarlyStoppingMetric(patience=PATIENCE)
    static_pair_initialized = False

    for epoch in range(EPOCHS):
        ds_train.set_epoch(epoch)
        ds_train.pair_mode = "random" if (PAIR_MODE in ["dynamic_ite","static_ite"] and epoch < WARMUP_EPOCHS) else PAIR_MODE
        lambda_ctr = 0.0 if (epoch < WARMUP_EPOCHS or not USE_CONTRASTIVE or PAIR_MODE == "none") else ALPHA
        lambda_mi = 0.0 if (epoch < MI_START_EPOCH or not USE_LOCAL_MI or PAIR_MODE == "none") else LAMBDA_MI_POS
        model.train(); treat_clf.train()
        epoch_loss_y = epoch_loss_d = epoch_loss_ctr = epoch_loss_mi = 0.0
        epoch_pos_count = epoch_total_pairs = ctr_active_batches = mi_active_batches = n_batches_seen = 0

        for batch in dl_train:
            x1, t1, d1, y1, x2, t2, d2, y2, label = [b.to(DEVICE) for b in batch]
            if x1.shape[0] == 0: continue
            n_batches_seen += 1
            y1 = y1.view(-1,1); y2 = y2.view(-1,1); label = label.float()
            z1, yhat1, d_hat1 = model(x1, t1, d1); z2, yhat2, d_hat2 = model(x2, t2, d2)
            loss_y = 0.5 * (F.mse_loss(yhat1,y1) + F.mse_loss(yhat2,y2))
            loss_d = 0.5 * (F.mse_loss(d_hat1, d1.unsqueeze(1)) + F.mse_loss(d_hat2, d2.unsqueeze(1)))
            pos_mask = (label == 1)

            # --------------------------------------------------------------
            # MI classifier update.
            # local  -> use only positive/effect-homogeneous pairs.
            # global -> use all representations in the current paired batch.
            # --------------------------------------------------------------
            mi_active = False
            if lambda_mi > 0:
                if MI_MODE == "local":
                    enough_mi_samples = pos_mask.sum().item() > POS_MIN_COUNT
                    if enough_mi_samples:
                        with torch.no_grad():
                            z_mi_det = torch.cat([z1[pos_mask], z2[pos_mask]], dim=0).detach()
                            t_mi_det = torch.cat([t1[pos_mask], t2[pos_mask]], dim=0)
                    else:
                        z_mi_det, t_mi_det = None, None
                elif MI_MODE == "global":
                    enough_mi_samples = x1.shape[0] > POS_MIN_COUNT
                    if enough_mi_samples:
                        with torch.no_grad():
                            z_mi_det = torch.cat([z1, z2], dim=0).detach()
                            t_mi_det = torch.cat([t1, t2], dim=0)
                    else:
                        z_mi_det, t_mi_det = None, None
                else:
                    raise ValueError(f"Unknown mi_mode: {MI_MODE}")

                if enough_mi_samples:
                    mi_active = True
                    for _ in range(TREAT_CLF_STEPS):
                        opt_treat_clf.zero_grad()
                        clf_loss = treatment_classifier_loss(treat_clf, z_mi_det, t_mi_det, num_treatments)
                        clf_loss.backward()
                        torch.nn.utils.clip_grad_norm_(treat_clf.parameters(), CLIP_NORM)
                        opt_treat_clf.step()

            for p in treat_clf.parameters(): p.requires_grad = False
            loss_ctr = contrastive_loss(z1, z2, label, margin=MARGIN) if lambda_ctr > 0 else torch.tensor(0.0, device=DEVICE)
            loss_mi = torch.tensor(0.0, device=DEVICE)

            if lambda_mi > 0:
                if MI_MODE == "local":
                    enough_mi_samples = pos_mask.sum().item() > POS_MIN_COUNT
                    if enough_mi_samples:
                        z_mi = torch.cat([z1[pos_mask], z2[pos_mask]], dim=0)
                        t_mi = torch.cat([t1[pos_mask], t2[pos_mask]], dim=0)
                elif MI_MODE == "global":
                    enough_mi_samples = x1.shape[0] > POS_MIN_COUNT
                    if enough_mi_samples:
                        z_mi = torch.cat([z1, z2], dim=0)
                        t_mi = torch.cat([t1, t2], dim=0)
                else:
                    raise ValueError(f"Unknown mi_mode: {MI_MODE}")

                if enough_mi_samples:
                    mi_lb = variational_mi_lower_bound(treat_clf, z_mi, t_mi, num_treatments)
                    loss_mi = torch.clamp(mi_lb, -5.0, 5.0)

            loss_main = loss_y + DOSE_LOSS_WEIGHT * loss_d + lambda_ctr * loss_ctr + lambda_mi * loss_mi
            epoch_loss_y += float(loss_y.item()); epoch_loss_d += float(loss_d.item()); epoch_loss_ctr += float(loss_ctr.item()); epoch_loss_mi += float(loss_mi.item())
            epoch_pos_count += int(pos_mask.sum().item()); epoch_total_pairs += int(label.numel())
            if lambda_ctr > 0: ctr_active_batches += 1
            if mi_active: mi_active_batches += 1

            opt_main.zero_grad(); loss_main.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM); opt_main.step()
            for p in treat_clf.parameters(): p.requires_grad = True

        should_bootstrap_pairs = (epoch == WARMUP_EPOCHS - 1)
        should_update_pairs = (epoch >= WARMUP_EPOCHS and PAIR_UPDATE_FREQ > 0 and ((epoch - WARMUP_EPOCHS) % PAIR_UPDATE_FREQ == 0))
        if PAIR_MODE in ["dynamic_ite","static_ite"] and (should_bootstrap_pairs or should_update_pairs):
            with torch.no_grad():
                preds_tr = predict_all_outcomes_at_eval_d(model, X_tr, split["D_tr"], batch_size=min(1024, BATCH_SIZE * 4))
                new_effect_vectors = build_effect_vectors_from_eval_y(preds_tr.cpu().numpy())
                if PAIR_MODE == "dynamic_ite" and USE_DYNAMIC_UPDATE:
                    ds_train.update_effect_vectors(new_effect_vectors)
                elif PAIR_MODE == "static_ite" and not static_pair_initialized:
                    ds_train.update_effect_vectors(new_effect_vectors); static_pair_initialized = True

        if n_batches_seen > 0:
            LOGGER.info(f"[TRAIN][{model_name}] ep={epoch:03d} pair_mode={ds_train.pair_mode} ly={epoch_loss_y / n_batches_seen:.6f} ld={epoch_loss_d / n_batches_seen:.6f} lctr={epoch_loss_ctr / n_batches_seen:.6f} lmi={epoch_loss_mi / n_batches_seen:.6f} pos_rate={epoch_pos_count / max(1, epoch_total_pairs):.4f} ctr_batches={ctr_active_batches}/{n_batches_seen} mi_batches={mi_active_batches}/{n_batches_seen} lambda_ctr={lambda_ctr:.4f} lambda_mi={lambda_mi:.4f}")

        model.eval()
        pred_grid_val = predict_response_curve_grid(model, X_val, dose_grid, batch_size=min(256, BATCH_SIZE), y_mean=y_mean, y_std=y_std)
        val_metrics = compute_true_tcga_mise_from_grids(pred_grid_val, true_grid_val, dose_grid)
        val_metric = val_metrics["sqrt_mise"]
        LOGGER.info(f"[VAL][{model_name}] ep={epoch:03d} true_sqrt_MISE={val_metric:.6f}")
        early_stopper(val_metric, model)
        if early_stopper.early_stop:
            break

    early_stopper.restore(model); model.eval()

    ds_test = torch.utils.data.TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(T_te, dtype=torch.long), torch.tensor(Df_te, dtype=torch.float32), torch.tensor(YF_te, dtype=torch.float32).unsqueeze(1))
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
    test_losses_norm = []
    with torch.no_grad():
        for xb, tb, db, yb in dl_test:
            xb = xb.to(DEVICE); tb = tb.to(DEVICE); db = db.to(DEVICE); yb = yb.to(DEVICE)
            _, y_hat, _ = model(xb, tb, db); test_losses_norm.append(F.mse_loss(y_hat, yb).item())
    test_factual_rmse_norm = float(np.sqrt(np.mean(test_losses_norm)))

    pred_grid_te = predict_response_curve_grid(model, X_te, dose_grid, batch_size=min(256, BATCH_SIZE), y_mean=y_mean, y_std=y_std)
    true_mise_metrics = compute_true_tcga_mise_from_grids(pred_grid_te, true_grid_te, dose_grid)
    preds_te_proxy = predict_all_outcomes_at_eval_d(model, X_te, D_te, batch_size=min(1024, BATCH_SIZE*4))
    preds_te_denorm = preds_te_proxy * y_std + y_mean
    eval_y_te_t = torch.tensor(split.get("proxy_eval_y_te", np.zeros((len(X_te), num_treatments), dtype=np.float32)), dtype=torch.float32, device=DEVICE)
    # placeholder proxy if needed
    proxy_metrics = {"point_mse": np.nan, "point_rmse": np.nan, "pehe_sqrt": np.nan, "mate": np.nan}

    return {
        "model": model_name,
        "best_val_sqrt_mise": early_stopper.best_value,
        "test_factual_rmse_norm": test_factual_rmse_norm,
        "mise": true_mise_metrics["mise"],
        "sqrt_mise": true_mise_metrics["sqrt_mise"],
        "point_mse": proxy_metrics["point_mse"],
        "point_rmse": proxy_metrics["point_rmse"],
        "pehe_sqrt": proxy_metrics["pehe_sqrt"],
        "mate": proxy_metrics["mate"],
        "epochs": epoch + 1,
    }

# ==============================================================================
# EXPERIMENT RUNNERS
# ==============================================================================
def build_component_ablation_configs(best_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    configs = {}
    configs["full_model"] = copy.deepcopy(best_params)
    p = copy.deepcopy(best_params); p["use_contrastive"] = False; configs["no_contrastive"] = p
    p = copy.deepcopy(best_params); p["use_local_mi"] = False; configs["no_local_mi"] = p
    p = copy.deepcopy(best_params); p["use_contrastive"] = False; p["use_local_mi"] = False; p["pair_mode"] = "none"; configs["supervised_only"] = p
    return configs

def build_pairing_ablation_configs(best_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    configs = {}
    p = copy.deepcopy(best_params); p["pair_mode"] = "dynamic_ite"; configs["pair_dynamic_ite"] = p
    p = copy.deepcopy(best_params); p["pair_mode"] = "static_ite"; p["use_dynamic_update"] = False; configs["pair_static_ite"] = p
    p = copy.deepcopy(best_params); p["pair_mode"] = "random"; configs["pair_random"] = p
    p = copy.deepcopy(best_params); p["pair_mode"] = "feature_knn"; configs["pair_feature_knn"] = p
    p = copy.deepcopy(best_params); p["pair_mode"] = "none"; p["use_contrastive"] = False; p["use_local_mi"] = False; configs["pair_none_supervised"] = p
    return configs

def build_mi_ablation_configs(best_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    TCGA MI-mode ablations for the table section "MI Mode".

    Returned settings correspond to:
    - Local MI w/o Schedule
    - Global MI
    - Local MI Only
    - Local MI w/ Random Pairs

    Note: the full scheduled local-MI model is already included in the
    component ablation group as "full_model". It is added here too as
    "mi_local_scheduled_reference" for direct ranking within this group.
    """
    configs = {}

    p = copy.deepcopy(best_params)
    p["mi_mode"] = "local"
    configs["mi_local_scheduled_reference"] = p

    p = copy.deepcopy(best_params)
    p["mi_mode"] = "local"
    p["mi_start_epoch"] = 0
    configs["mi_local_no_schedule"] = p

    p = copy.deepcopy(best_params)
    p["mi_mode"] = "global"
    p["use_local_mi"] = True
    configs["mi_global"] = p

    p = copy.deepcopy(best_params)
    p["use_contrastive"] = False
    p["use_local_mi"] = True
    p["mi_mode"] = "local"
    p["pair_mode"] = "dynamic_ite"
    configs["mi_local_only"] = p

    p = copy.deepcopy(best_params)
    p["use_contrastive"] = True
    p["use_local_mi"] = True
    p["mi_mode"] = "local"
    p["pair_mode"] = "random"
    configs["mi_local_random_pairs"] = p

    return configs

def build_sensitivity_configs(best_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    configs = {"best_reference": copy.deepcopy(best_params)}
    sensitivity_grid = {
        "alpha": [0.05, best_params["alpha"], 0.20],
        "margin": [0.40, best_params["margin"], 0.90],
        "pair_threshold_percentile": [20.0, best_params["pair_threshold_percentile"], 40.0],
        "lambda_mi_pos": [0.02, best_params["lambda_mi_pos"], 0.10],
        "pair_update_freq": [1, best_params["pair_update_freq"], 10],
        "mi_start_epoch": [10, best_params["mi_start_epoch"], 40],
        "warmup_epochs": [5, best_params["warmup_epochs"], 20],
    }
    for param_name, values in sensitivity_grid.items():
        for v in values:
            cfg = copy.deepcopy(best_params); cfg[param_name] = v
            configs[f"sens_{param_name}_{str(v).replace('.', 'p')}"] = cfg
    return configs

def save_experiment_results(experiment_name: str, df_all: pd.DataFrame, df_agg: pd.DataFrame):
    per_sim_path = os.path.join(OUT_DIR, f"{experiment_name}_per_run.csv")
    agg_path = os.path.join(OUT_DIR, f"{experiment_name}_aggregate.csv")
    rank_path = os.path.join(OUT_DIR, f"{experiment_name}_ranking.csv")
    df_all.to_csv(per_sim_path, index=False, sep=";")
    df_agg = df_agg.sort_values(by=["mean_sqrt_mise"], ascending=[True]).reset_index(drop=True)
    df_agg.to_csv(agg_path, index=False, sep=";"); df_agg.to_csv(rank_path, index=False, sep=";")
    print(f"\nSaved per-run results to: {per_sim_path}"); print(f"Saved aggregate results to: {agg_path}"); print(f"Saved ranking to: {rank_path}")

def run_setting(setting_name: str, params: Dict[str, Any], bundle: Dict[str, np.ndarray], true_grid_all: np.ndarray, dose_grid: np.ndarray, n_runs: int):
    print("\n" + "=" * 80); print(f"RUNNING SETTING: {setting_name}"); print("=" * 80)
    results_storage = []
    for run_id in range(n_runs):
        split = split_tcga_bundle(bundle, seed=100 + run_id, true_grid_all=true_grid_all)
        res = train_single_run(split, run_seed=1000 + run_id, hyperparams=params, model_name=setting_name, dose_grid=dose_grid)
        row = {
            "setting": setting_name, "run_id": run_id, "mise": res["mise"], "sqrt_mise": res["sqrt_mise"],
            "test_factual_rmse_norm": res["test_factual_rmse_norm"], "epochs": res["epochs"], "best_val_sqrt_mise": res["best_val_sqrt_mise"],
        }
        for k, v in params.items(): row[f"param_{k}"] = v
        results_storage.append(row)
        print(f"[{setting_name}] [Run {run_id + 1}/{n_runs}] true_sqrt_MISE: {res['sqrt_mise']:.4f} | Ep: {res['epochs']}", flush=True)
    df = pd.DataFrame(results_storage)
    agg = pd.DataFrame([{
        "setting": setting_name, "n_runs": n_runs, "mean_mise": df["mise"].mean(), "std_mise": df["mise"].std(),
        "mean_sqrt_mise": df["sqrt_mise"].mean(), "std_sqrt_mise": df["sqrt_mise"].std(),
        "mean_factual_rmse_norm": df["test_factual_rmse_norm"].mean(), "std_factual_rmse_norm": df["test_factual_rmse_norm"].std(),
        "mean_epochs": df["epochs"].mean(),
    }])
    return df, agg

def run_experiment_group(experiment_name: str, configs: Dict[str, Dict[str, Any]], bundle: Dict[str, np.ndarray], true_grid_all: np.ndarray, dose_grid: np.ndarray, n_runs: int):
    all_rows = []; all_agg = []; failed_settings = []
    for setting_name, params in configs.items():
        try:
            df, agg = run_setting(setting_name, params, bundle, true_grid_all, dose_grid, n_runs)
            all_rows.append(df); all_agg.append(agg)
        except Exception as e:
            LOGGER.exception(f"[EXPERIMENT] Setting failed: {setting_name}")
            failed_settings.append({"setting": setting_name, "error": repr(e)})
    if not all_rows:
        raise RuntimeError(f"All settings failed for experiment group: {experiment_name}")
    df_all = pd.concat(all_rows, axis=0, ignore_index=True); df_agg = pd.concat(all_agg, axis=0, ignore_index=True)
    if failed_settings:
        fail_path = os.path.join(OUT_DIR, f"{experiment_name}_failed_settings.csv")
        pd.DataFrame(failed_settings).to_csv(fail_path, index=False, sep=";")
        print(f"Saved failed settings log to: {fail_path}")
    save_experiment_results(experiment_name, df_all, df_agg)
    print("\n" + "=" * 80); print(f"SUMMARY: {experiment_name}"); print("=" * 80)
    print(df_agg.sort_values(by=["mean_sqrt_mise"]).to_string(index=False))

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    bundle = load_tcga_mitnet_npz(NPZ_PATH)
    _, dose_grid, true_grid_all = build_benchmark_and_true_grid_all(
        data_dir=TCGA_DATA_DIR, tcga_num_features=bundle["X"].shape[1], num_treatments=bundle["K"],
        dose_grid_size=DOSE_GRID_SIZE, batch_size=BENCHMARK_BATCH_SIZE, seed=BENCHMARK_SEED,
        strength_of_assignment_bias=BENCHMARK_ASSIGNMENT_BIAS, total_n=bundle["X"].shape[0],
    )
    component_configs = build_component_ablation_configs(BEST_PARAMS)
    pairing_configs = build_pairing_ablation_configs(BEST_PARAMS)
    mi_configs = build_mi_ablation_configs(BEST_PARAMS)
    sensitivity_configs = build_sensitivity_configs(BEST_PARAMS)

    run_experiment_group("component_ablation_true_mise", component_configs, bundle, true_grid_all, dose_grid, N_RUNS)
    run_experiment_group("pairing_ablation_true_mise", pairing_configs, bundle, true_grid_all, dose_grid, N_RUNS)
    run_experiment_group("mi_ablation_true_mise", mi_configs, bundle, true_grid_all, dose_grid, N_RUNS)
    run_experiment_group("sensitivity_ablation_true_mise", sensitivity_configs, bundle, true_grid_all, dose_grid, N_RUNS)

if __name__ == "__main__":
    main()
