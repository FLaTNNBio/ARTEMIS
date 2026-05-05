import os
import io
import sqlite3
import datetime
import argparse
from itertools import cycle
from typing import Tuple, List, Optional

import numpy as np
from pandas import read_csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit


# =========================
# Minimal utility functions
# =========================
LAST_ROW_ID = None
LAST_ID_SET = None


def log(*args):
    print(*args, flush=True)


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


# =========================
# Minimal base classes
# =========================
class BaseDataAccess:
    def get_split_indices(self):
        return (None, None)

    def make_propensity_lists(self, *args, **kwargs):
        return None


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


# =========================
# SQLite array adapters
# =========================
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
    TABLE_METHYLATION = "methylation"
    TABLE_SNP = "snp"

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
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS methylation ("
            "id TEXT NOT NULL PRIMARY KEY, data ARRAY, clinical_id TEXT NOT NULL, "
            "FOREIGN KEY(clinical_id) REFERENCES clinical(id));"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS snp ("
            "id TEXT NOT NULL PRIMARY KEY, data ARRAY, clinical_id TEXT NOT NULL, "
            "FOREIGN KEY(clinical_id) REFERENCES clinical(id));"
        )
        self.db.commit()

    def get_row(self, table_name, id, with_rowid=False):
        columns = "*"
        if with_rowid:
            columns = "rowid, " + columns
        query = f"SELECT {columns} FROM {table_name} WHERE rowid = ?;"

        if isinstance(id, tuple):
            id = id[0]

        id = int(id)  # <-- fix importante

        return self.db.execute(query, (id,)).fetchone()

    def get_rows_by_clinical_id(self, table_name, id, with_rowid=False):
        columns = "*"
        if with_rowid:
            columns = "rowid, " + columns
        query = f"SELECT {columns} FROM {table_name} WHERE clinical_id = ?;"
        return self.db.execute(query, (id,)).fetchone()

    def get_entry_with_id(self, id, args={"with_rnaseq": True}):
        with_rnaseq = args["with_rnaseq"]

        if isinstance(id, tuple):
            id = id[0]
        id = int(id)  # <-- fix importante

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
            "SELECT clinical.rowid FROM clinical "
            "WHERE clinical.id IN (SELECT clinical_id FROM rnaseq) ORDER BY clinical.rowid;"
        ).fetchall()
        return np.squeeze(return_value)

    def get_rnaseq_dimension(self):
        rnaseq = self.db.execute("SELECT data FROM rnaseq WHERE rowid = 1;").fetchone()[0]
        return rnaseq.shape[0]

    @staticmethod
    def binarize_days(days):
        days_copy = np.copy(days)
        days_copy[days_copy > 0] = 1
        days_copy[days_copy <= 0] = 0
        return days_copy

    def get_column(self, table_name, ids, column_name):
        ids = list(ids)
        tmp_name = "tmp_ids"
        self.db.execute(f"CREATE TEMP TABLE {tmp_name} (id INT);")
        if len(ids) != 0:
            self.db.executemany(f"INSERT INTO {tmp_name} VALUES (?);", ids)
        return_value = self.db.execute(
            f"SELECT {column_name} FROM {table_name} WHERE rowid IN (SELECT id FROM {tmp_name}) ORDER BY rowid;"
        ).fetchall()
        self.db.execute(f"DROP TABLE {tmp_name};")
        return np.squeeze(return_value)

    def get_dataset_names(self, ids):
        return self.get_column(DataAccess.TABLE_CLINICAL, ids, "dataset_name")

    def get_days_to_death(self, ids):
        return self.get_column(DataAccess.TABLE_CLINICAL, ids, "days_to_death")

    def get_days_to_recurrence(self, ids):
        return self.get_column(DataAccess.TABLE_CLINICAL, ids, "days_to_recurrence")

    def get_days_to_surgery(self, ids):
        return self.get_column(DataAccess.TABLE_CLINICAL, ids, "days_to_surgery")

    def get_did_radiation_therapy(self, ids):
        return self.get_column(DataAccess.TABLE_CLINICAL, ids, "did_radiation_therapy")

    def get_labels(self, args, patients, benchmark):
        tcga_num_features = int(np.rint(args["tcga_num_features"]))
        assignments = []

        for id in patients:
            pid = int(id[0]) if isinstance(id, tuple) else int(id)

            entry = self.get_entry_with_id(pid, {"with_rnaseq": True})[1]
            rnaseq_data = np.array(entry["rnaseq"][1])
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

        patient_ids = np.array([x["clinical"][0] for x in batch_data])
        rnaseq_data = np.array([x["rnaseq"][1] for x in batch_data])
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
    def __init__(self, data_dir, num_treatments=2, num_centroids_mean=7, num_centroids_std=2,
                 num_relevant_gene_loci_mean=10, num_relevant_gene_loci_std=3, response_mean_of_mean=0.45,
                 response_std_of_mean=0.15, response_mean_of_std=0.1, response_std_of_std=0.05,
                 strength_of_assignment_bias=10, epsilon_std=0.15, with_exposure=True, **kwargs):
        super(TCGABenchmark, self).__init__(DataAccess(data_dir, **kwargs), num_treatments, **kwargs)
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
        self.selected_features = None

    def get_scaling_constant(self):
        return self.scaling_constant

    def has_exposure(self):
        return self.with_exposure

    def initialise(self, args):
        self.random_generator = np.random.RandomState(909)
        self.centroids = None
        all_features = self.data_access.get_rnaseq_dimension()
        if self.num_features > 0 and self.num_features != all_features:
            self.selected_features = self.random_generator.choice(all_features, self.num_features, replace=False)
        else:
            self.selected_features = np.arange(all_features)

    def select_features(self, x):
        return x[:, self.selected_features] if x.ndim == 2 else x[self.selected_features]

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
                    # Last treatment is control = worse expected outcomes.
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

                rnaseq_data = \
                self.data_access.get_entry_with_id(int(ids[local_idx]), {"with_rnaseq": True})[1]["rnaseq"][1]

                centroid_data = (
                    gene_loci_indices,
                    rnaseq_data[gene_loci_indices],
                    response_mean,
                    response_std
                )
                centroids_tmp.append(centroid_data)

            current_idx += batch_size

        return centroids_tmp

    def fit(self, generator, steps, batch_size):
        num_samples = steps * batch_size
        centroid_indices = sorted(self.random_generator.permutation(num_samples)[:self.num_treatments + 1])
        if self.with_exposure:
            self.dosage_centroids = []
            for treatment_idx in range(self.num_treatments):
                dosage_centroid_indices = sorted(self.random_generator.permutation(num_samples)[:self.num_archetypes_per_treatment])
                self.dosage_centroids.append(self.get_from_generator_with_offsets(generator, dosage_centroid_indices))
                for dosage_idx in range(self.num_archetypes_per_treatment):
                    min_response = self.random_generator.normal(0.0, 0.1)
                    self.dosage_centroids[treatment_idx][dosage_idx] += (min_response,)
        self.centroids = self.get_from_generator_with_offsets(generator, centroid_indices, adjust_last=True)
        self.assignment_cache = {}

    def get_centroid_weights(self, x, centroids=None):
        if centroids is None:
            centroids = self.centroids
        similarities = [
            cosine_similarity(x[indices].reshape(1, -1), centroid.reshape(1, -1))[0, 0]
            for indices, centroid, *_ in centroids
        ]
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
        return dose_response_curve

    def _assign(self, x):
        assert self.centroids is not None, "Must call fit before assign."
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
                dose_response_curve = self.get_dose_response_curve(x, treatment_idx)
                treatment_strength = clip_percentage(self.random_generator.normal(0.65, 0.1))
                treatment_strengths.append(treatment_strength)
                this_y = dose_response_curve(treatment_strength)
                y.append(this_y * expected_responses[treatment_idx])
            treatment_strengths = np.array(treatment_strengths)
        else:
            raise NotImplementedError("Standalone evaluator only supports with_exposure=True")
        y = np.array(y)
        treatment_chosen = self.random_generator.choice(self.num_treatments, p=stable_softmax(self.strength_of_assignment_bias * y))
        return treatment_chosen, self.scaling_constant * y, treatment_strengths

    def get_assignment(self, id, x):
        if self.centroids is None:
            return 0, 0, 0
        if id not in self.assignment_cache:
            entry = self.data_access.get_entry_with_id(id, {"with_rnaseq": True})[1]
            rnaseq_data = np.array(entry["rnaseq"][1])
            rnaseq_data = (rnaseq_data - self.data_access.min_val) / (self.data_access.max_val - self.data_access.min_val + 1e-5)
            if self.num_features > 0:
                rnaseq_data = rnaseq_data[:self.num_features]
            values = self._assign(rnaseq_data)
            self.assignment_cache[id] = values
        assigned_treatment, assigned_y, treatment_strength = self.assignment_cache[id]
        if self.assign_counterfactuals:
            return assigned_treatment, assigned_y, treatment_strength
        return assigned_treatment, assigned_y[assigned_treatment], treatment_strength[assigned_treatment]


# =========================
# Generators
# =========================
def report_distribution(data, labels, num_classes, set_name):
    counts = np.zeros((num_classes,))
    for i in range(num_classes):
        counts[i] = np.sum(labels == i) / float(len(labels))
    log("INFO: Using", set_name, "set (n=", len(data), ") with distribution", counts)


def make_generator(args, benchmark, is_validation=False, is_test=False,
                   validation_fraction=0.2, test_fraction=0.2, seed=909, randomise=True,
                   stratify=True, resample_with_replacement=False):
    fraction_of_data_set = args["fraction_of_data_set"]
    patients = benchmark.get_data_access().get_labelled_patients()
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
            patient_id, result = benchmark.get_data_access().get_entry_with_id(next_patient_id, {"with_rnaseq": True})
            LAST_ROW_ID = patient_id
            yield result

    return generator(), num_steps


def to_categorical(indices, num_classes):
    indices = np.asarray(indices, dtype=int)
    out = np.zeros((len(indices), num_classes), dtype=np.float32)
    out[np.arange(len(indices)), indices] = 1.0
    return out


def make_keras_generator(args, wrapped_generator, num_steps,
                         batch_size=1, num_losses=1,
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


# =========================
# True MISE reconstruction
# =========================
def get_normalised_rnaseq(benchmark: TCGABenchmark, patient_rowid: int) -> np.ndarray:
    entry = benchmark.get_data_access().get_entry_with_id(patient_rowid, {"with_rnaseq": True})[1]
    rnaseq_data = np.array(entry["rnaseq"][1], dtype=np.float32)
    rnaseq_data = (rnaseq_data - benchmark.get_data_access().min_val) / (
        benchmark.get_data_access().max_val - benchmark.get_data_access().min_val + 1e-5
    )
    if benchmark.num_features > 0:
        rnaseq_data = rnaseq_data[:benchmark.num_features]
    return rnaseq_data


def reconstruct_true_curves_for_patient(benchmark: TCGABenchmark, patient_rowid: int, dose_grid: np.ndarray,
                                        eps: float = 1e-10) -> np.ndarray:
    x = get_normalised_rnaseq(benchmark, patient_rowid)
    treatment_chosen, y_all, treatment_strengths = benchmark.get_assignment(patient_rowid, x)
    # y_all shape [K], already scaled by scaling_constant.
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
    grids = [reconstruct_true_curves_for_patient(benchmark, int(pid), dose_grid) for pid in patient_ids]
    return np.stack(grids, axis=0)  # [N, K, G]


def compute_true_tcga_mise(pred_grid: np.ndarray, true_grid: np.ndarray, dose_grid: np.ndarray) -> Tuple[float, float]:
    if pred_grid.shape != true_grid.shape:
        raise ValueError(f"Shape mismatch: pred {pred_grid.shape} vs true {true_grid.shape}")
    sq_err = (pred_grid - true_grid) ** 2
    int_err = np.trapz(sq_err, x=dose_grid, axis=2)  # [N, K]
    mise = float(np.mean(int_err))
    sqrt_mise = float(np.sqrt(mise))
    return mise, sqrt_mise


# =========================
# Main
# =========================
def build_benchmark_and_fit(data_dir: str, tcga_num_features: int, batch_size: int,
                            validation_fraction: float, test_fraction: float, seed: int,
                            strength_of_assignment_bias: float) -> Tuple[TCGABenchmark, np.ndarray]:
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
        num_treatments=3,
        strength_of_assignment_bias=strength_of_assignment_bias,
        with_exposure=True,
        seed=seed,
        tcga_num_features=tcga_num_features,
    )
    benchmark.initialise(args)

    wrapped_generator, num_steps = make_generator(
        args=args,
        benchmark=benchmark,
        is_validation=False,
        is_test=False,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        randomise=True,
        stratify=True,
        resample_with_replacement=False,
    )

    keras_gen, keras_steps = make_keras_generator(
        args=args,
        wrapped_generator=wrapped_generator,
        num_steps=num_steps,
        batch_size=batch_size,
        num_losses=1,
        benchmark=benchmark,
        is_train=True,
    )

    log("[FIT] Building benchmark state...")
    benchmark.fit(keras_gen, steps=keras_steps, batch_size=batch_size)
    log("[FIT] Done.")

    # Recreate the test split deterministically.
    test_wrapped, _ = make_generator(
        args=args,
        benchmark=benchmark,
        is_validation=False,
        is_test=True,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        randomise=False,
        stratify=True,
        resample_with_replacement=False,
    )

    test_patients = benchmark.get_data_access().get_labelled_patients()
    # Recompute same split more directly to extract ids.
    labels, _ = benchmark.get_data_access().get_labels(args, map(lambda x: (x,), test_patients), benchmark)
    num_patients = len(test_patients)
    num_test_patients = int(np.floor(num_patients * test_fraction))
    test_sss = StratifiedShuffleSplit(n_splits=1, test_size=num_test_patients, random_state=0)
    _, test_indices = next(test_sss.split(test_patients, labels))
    test_patient_ids = test_patients[test_indices]

    return benchmark, test_patient_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing tcga.db, min_val.npy, max_val.npy")
    parser.add_argument("--tcga_num_features", type=int, default=20531)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dose_grid_size", type=int, default=65)
    parser.add_argument("--validation_fraction", type=float, default=0.27)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--strength_of_assignment_bias", type=float, default=10.0)
    parser.add_argument("--sanity_oracle", action="store_true")
    args = parser.parse_args()

    benchmark, test_patient_ids = build_benchmark_and_fit(
        data_dir=args.data_dir,
        tcga_num_features=args.tcga_num_features,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        strength_of_assignment_bias=args.strength_of_assignment_bias,
    )

    dose_grid = np.linspace(0.0, 1.0, args.dose_grid_size, dtype=np.float32)
    log(f"[EVAL] Building true response grid for {len(test_patient_ids)} test patients...")
    true_grid = build_true_response_grid(benchmark, test_patient_ids, dose_grid)
    log(f"[EVAL] true_grid shape = {true_grid.shape}")

    if args.sanity_oracle:
        pred_grid = true_grid.copy()
        mise, sqrt_mise = compute_true_tcga_mise(pred_grid, true_grid, dose_grid)
        print("sanity_oracle_mise=", mise)
        print("sanity_oracle_sqrt_mise=", sqrt_mise)
        return

    print("Built true response grid successfully.")
    print("Now plug your model predictions on the same [N, K, G] grid into compute_true_tcga_mise(...).")


if __name__ == "__main__":
    main()
