import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import re
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest
import shap
def standardize_symptom_names(symptom: str) -> str:
    if not isinstance(symptom, str):
        return str(symptom).lower()

    # Convert to lowercase
    symptom = symptom.lower()

    # Remove any special characters
    symptom = re.sub(r'[^a-z0-9\s]', '', symptom)

    # Replace multiple spaces with single space
    symptom = re.sub(r'\s+', ' ', symptom)

    # Trim whitespace
    symptom = symptom.strip()

    return symptom


def standardize_column_names(df):
    df_copy = df.copy()
    df_copy.columns = [str(col) for col in df_copy.columns]
    print(f"Original first 5 column names: {list(df_copy.columns)[:5]}")
    if len(df_copy.columns) > 0:
        first_col = df_copy.columns[0]
        unique_vals = df_copy[first_col].nunique()
        print(f"First column '{first_col}' has {unique_vals} unique values")
        sample_values = df_copy[first_col].unique()[:5]
        print(f"Sample values: {sample_values}")
        df_copy = df_copy.rename(columns={first_col: 'disease'})
        print(f"Renamed '{first_col}' to 'disease'")
        col_mapping = {old_col: f'symptom_{i}' for i, old_col in enumerate(df_copy.columns[1:])}
        df_copy = df_copy.rename(columns=col_mapping)
        print(f"Renamed symptom columns to standard format")
    else:
        raise ValueError("DataFrame appears to be empty")
    if 'disease' not in df_copy.columns:
        raise ValueError("Failed to create 'disease' column")

    print(f"Standardized column names. First 5: {list(df_copy.columns)[:5]}")
    return df_copy

def prepare_severity_mapping(severity_data: pd.DataFrame) -> Dict[str, int]:
    # If severity data is empty or doesn't have required columns, return empty dict
    if severity_data.empty or 'Symptom' not in severity_data.columns or 'weight' not in severity_data.columns:
        print("Warning: No valid severity data found, using default weights")
        return {}
    severity_mapping = {}
    for _, row in severity_data.iterrows():
        symptom = standardize_symptom_names(row['Symptom'])
        weight = row.get('weight', 1)
        severity_mapping[symptom] = weight

    print(f"Created severity mapping for {len(severity_mapping)} symptoms")
    return severity_mapping

def prepare_precaution_mapping(precaution_data: pd.DataFrame) -> Dict[str, List[str]]:
    if precaution_data.empty or 'Disease' not in precaution_data.columns:
        print("Warning: No valid precaution data found, using default precautions")
        return {}
    precaution_mapping = {}

    for _, row in precaution_data.iterrows():
        disease = row['Disease'].lower()
        precautions = []
        for i in range(1, 5):
            col_name = f'Precaution_{i}'
            if col_name in row and pd.notna(row[col_name]):
                precautions.append(row[col_name])
        precaution_mapping[disease] = precautions

    print(f"Created precaution mapping for {len(precaution_mapping)} diseases")
    return precaution_mapping


def examine_dataset(file_path):
    print(f"\n======= EXAMINING DATASET: {file_path} =======\n")
    try:
        df = pd.read_csv(file_path, nrows=10)
        print(f"Successfully read with pd.read_csv")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First 3 rows:")
        print(df.head(3))
        print("\nUnique values per column:")
        for col in df.columns:
            print(f"  {col}: {df[col].nunique()} unique values")
        print("\nData types:")
        print(df.dtypes)
        print("\nMissing values:")
        print(df.isnull().sum())

    except Exception as e:
        print(f"Error with pandas read_csv: {e}")
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                rows = [row for i, row in enumerate(reader) if i < 5]
            print("\nRead with csv module:")
            print(f"First row length: {len(rows[0])}")
            print(f"First few values: {rows[0][:5]}")
            if len(rows) > 1:
                col_0_values = set(row[0] for row in rows[1:] if row)
                print(f"First column has {len(col_0_values)} unique values")
                print(f"Sample values: {list(col_0_values)[:5]}")
        except Exception as e2:
            print(f"Error with csv module: {e2}")
    print("\n======= EXAMINATION COMPLETE =======\n")
def analyze_dataset_structure(file_path):
    import csv
    print(f"Analyzing dataset structure of {file_path}...")
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        sample = f.read(4096)
    delimiters = [',', ';', '\t', '|']
    delimiter_counts = {d: sample.count(d) for d in delimiters}
    likely_delimiter = max(delimiter_counts, key=delimiter_counts.get)
    print(f"Likely delimiter: '{likely_delimiter}' (found {delimiter_counts[likely_delimiter]} times)")
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=likely_delimiter)
            header = next(reader)
            print(f"Header has {len(header)} columns:")
            print(header)
            print("\nFirst 3 rows:")
            for i, row in enumerate(reader):
                if i < 3:
                    print(f"Row {i+1}: {row[:5]}... ({len(row)} columns)")
                else:
                    break
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f, delimiter=likely_delimiter)
                next(reader)  # Skip header
                rows = [row for i, row in enumerate(reader) if i < 100]
            max_cols = max(len(row) for row in rows)
            min_cols = min(len(row) for row in rows)
            print(f"\nColumn count analysis (first 100 rows):")
            print(f"- Maximum columns: {max_cols}")
            print(f"- Minimum columns: {min_cols}")
            if min_cols != max_cols:
                print(f"WARNING: Inconsistent column counts detected!")
    except Exception as e:
        print(f"Error analyzing file: {e}")
def clean_dataset():
    print("Attempting to clean and fix the dataset...")

    try:
        encodings = ['utf-8-sig', 'utf-8', 'latin1']
        raw_data = None
        for encoding in encodings:
            try:
                print(f"Trying to read raw data with {encoding} encoding...")
                raw_data = pd.read_csv('Cleaned_Dataset.csv', encoding=encoding)
                print(f"Successfully read raw data with {encoding}")
                break
            except Exception as e:
                print(f"Failed with {encoding} encoding: {e}")
                continue
        if raw_data is None:
            print("Failed to read the dataset with any encoding.")
            return False
        cleaned_data = raw_data.copy()
        first_col = cleaned_data.columns[0]
        cleaned_data = cleaned_data.rename(columns={first_col: 'disease'})
        for i, col in enumerate(cleaned_data.columns[1:], 0):
            cleaned_data = cleaned_data.rename(columns={col: f'symptom_{i}'})

        cleaned_data = cleaned_data.fillna(0)

        cleaned_data.to_csv('Better_Cleaned_Dataset.csv', index=False)
        print(f"Saved cleaned dataset with {len(cleaned_data)} rows and {len(cleaned_data.columns)} columns")
        return True

    except Exception as e:
        print(f"Error cleaning dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_datasets():
    start_time = time.time()
    print(f"[{time.time() - start_time:.2f}s] Starting to load datasets...")
    analyze_dataset_structure('Cleaned_Dataset.csv')
    try:
        encodings = ['utf-8-sig', 'utf-8', 'latin1']
        symptoms_data = None
        for encoding in encodings:
            try:
                print(f"Trying to load with {encoding} encoding...")
                symptoms_data = pd.read_csv('Cleaned_Dataset.csv', encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"Failed with {encoding} encoding")
                continue
        if symptoms_data is None:
            raise ValueError("Failed to load dataset with any encoding")
        print(f"[{time.time() - start_time:.2f}s] Loaded disease-symptom dataset with shape: {symptoms_data.shape}")
        print(f"First few column names: {symptoms_data.columns[:5].tolist()}")
        print(f"First 3 rows of data:")
        print(symptoms_data.head(3))
    except Exception as e:
        print(f"Error loading symptom dataset: {e}")
        raise
    try:
        symptom_cols = [col for col in symptoms_data.columns if col != symptoms_data.columns[0]]
        severity_data = pd.DataFrame({
            'Symptom': symptom_cols,
            'weight': [5] * len(symptom_cols)
        })
        print(f"Created placeholder severity data with {len(severity_data)} symptoms")
    except Exception as e:
        print(f"Error creating severity data: {e}")
        severity_data = pd.DataFrame(columns=['Symptom', 'weight'])
    try:
        disease_col = symptoms_data.columns[0]
        unique_diseases = symptoms_data[disease_col].unique()

        precaution_data = pd.DataFrame({
            'Disease': unique_diseases,
            'Precaution_1': ['Consult a doctor'] * len(unique_diseases),
            'Precaution_2': ['Rest and hydrate'] * len(unique_diseases),
            'Precaution_3': ['Monitor symptoms'] * len(unique_diseases),
            'Precaution_4': ['Follow medical advice'] * len(unique_diseases)
        })
        print(f"Created placeholder precaution data for {len(precaution_data)} diseases")
    except Exception as e:
        print(f"Error creating precaution data: {e}")
        precaution_data = pd.DataFrame(columns=['Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'])
    print(f"[{time.time() - start_time:.2f}s] All datasets loaded successfully")
    return symptoms_data, severity_data, precaution_data

warnings.filterwarnings('ignore')

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
all_figures = {}
all_results = {}

def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_time = time.time()
    print(f"[{time.time() - start_time:.2f}s] Starting to load datasets...")
    try:
        encodings = ['utf-8-sig', 'utf-8', 'latin1']
        symptoms_data = None

        for encoding in encodings:
            try:
                print(f"Trying to load with {encoding} encoding...")
                symptoms_data = pd.read_csv('Cleaned_Dataset.csv', encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"Failed with {encoding} encoding")
                continue
        if symptoms_data is None:
            raise ValueError("Failed to load dataset with any encoding")

        print(f"[{time.time() - start_time:.2f}s] Loaded disease-symptom dataset with shape: {symptoms_data.shape}")
        print(f"Columns: {symptoms_data.columns.tolist()}")
    except Exception as e:
        print(f"Error loading symptom dataset: {e}")
        raise
    try:
        severity_data = pd.DataFrame({
            'Symptom': symptoms_data.columns[1:],
            'weight': [5] * (len(symptoms_data.columns) - 1)
        })
        print(f"Created placeholder severity data with {len(severity_data)} symptoms")
    except Exception as e:
        print(f"Error creating severity data: {e}")
        severity_data = pd.DataFrame(columns=['Symptom', 'weight'])

    try:
        unique_diseases = symptoms_data['diseases'].unique() if 'diseases' in symptoms_data.columns else []
        if not len(unique_diseases):
            unique_diseases = symptoms_data['disease'].unique() if 'disease' in symptoms_data.columns else []

        precaution_data = pd.DataFrame({
            'Disease': unique_diseases,
            'Precaution_1': ['Consult a doctor'] * len(unique_diseases),
            'Precaution_2': ['Rest and hydrate'] * len(unique_diseases),
            'Precaution_3': ['Monitor symptoms'] * len(unique_diseases),
            'Precaution_4': ['Follow medical advice'] * len(unique_diseases)
        })
        print(f"Created placeholder precaution data for {len(precaution_data)} diseases")
    except Exception as e:
        print(f"Error creating precaution data: {e}")
        precaution_data = pd.DataFrame(columns=['Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'])

    print(f"[{time.time() - start_time:.2f}s] All datasets loaded successfully")
    return symptoms_data, severity_data, precaution_data

def transform_disease_symptom_data(df: pd.DataFrame, severity_mapping: Dict[str, int]) -> pd.DataFrame:
    start_time = time.time()
    print(f"[{time.time() - start_time:.2f}s] Starting data transformation...")
    df_transformed = df.copy()
    df_transformed = standardize_column_names(df_transformed)
    symptom_cols = [col for col in df_transformed.columns if col != 'disease']
    weighted_features = {}
    for symptom_col in tqdm(symptom_cols, desc="Processing symptom weights"):
        std_symptom = standardize_symptom_names(symptom_col)
        weight = severity_mapping.get(std_symptom, 4)
        weighted_features[symptom_col] = weight
    all_results['weighted_features'] = pd.DataFrame({
        'Symptom': list(weighted_features.keys()),
        'Weight': list(weighted_features.values())
    }).sort_values('Weight', ascending=False)
    for symptom_col, weight in weighted_features.items():
        weighted_col = f"{symptom_col}_weighted"
        df_transformed[weighted_col] = df_transformed[symptom_col] * weight

    print(f"Enhanced dataset with {len(weighted_features)} weighted symptom features")
    print(f"[{time.time() - start_time:.2f}s] Data transformation completed")

    return df_transformed, symptom_cols, weighted_features
SELECTED_DISEASES = [
    'tuberculosis', 'pneumonia', 'influenza', 'common cold',
    'chickenpox', 'hiv infection', 'hepatitis', 'malaria',
    'urinary tract infection', 'conjunctivitis due to bacteria',
    'pharyngitis', 'syphilis', 'chlamydia', 'typhoid fever',
    'meningitis', 'infectious gastroenteritis', 'mumps',
    'impetigo', 'scabies', 'lice'
]
def analyze_disease_distribution(df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[str]]:
    if 'disease' not in df.columns:
        raise ValueError("'disease' column not found in dataframe")
    disease_counts = df['disease'].value_counts()
    min_samples_per_class = 5
    common_diseases = set(disease_counts[disease_counts >= min_samples_per_class].index)
    if len(common_diseases) < 2:
        print(f"WARNING: Only found {len(common_diseases)} diseases with at least {min_samples_per_class} samples")
        print("Using all diseases with at least 1 sample")
        common_diseases = set(disease_counts.index)
    diseases_df = pd.DataFrame({
        'Disease': disease_counts.index,
        'Count': disease_counts.values
    })

    print(f"Dataset has {len(disease_counts)} diseases")
    print(f"Found {len(common_diseases)} diseases with at least {min_samples_per_class} samples")
    return diseases_df, common_diseases

def create_disease_symptom_profiles(df: pd.DataFrame, symptom_cols: List[str],
                                   weighted_features: Dict[str, int]) -> pd.DataFrame:
    profiles = []

    for disease in df['disease'].unique():
        disease_df = df[df['disease'] == disease]
        for symptom in symptom_cols:
            freq = disease_df[symptom].mean()
            if freq >= 0.25:
                weight = weighted_features.get(symptom, 0)
                profiles.append({
                    'Disease': disease,
                    'Symptom': symptom,
                    'Frequency': freq,
                    'Weight': weight,
                    'Importance': freq * weight
                })

    profiles_df = pd.DataFrame(profiles)
    profiles_df['Rank'] = profiles_df.groupby('Disease')['Importance'].rank(ascending=False)

    return profiles_df.sort_values(['Disease', 'Rank'])

def select_features_enhanced(df: pd.DataFrame, symptom_cols: List[str],
                          weighted_features: Dict[str, int],
                          n_features: Optional[int] = None) -> List[str]:
    start_time = time.time()
    print(f"[{time.time() - start_time:.2f}s] Starting feature selection on {len(symptom_cols)} features...")
    all_features = symptom_cols + [f"{col}_weighted" for col in symptom_cols]
    X = df[all_features].fillna(0)  # Fill NaN with 0
    y = df['disease_code']
    print(f"Performing feature selection on {len(all_features)} features")
    print(f"NaN values in X: {X.isna().sum().sum()}")
    print(f"NaN values in y: {y.isna().sum()}")
    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X)
    var_features = np.array(all_features)[selector.get_support()].tolist()
    print(f"Features after variance filtering: {len(var_features)}")
    mi_scores = mutual_info_classif(X[var_features], y)
    mi_features = pd.Series(mi_scores, index=var_features).sort_values(ascending=False)
    if n_features is None:
        n_features = max(50, int(np.sqrt(len(all_features))))
    selected_features = mi_features.head(n_features).index.tolist()
    weighted_selected = [f for f in selected_features if '_weighted' in f]
    if len(weighted_selected) < min(10, n_features // 4):
        weighted_candidates = [f"{col}_weighted" for col in symptom_cols]
        weighted_mi = mi_features[mi_features.index.isin(weighted_candidates)]
        n_to_add = min(10, n_features // 4) - len(weighted_selected)
        if n_to_add > 0:
            to_add = weighted_mi.head(n_to_add + len(weighted_selected))
            to_add = to_add.index[~to_add.index.isin(selected_features)].tolist()[:n_to_add]
            selected_features.extend(to_add)
    feature_importance_df = pd.DataFrame({
        'Feature': mi_features.index,
        'MI_Score': mi_features.values
    })
    all_results['feature_importance'] = feature_importance_df
    plt.figure(figsize=(12, 8))
    mi_features.head(30).plot(kind='barh')
    plt.title('Top 30 Features by Mutual Information Score')
    plt.tight_layout()
    print(f"Selected {len(selected_features)} features for modeling")
    print(f"[{time.time() - start_time:.2f}s] Feature selection completed. Selected {len(selected_features)} features.")

    return selected_features
def train_enhanced_models(df: pd.DataFrame, feature_cols: List[str],
                          hyperparameter_tuning: bool = False) -> Dict:
    start_time = time.time()
    print(f"[{time.time() - start_time:.2f}s] Starting model training...")
    X = df[feature_cols]
    y = df['disease_code']
    n_classes = len(np.unique(y))
    if n_classes < 2:
        print(f"ERROR: Only {n_classes} class found. Classification requires at least 2 classes.")
        print("Cannot continue with model training.")
        # Return an empty dictionary with required keys
        return {
            'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
            'best_model': None, 'best_model_name': None, 'models': {},
            'results': pd.DataFrame()
        }
    disease_counts = y.value_counts()
    valid_diseases = disease_counts[disease_counts >= 3].index
    filtered_data = df[df['disease_code'].isin(valid_diseases)]
    X = filtered_data[feature_cols]
    y = filtered_data['disease_code']
    X = X.fillna(0)
    y = y.fillna(y.mode()[0])
    model_pipelines = {}
    best_score = 0
    best_model_name = None
    best_model = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    try:
        class_counts = np.bincount(y_train.astype(int))
        class_counts = class_counts[class_counts > 0]
        min_samples = min(class_counts) if len(class_counts) > 0 else 0
        k_neighbors = min(min_samples - 1, 5) if min_samples > 1 else 1
        print(f"Minimum samples in any class: {min_samples}, using k_neighbors={k_neighbors}")
        if min_samples > 1:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            print(f"Applied SMOTE with k_neighbors={k_neighbors}")
        else:
            print("Too few samples for SMOTE, proceeding with original data")
            X_train_res, y_train_res = X_train, y_train
    except Exception as e:
        print(f"SMOTE failed: {e}, proceeding with original data")
        X_train_res, y_train_res = X_train, y_train
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss')
    }
    param_grids = {
        'RandomForest': {'n_estimators': [50], 'max_depth': [20], 'min_samples_split': [2]},
        'GradientBoosting': {'n_estimators': [30], 'learning_rate': [0.1], 'max_depth': [3]},
        'XGBoost': {'n_estimators': [30], 'learning_rate': [0.1], 'max_depth': [3]}
    }
    try:
        results = []

        for name, model in tqdm(models.items(), desc="Training Models"):
            model_start = time.time()
            print(f"\n[{time.time() - start_time:.2f}s] Training {name}...")

            if hyperparameter_tuning:
                grid_search = GridSearchCV(
                    model,
                    param_grids[name],
                    cv=3,
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train_res, y_train_res)
                trained_model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                trained_model = model.fit(X_train_res, y_train_res)
            model_pipelines[name] = trained_model
            y_pred = trained_model.predict(X_test)
            if hasattr(trained_model, "predict_proba"):
                y_proba = trained_model.predict_proba(X_test)
            else:
                y_proba = np.zeros((len(y_test), len(np.unique(y))))
                y_proba[np.arange(len(y_test)), y_pred] = 1
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            try:
                classes = np.unique(np.concatenate([y_test, y_train]))
                n_classes = len(classes)
                y_test_one_hot = np.zeros((len(y_test), n_classes))
                for i, c in enumerate(classes):
                    y_test_one_hot[y_test == c, i] = 1
                if y_proba.shape[1] != n_classes:
                    adjusted_y_proba = np.zeros((len(y_test), n_classes))
                    for i, c in enumerate(classes):
                        if i < y_proba.shape[1]:
                            adjusted_y_proba[:, i] = y_proba[:, i]
                    y_proba = adjusted_y_proba
                roc_auc = roc_auc_score(y_test_one_hot, y_proba, multi_class='ovr')
            except Exception as e:
                print(f"ROC-AUC calculation failed: {e}. Setting to 0.5")
            print(f"[{time.time() - start_time:.2f}s] {name} training completed in {time.time() - model_start:.2f}s")
            print(f"{name} Results:")
            print(f" Accuracy: {accuracy:.4f}")
            print(f"  F1 Score (weighted): {f1:.4f}")
            print(f"  Precision (weighted): {precision:.4f}")
            print(f"  Recall (weighted): {recall:.4f}")
            print(f"  ROC-AUC (OvR): {roc_auc:.4f}")
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'F1_Score': f1,
                'Precision': precision,
                'Recall': recall,
                'ROC_AUC': roc_auc
            })
            joblib.dump(trained_model, MODELS_DIR / f'{name}_model.joblib')
            if f1 > best_score:
                best_score = f1
                best_model_name = name
                best_model = trained_model
        plt.figure(figsize=(12, 6))
        results_df = pd.DataFrame(results)
        metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall', 'ROC_AUC']
        for metric in metrics:
            plt.plot(results_df['Model'], results_df[metric], marker='o', label=metric)
        plt.legend()
        plt.title('Model Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        print(f"\nBest model: {best_model_name} with F1 score: {best_score:.4f}")
        joblib.dump(best_model, MODELS_DIR / 'best_model.joblib')
        print(f"[{time.time() - start_time:.2f}s] All models trained successfully")
    except Exception as e:
        print(f"[{time.time() - start_time:.2f}s] Error during model training: {e}")
        import traceback
        print(f"Error during model training: {e}")
        traceback.print_exc()
        results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'F1_Score', 'Precision', 'Recall', 'ROC_AUC'])
    return {
        'X_train': X_train_res,
        'X_test': X_test,
        'y_train': y_train_res,
        'y_test': y_test,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'models': model_pipelines,
        'results': results_df if 'results_df' in locals() else pd.DataFrame()
    }
def create_advanced_explanation_system(model, feature_names: List[str],
                                     label_encoder: LabelEncoder,
                                     weighted_features: Dict[str, int],
                                     disease_symptom_profiles: pd.DataFrame,
                                     precaution_mapping: Dict[str, List[str]]) -> Dict:
    feature_importances = None
    model_obj = model
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model_obj = model.named_steps['model']
    if hasattr(model_obj, 'feature_importances_'):
        feature_importances = model_obj.feature_importances_
    shap_explainer = None
    try:
        if isinstance(model_obj, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)):
            shap_explainer = shap.TreeExplainer(model_obj)
            print("Created SHAP tree explainer for model")
        else:
            shap_explainer = shap.KernelExplainer(model_obj.predict_proba,
                                                 shap.sample(X_train, 100))
            print("Created SHAP kernel explainer for model")
    except Exception as e:
        print(f"Could not create SHAP explainer: {e}")
    base_to_weighted = {}
    weighted_to_base = {}
    for feature in feature_names:
        if '_weighted' in feature:
            base_feature = feature.replace('_weighted', '')
            base_to_weighted[base_feature] = feature
            weighted_to_base[feature] = base_feature
    return {
        'model': model,
        'feature_names': feature_names,
        'label_encoder': label_encoder,
        'feature_importances': feature_importances,
        'shap_explainer': shap_explainer,
        'weighted_features': weighted_features,
        'base_to_weighted': base_to_weighted,
        'weighted_to_base': weighted_to_base,
        'disease_symptom_profiles': disease_symptom_profiles,
        'precaution_mapping': precaution_mapping
    }
def explain_prediction(explanation_system: Dict, symptoms: List[str]) -> Dict:
    model = explanation_system['model']
    feature_names = explanation_system['feature_names']
    label_encoder = explanation_system['label_encoder']
    weighted_features = explanation_system['weighted_features']
    base_to_weighted = explanation_system['base_to_weighted']
    disease_profiles = explanation_system['disease_symptom_profiles']
    precaution_mapping = explanation_system['precaution_mapping']
    X = np.zeros((1, len(feature_names)))
    used_symptoms = []
    for i, feature in enumerate(feature_names):
        if feature in symptoms:
            X[0, i] = 1
            used_symptoms.append(feature)
        elif feature in weighted_to_base and weighted_to_base[feature] in symptoms:
            base_symptom = weighted_to_base[feature]
            weight = weighted_features.get(base_symptom, 1)
            X[0, i] = weight
    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0]
    predicted_disease = label_encoder.inverse_transform([y_pred])[0]
    sorted_indices = np.argsort(y_proba)[::-1]
    alternatives = []
    for idx in sorted_indices[1:4]:  # Next 3 highest probabilities
        disease = label_encoder.inverse_transform([idx])[0]
        probability = y_proba[idx] * 100
        alternatives.append({
            'disease': disease,
            'probability': probability
        })
    explanation = []
    shap_explainer = explanation_system.get('shap_explainer')
    if shap_explainer:
        try:
            shap_values = shap_explainer.shap_values(X)
            if isinstance(shap_values, list):
                pred_idx = y_pred
                class_shap_values = shap_values[pred_idx][0]
            else:
                class_shap_values = shap_values[0]
            feature_shap = list(zip(feature_names, class_shap_values))
            feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
            for feature, shap_value in feature_shap[:5]:
                if feature in weighted_to_base:
                    feature = weighted_to_base[feature]

                if shap_value > 0:
                    impact = f"+{shap_value:.2f}"
                    direction = "increases"
                else:
                    impact = f"{shap_value:.2f}"
                    direction = "decreases"
                weight = weighted_features.get(feature, 1)

                explanation.append({
                    'symptom': feature,
                    'impact': impact,
                    'weight': weight,
                    'direction': direction
                })
        except Exception as e:
            print(f"Error generating SHAP explanation: {e}")
    if not explanation:
        disease_symptoms = disease_profiles[disease_profiles['Disease'] == predicted_disease]

        if not disease_symptoms.empty:
            disease_symptoms = disease_symptoms.sort_values('Importance', ascending=False)
            for _, row in disease_symptoms.head(5).iterrows():
                symptom = row['Symptom']
                weight = row['Weight']
                freq = row['Frequency']
                present = symptom in symptoms
                if present:
                    impact = f"+{weight * freq:.2f}"
                    direction = "supports"
                else:
                    impact = "N/A"
                    direction = "typically present but missing"

                explanation.append({
                    'symptom': symptom,
                    'impact': impact,
                    'weight': weight,
                    'direction': direction
                })
    precautions = precaution_mapping.get(predicted_disease.lower(), [])
    result = {
        'prediction': {
            'disease': predicted_disease,
            'confidence': y_proba[y_pred] * 100
        },
        'alternatives': alternatives,
        'explanation': explanation,
        'precautions': precautions
    }

    return result
def export_enhanced_model_package(model, label_encoder, feature_names,
                                symptom_cols, weighted_features,
                                disease_symptom_profiles, precaution_mapping):
    explanation_system = create_advanced_explanation_system(
        model, feature_names, label_encoder,
        weighted_features, disease_symptom_profiles, precaution_mapping
    )
    model_package = {
        'model': model,
        'label_encoder': label_encoder,
        'selected_features': feature_names,
        'symptom_vocabulary': symptom_cols,
        'weighted_features': weighted_features,
        'disease_symptom_profiles': disease_symptom_profiles,
        'precaution_mapping': precaution_mapping,
        'explanation_system': explanation_system,
        'training_date': pd.Timestamp.now().isoformat(),
        'model_metadata': {
            'model_type': type(model).__name__,
            'n_features': len(feature_names),
            'n_classes': len(label_encoder.classes_)
        }
    }
    package_path = MODELS_DIR / "enhanced_model_package.joblib"
    joblib.dump(model_package, package_path)
    print(f"Enhanced model package exported to {package_path}")
    return package_path
def make_prediction(model_package_path: str, symptoms: List[str]) -> Dict:
    model_package = joblib.load(model_package_path)
    model = model_package['model']
    label_encoder = model_package['label_encoder']
    feature_names = model_package['selected_features']
    symptom_vocabulary = model_package['symptom_vocabulary']
    weighted_features = model_package.get('weighted_features', {})
    disease_symptom_profiles = model_package.get('disease_symptom_profiles', pd.DataFrame())
    precaution_mapping = model_package.get('precaution_mapping', {})
    standardized_symptoms = [standardize_symptom_names(s) for s in symptoms]
    X = np.zeros((1, len(feature_names)))
    used_symptoms = []
    weighted_to_base = {}
    for feature in feature_names:
        if '_weighted' in feature:
            base_feature = feature.replace('_weighted', '')
            weighted_to_base[feature] = base_feature
    for i, feature in enumerate(feature_names):
        if feature in standardized_symptoms:
            X[0, i] = 1
            used_symptoms.append(feature)
        elif feature in weighted_to_base and weighted_to_base[feature] in standardized_symptoms:
            base_symptom = weighted_to_base[feature]
            weight = weighted_features.get(base_symptom, 1)
            X[0, i] = weight
    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0]
    predicted_disease = label_encoder.inverse_transform([y_pred])[0]
    sorted_indices = np.argsort(y_proba)[::-1]
    alternatives = []

    for idx in sorted_indices[1:4]:  # Next 3 highest probabilities
        disease = label_encoder.inverse_transform([idx])[0]
        probability = y_proba[idx] * 100
        alternatives.append({
            'disease': disease,
            'probability': probability
        })
    explanation = []
    if not disease_symptom_profiles.empty:
        disease_symptoms = disease_symptom_profiles[disease_symptom_profiles['Disease'] == predicted_disease]

        if not disease_symptoms.empty:
            disease_symptoms = disease_symptoms.sort_values('Importance', ascending=False)
            for _, row in disease_symptoms.head(5).iterrows():
                symptom = row['Symptom']
                weight = row['Weight']
                freq = row['Frequency']
                present = symptom in standardized_symptoms
                if present:
                    impact = f"+{weight * freq:.2f}"
                    direction = "supports"
                else:
                    impact = "N/A"
                    direction = "typically present but missing"

                explanation.append({
                    'symptom': symptom,
                    'impact': impact,
                    'weight': weight,
                    'direction': direction
                })
    if not explanation and weighted_features:
        model_obj = model
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            model_obj = model.named_steps['model']
        if hasattr(model_obj, 'feature_importances_'):
            importances = model_obj.feature_importances_
            feature_imp = list(zip(feature_names, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            for feature, imp in feature_imp[:5]:
                if feature in weighted_to_base:
                    feature = weighted_to_base[feature]
                if feature in standardized_symptoms:
                    weight = weighted_features.get(feature, 1)
                    impact = f"+{weight * imp:.2f}"
                    direction = "supports"
                else:
                    impact = "0.00"
                    direction = "not present"

                explanation.append({
                    'symptom': feature,
                    'impact': impact,
                    'weight': weighted_features.get(feature, 1),
                    'direction': direction
                })
    precautions = precaution_mapping.get(predicted_disease.lower(), [])
    result = {
        'prediction': {
            'disease': predicted_disease,
            'confidence': y_proba[y_pred] * 100
        },
        'alternatives': alternatives,
        'explanation': explanation,
        'precautions': precautions
    }

    return result
def example_usage():
    symptoms_data, severity_data, precaution_data = load_datasets()
    symptoms_data = standardize_column_names(symptoms_data)
    severity_mapping = prepare_severity_mapping(severity_data)
    precaution_mapping = prepare_precaution_mapping(precaution_data)
    transformed_data, symptom_cols, weighted_features = transform_disease_symptom_data(
        symptoms_data, severity_mapping
    )
    diseases_df, common_diseases = analyze_disease_distribution(transformed_data)
    profiles = create_disease_symptom_profiles(
        transformed_data, symptom_cols, weighted_features
    )
    label_encoder = LabelEncoder()
    transformed_data['disease_code'] = label_encoder.fit_transform(transformed_data['disease'])
    selected_features = select_features_enhanced(
        transformed_data, symptom_cols, weighted_features
    )
    training_results = train_enhanced_models(
        transformed_data, selected_features, hyperparameter_tuning=True
    )
    model_package_path = export_enhanced_model_package(
        training_results['best_model'],
        label_encoder,
        selected_features,
        symptom_cols,
        weighted_features,
        profiles,
        precaution_mapping
    )

    print(f"\nEnhanced model package exported to: {model_package_path}")
    test_symptoms = ["fever", "fatigue", "yellowing of eyes", "liver pain"]
    result = make_prediction(model_package_path, test_symptoms)
    print("\nExample Prediction:")
    print(f"Predicted Disease: {result['prediction']['disease']}")
    print(f"Confidence: {result['prediction']['confidence']:.2f}%")
    print("\nExplanation:")
    for item in result['explanation']:
        print(f"  - {item['symptom']} (weight: {item['weight']}): {item['impact']} ({item['direction']})")
    print("\nPrecautions:")
    for precaution in result['precautions']:
        print(f"  - {precaution}")
    print("\nAlternative Possibilities:")
    for alt in result['alternatives']:
        print(f"  - {alt['disease']} ({alt['probability']:.2f}%)")

    return result

def main(dev_mode=True):
    try:
        start_time = time.time()
        print(f"[0.00s] Starting Enhanced Disease Prediction ML Pipeline")
        print("Examining dataset structure...")
        examine_dataset('Cleaned_Dataset.csv')
        print(f"[{time.time() - start_time:.2f}s] Loading datasets...")
        try:
            symptoms_data, severity_data, precaution_data = load_datasets()
        except Exception as e:
            print(f"Error loading cleaned dataset: {e}")
            print("Trying to clean the dataset again...")
            clean_dataset()
            print("Loading newly cleaned dataset...")
            symptoms_data, severity_data, precaution_data = load_datasets('Better_Cleaned_Dataset.csv')
        print(f"Original dataset columns: {symptoms_data.columns.tolist()}")
        symptoms_data = standardize_column_names(symptoms_data)
        if 'disease' not in symptoms_data.columns:
            print("ERROR: 'disease' column not found after standardization")
            print(f"Available columns: {symptoms_data.columns.tolist()}")
            disease_like_cols = [col for col in symptoms_data.columns if 'disease' in col.lower()]
            if disease_like_cols:
                print(f"Found potential disease column: {disease_like_cols[0]}")
                symptoms_data = symptoms_data.rename(columns={disease_like_cols[0]: 'disease'})
            else:
                raise ValueError("No suitable disease column found")
        print(f"Available diseases before filtering: {len(symptoms_data['disease'].unique())}")
        print(f"Sample disease names: {list(symptoms_data['disease'].unique())[:5]}")
        print(f"Looking for these diseases: {SELECTED_DISEASES[:5]}...")
        available_diseases = set(symptoms_data['disease'].str.lower().unique())
        matching_diseases = [d.lower() for d in SELECTED_DISEASES if d.lower() in available_diseases]

        if len(matching_diseases) < 2:
            print(f"WARNING: Only found {len(matching_diseases)} matching diseases. Need at least 2 for classification.")
            print(f"Matching diseases: {matching_diseases}")
            print("Will use all available diseases instead.")
            filtered_data = symptoms_data
        else:
            filtered_data = symptoms_data[symptoms_data['disease'].str.lower().isin(matching_diseases)]
            print(f"Filtered to {len(filtered_data['disease'].unique())} diseases")
        print(f"[{time.time() - start_time:.2f}s] Preprocessing datasets...")
        filtered_data = standardize_column_names(filtered_data)  # Re-standardize to be safe
        severity_mapping = prepare_severity_mapping(severity_data)
        precaution_mapping = prepare_precaution_mapping(precaution_data)
        print(f"[{time.time() - start_time:.2f}s] Transforming data with severity information...")
        transformed_data, symptom_cols, weighted_features = transform_disease_symptom_data(
            filtered_data, severity_mapping
        )
        diseases_df, common_diseases = analyze_disease_distribution(transformed_data)
        profiles = create_disease_symptom_profiles(
            transformed_data, symptom_cols, weighted_features
        )
        label_encoder = LabelEncoder()
        transformed_data['disease_code'] = label_encoder.fit_transform(transformed_data['disease'])
        selected_features = select_features_enhanced(
            transformed_data, symptom_cols, weighted_features, n_features=30  # LIMIT TO 30 FEATURES
        )
        training_results = train_enhanced_models(
            transformed_data, selected_features, hyperparameter_tuning=False  # DISABLE TUNING
        )
        if training_results['best_model'] is None:
            print("ERROR: Model training failed or produced no valid model.")
            return False
        model_package_path = export_enhanced_model_package(
            training_results['best_model'],
            label_encoder,
            selected_features,
            symptom_cols,
            weighted_features,
            profiles,
            precaution_mapping
        )
        print("\nEnhanced Disease Prediction Pipeline completed successfully!")
        print(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
        print(f"Best model: {training_results['best_model_name']}")
        print(f"Enhanced model package exported to: {model_package_path}")
        if dev_mode:
            print("\nRunning example prediction...")
            test_symptoms = ["fever", "fatigue", "yellowing of eyes", "liver pain"]
            result = make_prediction(model_package_path, test_symptoms)
            print("\nExample Prediction:")
            print(f"Predicted Disease: {result['prediction']['disease']}")
            print(f"Confidence: {result['prediction']['confidence']:.2f}%")
            print("\nExplanation:")
            for item in result['explanation']:
                print(f"  - {item['symptom']} (weight: {item['weight']}): {item['impact']} ({item['direction']})")

            print("\nPrecautions:")
            for precaution in result['precautions']:
                print(f"  - {precaution}")

    except Exception as e:
        print(f"\nERROR: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True
if __name__ == '__main__':
    main()