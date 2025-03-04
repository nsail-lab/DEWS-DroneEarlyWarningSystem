import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import math
import random
from haversine import haversine, Unit
#from utils import get_trajectory_stats
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import tempfile
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE,SelectKBest,chi2,mutual_info_classif,SequentialFeatureSelector
import time

import json
from shapely.geometry import Point, Polygon

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.linear_model import LogisticRegression
import argparse

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from get_features import get_features, get_similarity_features

def get_highly_correlated_features(X, threshold=0.95):
    # Calculate correlation matrix
    # Check if X contains any NaNs
    if np.isnan(X).any():
        print("X contains NaN values. Please clean the data.")
    else:
        # Check if all columns are numeric
        if not np.issubdtype(X.dtype, np.number):
            print("X contains non-numeric data. Please ensure all data is numeric.")
        else:
            try:
                # Calculate correlation matrix
                corr_matrix = np.corrcoef(X, rowvar=False)

                # Identify highly correlated columns (correlation > 0.9)
                upper_triangle_indices = np.triu_indices_from(corr_matrix, k=1)
                high_corr_pairs = [(i, j) for i, j in zip(*upper_triangle_indices) if abs(corr_matrix[i, j]) > threshold]

                # Remove one of each pair of highly correlated columns
                columns_to_remove = set()
                for i, j in high_corr_pairs:
                    columns_to_remove.add(j)
                columns_to_remove = np.array(list(columns_to_remove))
            except Exception as e:
                print(f"An error occurred: {e}")
    return columns_to_remove                

def get_shape_id(x, y, assets):
    point = Point(x, y)
    for shape in assets['shapes']:
        polygon = Polygon(shape['geometry']['coordinates'][0])
        if polygon.contains(point):
            return shape['id']
    return None

# categories features: ['capabilities','basic','altitude','speed','assets','noflyzones']
# categories similarity: ['self', 'cross'], 'all
def build_dataset(df, df_threat_scores, drop_out_features=[], drop_out_similarity=[], metric_similarity='cosine',early_n=None, early_t=360, k_neighbors=5):

    df_trajectories = df.groupby('id').apply(lambda group: get_features(group,
                                                                    drop_out = drop_out_features, 
                                                                    early_n = early_n,
                                                                    early_t=early_t))

    # dealing with None
    speed_columns = [c for c in df_trajectories.columns.tolist() if c.startswith('sp_')]
    df_trajectories[speed_columns] = df_trajectories[speed_columns].fillna(0.)

    df_trajectories = df_trajectories.fillna(df_trajectories.mean(numeric_only=True))
    df_trajectories = df_trajectories.fillna(0.)

    df_sim_features = get_similarity_features(df_trajectories, 
                                          df_trajectories.copy(),
                                          df_threat_scores,
                                          k=k_neighbors,
                                          metric=metric_similarity,
                                          drop_out=drop_out_similarity)
    df_sim_features = df_sim_features.fillna(0.)
    df_features = pd.merge(df_trajectories, df_sim_features, on='id')

    return df_features

def transform_y(y, threshold):
    return (y > threshold).astype(int)

def input_fn(df):
  TRAIN_COLUMNS = df.drop('y').columns
  feature_cols = {k: tf.constant(df[k].values)
                     for k in TRAIN_COLUMNS}
  # Converts the label column into a constant Tensor.
  label = tf.constant(df['y'].values)
  # Returns the feature columns and the label.
  return feature_cols, label

class Classifier():
    def __init__(self, model_name, random_state, input_shape=None):
        self.model_name = model_name
        self.random_state = random_state

        self.model = None 
        if model_name == 'logistic_regression':
            self.model = LogisticRegression(random_state=random_state)
        elif model_name == 'knn':
            self.model = KNeighborsClassifier()
        elif model_name == 'svm':
            self.model = SVC(probability=True, random_state=random_state)
        elif model_name == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=random_state)
        elif model_name == 'random_forest':
            self.model = RandomForestClassifier(random_state=random_state)
        elif model_name == 'gradient_boosting':
            self.model = GradientBoostingClassifier(random_state=random_state)
        elif model_name == 'naive_bayes':
            self.model = GaussianNB()
        elif model_name == 'adaboost':
            self.model = AdaBoostClassifier(random_state=random_state)
        elif model_name == 'extra_trees':
            self.model = ExtraTreesClassifier(random_state=random_state)
        elif model_name == 'mlp':
            self.model = MLPClassifier(random_state=random_state)
        elif model_name == 'wide_n_deep':
            # Define input shape
            input_layer = Input(shape=(input_shape,))

            # Define deep path
            deep_layer1 = Dense(128, activation='relu')(input_layer)
            deep_layer2 = Dense(64, activation='relu')(deep_layer1)
            deep_output = Dense(32, activation='relu')(deep_layer2)

            # Define wide path
            wide_output = Dense(1, activation='linear')(input_layer)

            # Concatenate deep and wide paths
            concatenated = Concatenate()([deep_output, wide_output])

            # Final output layer
            output_layer = Dense(1, activation='sigmoid')(concatenated)

            # Create model
            self.model = Model(inputs=input_layer, outputs=output_layer)
            # Compile model
            self.model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            # Print model summary
            # self.model.summary()
        else:
            print(f"[ERROR] model name ({model_name}) not found.")

    def fit(self, X, y):
        if self.model_name != 'wide_n_deep':
            self.model.fit(X,y)
        else:
            self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)#, validation_data=(X_val, y_val))

    def predict_proba(self, X):
        if self.model_name != 'wide_n_deep':
            return self.model.predict_proba(X)[:, 1]
        else:
            return np.array([x[0] for x in self.model.predict(X, verbose=0)])
        
def main():
    parser = argparse.ArgumentParser(description="Drone Threat Model")
    parser.add_argument('--k', type=int, default=8, help='Number of points in each trajectory')
    parser.add_argument('--time_based', type=bool, default=True, help='Filtering is performed on the basis of seconds from departure')
    parser.add_argument('--filter_n', type=int, default=720, help='Filter trajectories with fewer timestamps')
    parser.add_argument('--mi_k', type=int, default=10, help='Number of features selected according to mutual information gain')
    parser.add_argument('--threshold', type=int, default=7, help='Threshold for transforming target variable')
    parser.add_argument('--y_aggregation', type=str, default='max', help='Aggregation fn for label')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--clfs', type=str, nargs='+', default=['wide_n_deep'], help='Classifiers to use')
    parser.add_argument('--clfs_weights', type=float, nargs='+', default=[1.0], help='Weights for each classifier')
    parser.add_argument('--drop_out_features', type=str, nargs='+', default=[], help='Category of features to not use')
    parser.add_argument('--drop_out_similarity', type=str, nargs='+', default=[], help='Category of similarity features to not use')
    parser.add_argument('--metric_similarity', type=str, default='cosine', help='Metric for distance computation')
    parser.add_argument('--k_neighbors', type=int, default=5, help='Number of neighbors to be considered for trajectory similarity')
    parser.add_argument('--file_path', type=str, help='Output file path')
    parser.add_argument('--folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--n_samples_runtime', type=int, default=30, help='Number of samples for inference time measurement')
    parser.add_argument('--root', type=str, default = '/home/afg0547/DroneThreatModel', help ='Directory containing src and data')
    
    args = parser.parse_args()

    print(args)

    ROOT = args.root 
    DATA_ROOT = os.path.join(ROOT,'data')
    TIMESTAMP = '20240422'

    # data points
    df = pd.read_csv(os.path.join(DATA_ROOT,f"{TIMESTAMP}_df_datapoints_forannotation.csv"))
    df.timestamp = pd.to_datetime(df.timestamp)

    # features related to point-asset
    df_asset_info = pd.read_csv(os.path.join(DATA_ROOT,f"{TIMESTAMP}_point_asset.csv"))
    # features related to point-noflyzone
    df_nofly_info = pd.read_csv(os.path.join(DATA_ROOT,f"20240725_point_noflyzones.csv"))

    df_threat_scores = pd.read_csv(os.path.join(DATA_ROOT,f"{TIMESTAMP}_threat_annotations.csv"))
    df_threat_scores = df_threat_scores.groupby('trajectory_id').agg({'threat_score':args.y_aggregation})

    df = df[df.id.isin(df_threat_scores.index)]


    df = pd.merge(df,df_asset_info.drop('id',axis=1),left_index=True,right_on='point_id',how='left').drop('point_id',axis=1)
    df = pd.merge(df,df_nofly_info.drop('id',axis=1),left_index=True,right_on='point_id',how='left').drop('point_id',axis=1)

    df = df[df['vehicle_model'] != 'M300 RTK']

    if '' in args.drop_out_features:
        args.drop_out_features = ['assets'] 
    else:
        args.drop_out_features = ['assets'] + args.drop_out_features

    df_features = build_dataset(df, 
                            df_threat_scores, 
                            drop_out_features= args.drop_out_features, 
                            drop_out_similarity= args.drop_out_similarity, 
                            metric_similarity=args.metric_similarity,
                            early_n=args.filter_n if args.time_based is False else None, 
                            early_t=args.filter_n if args.time_based is True else None, 
                            k_neighbors=args.k_neighbors)

    df_dataset = pd.merge(df_threat_scores, df_features, left_index=True, right_on='id')
    df_dataset = df_dataset.sort_values('t_start', ascending=True) 
    
    
    df_dataset.to_csv(os.path.join(DATA_ROOT,f"{TIMESTAMP}_toshare.csv"),index=False)
    
    exit(0)
    
    X = df_dataset.drop(['threat_score', 'id', 't_start', 'vehicle_model', 'serial'], axis=1)
    X = X.apply(pd.to_numeric, errors='coerce')
    for x in X.columns:
        X[x] = X[x].astype(float)
    original_columns_init = np.array(X.columns)

    X = X.to_numpy()
    y = df_dataset['threat_score'].to_numpy()
    
    trajectory_ids = df_dataset['id'].to_numpy()

    y_binary = transform_y(y, args.threshold)
    
    if len(np.unique(y_binary)) < 2 : 
        print(f"[ERROR] the selected threshold ({args.threshold}) makes all the samples end in one single class")

    support_class_0 = np.unique(y_binary, return_counts=True)[1][0]
    support_class_1 = np.unique(y_binary, return_counts=True)[1][1]

    scaler = StandardScaler()


    n_samples = len(X)
    fold_size = int(n_samples / args.folds)
    fold_results = []

    for fold in range(args.folds):
        original_columns = original_columns_init.copy()

        end_train = int((0.5 + 0.1*fold) * n_samples)
        end_test = end_train + int(0.1 * n_samples)
        
        X_train, X_test = X[:end_train], X[end_train:end_test]
        y_train, y_test = y_binary[:end_train], y_binary[end_train:end_test]
        ids_test = trajectory_ids[end_train:end_test]

        #Removing constant columns
        constant_columns = [col for col in range(X_train.shape[1]) if np.all(X_train[:, col] == X_train[0, col])]
        X_train = np.delete(X_train, constant_columns, axis=1)
        X_test = np.delete(X_test, constant_columns, axis=1)
        constant_columns_names = original_columns[constant_columns]
        original_columns = np.delete(original_columns, constant_columns)

        #Removing highly-correlated columns
        highly_correlated_columns = get_highly_correlated_features(X_train, threshold=0.95)
        X_train = np.delete(X_train, highly_correlated_columns, axis=1)
        X_test = np.delete(X_test, highly_correlated_columns, axis=1)
        highly_correlated_columns_names = original_columns[highly_correlated_columns]
        original_columns = np.delete(original_columns, highly_correlated_columns)

        #Removing based on Mutual Information
        selector = SelectKBest(score_func=mutual_info_classif, k=args.mi_k)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        mask = selector.get_support()
        mi_columns = np.where(~mask)[0]
        mi_columns_names = original_columns[mi_columns]     
        original_columns = np.delete(original_columns, mi_columns)

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clfs = []
        for clf_name in args.clfs:
            clf = Classifier(model_name=clf_name, random_state=args.random_state, input_shape=X_train.shape[1])
            clf.fit(X_train, y_train)
            clfs.append(clf)

        weighted_preds = np.zeros_like(y_test, dtype=float)

        for clf, weight in zip(clfs, args.clfs_weights):
            weighted_preds += weight * clf.predict_proba(X_test)

        weighted_preds /= np.sum(args.clfs_weights)
        y_pred = (weighted_preds > 0.5).astype(int)

        acc_score = accuracy_score(y_test, y_pred)
        prec_0_score = precision_score(y_test, y_pred, pos_label=0)
        rec_0_score = recall_score(y_test, y_pred, pos_label=0)
        f1_0_score = f1_score(y_test, y_pred, pos_label=0)
        prec_1_score = precision_score(y_test, y_pred, pos_label=1)
        rec_1_score = recall_score(y_test, y_pred, pos_label=1)
        f1_1_score = f1_score(y_test, y_pred, pos_label=1)
        
        fold_results.append({
            'kept_columns': list(original_columns),
            'removed_constant_columns': list(constant_columns_names),
            'removed_highly_correlated_columns': list(highly_correlated_columns_names),
            'removed_mi_columns': list(mi_columns_names),
            'fold': fold + 1,
            'support_0': int(np.unique(y_binary[:end_train], return_counts=True)[1][0]),
            'support_1': int(np.unique(y_binary[:end_train], return_counts=True)[1][1]),
            'accuracy': acc_score,
            'prec_0': prec_0_score,
            'rec_0': rec_0_score,
            'f1_0': f1_0_score,
            'prec_1': prec_1_score,
            'rec_1': rec_1_score,
            'f1_1': f1_1_score,
            'predictions': [{'id': int(id), 'pred': int(pred)} for id, pred in zip(ids_test, y_pred)]
        })

    # Calculate average and standard deviation
    avg_results = {
        'accuracy': np.mean([result['accuracy'] for result in fold_results]),
        'prec_0': np.mean([result['prec_0'] for result in fold_results]),
        'rec_0': np.mean([result['rec_0'] for result in fold_results]),
        'f1_0': np.mean([result['f1_0'] for result in fold_results]),
        'prec_1': np.mean([result['prec_1'] for result in fold_results]),
        'rec_1': np.mean([result['rec_1'] for result in fold_results]),
        'f1_1': np.mean([result['f1_1'] for result in fold_results]),
        'std_accuracy': np.std([result['accuracy'] for result in fold_results]),
        'std_prec_0': np.std([result['prec_0'] for result in fold_results]),
        'std_rec_0': np.std([result['rec_0'] for result in fold_results]),
        'std_f1_0': np.std([result['f1_0'] for result in fold_results]),
        'std_prec_1': np.std([result['prec_1'] for result in fold_results]),
        'std_rec_1': np.std([result['rec_1'] for result in fold_results]),
        'std_f1_1': np.std([result['f1_1'] for result in fold_results]),
    }

    start_time = time.time()
    df_features = build_dataset(df, 
                            df_threat_scores, 
                            drop_out_features= args.drop_out_features, 
                            drop_out_similarity= args.drop_out_similarity, 
                            metric_similarity=args.metric_similarity,
                            early_n=args.filter_n if args.time_based is False else None, 
                            early_t=args.filter_n if args.time_based is True else None, 
                            k_neighbors=args.k_neighbors)
    feature_extraction_time = time.time() - start_time
    
    inference_times = []
    for _ in range(args.n_samples_runtime):

        start_time = time.time()
        
        df_sample_features = df_features.sample(n=1) #df_dataset[df_dataset.id==df_dataset.sample(n=1).id.iloc[0]]
        df_sample_features = df_sample_features.apply(pd.to_numeric, errors='coerce')        

        df_sample_features = df_sample_features[original_columns]
        X_sample = df_sample_features.to_numpy()
        X_sample = X_sample.astype(np.float64)

        # Model inference
        for clf in clfs:
            y_sample_pred = clf.predict_proba(X_sample)
        
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)

    # Calculate mean and standard deviation of inference times
    inference_mean = np.mean(inference_times)
    inference_std = np.std(inference_times)
    n_samples = len(inference_times)
    
    results={
        'args': vars(args),
        'support_0': int(support_class_0),
        'support_1': int(support_class_1),
        'fold_results': fold_results,
        'avg_results': avg_results,
        'inference_results': {
            'n_samples': n_samples,
            'feature_extraction_time': feature_extraction_time,
            'mean_inference_time': inference_mean,
            'std_inference_time': inference_std
        }
    }

    # Convert the dictionary to a JSON string
    json_results = json.dumps(results)

    # Open the file in append mode and write the JSON string
    with open(args.file_path, 'a') as f:
        f.write(json_results + '\n')

    print("JSON object appended successfully.")



if __name__ == "__main__":
    main()
