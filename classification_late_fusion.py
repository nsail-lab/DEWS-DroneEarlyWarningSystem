import json 
import pandas as pd 

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tqdm import tqdm 
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

import copy 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from get_features import get_features, get_similarity_features

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for the Drone Threat Model")

    parser.add_argument('--data_path', type=str, default='/home/afg0547/DroneThreatModel/data/results/df_experiments.csv',
                        help='Path to the classification JSON file')
    parser.add_argument('--filter_n', type=int, default=10,
                        help='Filter parameter')
    parser.add_argument('--threshold', type=int, default=3,
                        help='Threshold value')
    
    parser.add_argument('--drop_out_features', type=str, default='',
                        help='drop_out_features')
    parser.add_argument('--drop_out_similarity', type=str, default='',
                        help='drop_out_similarity')
    
    parser.add_argument('--n_combinations', type=int, default=10000,
                        help='Number of weights combinations to run')
    parser.add_argument('--root', type=str, default='/home/afg0547/DroneThreatModel',
                        help='Root directory')
    parser.add_argument('--y_aggregation', type=str, default='max',
                        help='Aggregation method for y (e.g., max, mean)')
    parser.add_argument('--metric_similarity', type=str, default='cosine',
                        help='Metric to calculate similarity (e.g., cosine, euclidean)')
    parser.add_argument('--time_based', type=bool, default=True,
                        help='Flag to use time-based features')
    parser.add_argument('--output_file_path', type=str, default='/home/afg0547/DroneThreatModel/data/results/LATE_FUSION.json',
                        help='Path to save the late fusion JSON file')


    return parser.parse_args()


def transform_y(y, threshold):
    return (y > threshold).astype(int)

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
    args = parse_args()


    df_classifiers = pd.read_csv(args.data_path)

    print("Size: ", len(df_classifiers))

    df_classifiers['f1'] = (df_classifiers['f1_1'] + df_classifiers['f1_0']) / 2

    #df_classifiers[mi_k'] = 10
    df_classifiers = df_classifiers.fillna('')
    
    df_classifiers = df_classifiers[
        (df_classifiers['filter_n'] == args.filter_n) & 
        (df_classifiers['threshold'] == args.threshold) &
        (df_classifiers['drop_out_features'] == args.drop_out_features)  &
        (df_classifiers['drop_out_similarity'] == args.drop_out_similarity) 
    ]

    
    df_classifiers.fold_results = df_classifiers.fold_results.apply(eval)

    result_df = df_classifiers.loc[df_classifiers.groupby('clfs')['f1'].idxmax()]


    ROOT = args.root #'/tank/local/hgi0312/DroneThreatModel'
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

    classifiers = {}
    classifier_probs = {}

    for i, row in result_df.iterrows():
        classifiers[row['clfs']] = []
        classifier_probs[row['clfs']] = []

        df_features = build_dataset(df, 
                                    df_threat_scores, 
                                    drop_out_features= [row['drop_out_features']], 
                                    drop_out_similarity= [row['drop_out_similarity']], 
                                    metric_similarity=row['metric_similarity'],
                                    early_n=row['filter_n'] if row['time_based'] is False else None, 
                                    early_t=row['filter_n'] if row['time_based'] is True else None, 
                                    k_neighbors=row['k_neighbors'])

        df_dataset = pd.merge(df_threat_scores, df_features, left_index=True, right_on='id')
        df_dataset = df_dataset.sort_values('t_start', ascending=True) 

        X = df_dataset.drop(['threat_score', 'id', 't_start', 'vehicle_model', 'serial'], axis=1)
        X = X.apply(pd.to_numeric, errors='coerce')
        for x in X.columns:
            X[x] = X[x].astype(float)

        y = df_dataset['threat_score'].to_numpy()
        trajectory_ids = df_dataset['id'].to_numpy()

        y_binary = transform_y(y, args.threshold)
            
        if len(np.unique(y_binary)) < 2 : 
            print(f"[ERROR] the selected threshold ({args.threshold}) makes all the samples end in one single class")

        support_class_0 = np.unique(y_binary, return_counts=True)[1][0]
        support_class_1 = np.unique(y_binary, return_counts=True)[1][1]
        scaler = StandardScaler()

        n_samples = len(X)
        fold_size = int(n_samples / row['folds'])
        fold_results = []

        for fold in range(row['folds']):
            end_train = int((0.5 + 0.1*fold) * n_samples)
            end_test = end_train + int(0.1 * n_samples)

            X_fold = X[row['fold_results'][fold]['kept_columns']].to_numpy()

            X_train, X_test = X_fold[:end_train], X_fold[end_train:end_test]
            y_train, y_test = y_binary[:end_train], y_binary[end_train:end_test]

            ids_test = trajectory_ids[end_train:end_test]
            
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)  

            clf = Classifier(model_name=row['clfs'], random_state=row['random_state'], input_shape=X_train.shape[1])
            clf.fit(X_train, y_train)


            classifiers[row['clfs']].append(copy.deepcopy(clf))
            classifier_probs[row['clfs']].append(copy.deepcopy(clf.predict_proba(X_test)))

    results_all = []
    for i in tqdm(range(args.n_combinations)):

        random_numbers = np.random.rand(len(classifiers))
        weights = random_numbers / np.sum(random_numbers)
        weights = {clf: w for clf,w in zip(classifiers, weights)}
            
        scaler = StandardScaler()
        fold_results = []
        

        for fold in range(5):
            end_train = int((0.5 + 0.1*fold) * n_samples)
            end_test = end_train + int(0.1 * n_samples)

            y_train, y_test = y_binary[:end_train], y_binary[end_train:end_test]
            ids_test = trajectory_ids[end_train:end_test]

            weighted_preds = np.zeros_like(y_test, dtype=float) 
            
            for i, row in result_df.iterrows():
                weighted_preds += weights[row['clfs']] * classifier_probs[row['clfs']][fold]
            
            y_pred = (weighted_preds > 0.5).astype(int) 
            acc_score = accuracy_score(y_test, y_pred)
            prec_0_score = precision_score(y_test, y_pred, pos_label=0)
            rec_0_score = recall_score(y_test, y_pred, pos_label=0)
            f1_0_score = f1_score(y_test, y_pred, pos_label=0)
            prec_1_score = precision_score(y_test, y_pred, pos_label=1)
            rec_1_score = recall_score(y_test, y_pred, pos_label=1)
            f1_1_score = f1_score(y_test, y_pred, pos_label=1)
            
            fold_results.append({
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
                'predictions': [{'id': int(id), 'pred': int(pred)} for id, pred in zip(ids_test, y_pred)], 
            })
        
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

        results={
            'args': vars(args),
            'weights': weights,
            'support_0': int(support_class_0),
            'support_1': int(support_class_1),
            'fold_results': fold_results,
            'avg_results': avg_results,
        }    

        json_results = json.dumps(results)
        # Open the file in append mode and write the JSON string
        with open(args.output_file_path, 'a') as f:
            f.write(json_results + '\n')
        

if __name__ == "__main__":
    main()