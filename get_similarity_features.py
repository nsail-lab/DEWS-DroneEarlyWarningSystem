from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Function to find k closest trajectories using cosine similarity and return their indices
def find_k_closest_trajectories(input_trajectory, trajectories, k, metric='cosine'):
    
    input_trajectory = np.array(input_trajectory).reshape(1, -1)
    trajectories = np.array(trajectories)
    
    scaler = StandardScaler()
    scaled_trajectories = scaler.fit_transform(trajectories)

    # Standardize the input trajectory
    input_trajectory = np.array(input_trajectory).reshape(1, -1)

    scaled_input_trajectory = scaler.transform(input_trajectory)
    
    # Compute distances
    distances = cdist(scaled_input_trajectory, scaled_trajectories, metric=metric).flatten()
    
    # Get indices of the k smallest distances
    closest_indices = np.argsort(distances)[:k]

    # Return the k closest trajectories, their distances, and their indices
    return trajectories[closest_indices], distances[closest_indices], closest_indices


def get_similarity_features_one_traj(trajectory, df_reference, threat_scores, k=3, metric='cosine'):

    if len(df_reference) == 0:
        features = {}
        for i in range(k):
            features[f'sim_k{i}'] = 0
            features[f'threat_k{i}'] = 0
    else:
        # Aligning trajectory and reference set
        reference = df_reference.drop(['t_start','vehicle_model','serial'],axis=1,inplace=False)
        trajectory = trajectory[reference.columns.tolist()].copy()
        # Cast non-numeric columns
        if 'enter_noflyzone' in reference.columns.tolist():
            reference.enter_noflyzone = reference.enter_noflyzone.astype(int)
            trajectory.enter_noflyzone = int(trajectory.enter_noflyzone)
        if 'communication_channel' in reference.columns.tolist():
            communication_channel = reference.communication_channel.apply(lambda x: 0 if x=='RF' else 1)
            reference.communication_channel = communication_channel.tolist()
            trajectory.communication_channel = int(trajectory.communication_channel!='RF')

        # Extracting closest trajectories
        trajectories, distances, indices = find_k_closest_trajectories(trajectory, reference,k=k,metric=metric)

        ids = reference.iloc[indices].index
        scores = threat_scores[threat_scores.index.isin(ids)].threat_score.tolist()
        assert len(scores) == len(distances), f'xxx {len(scores)}, {len(distances)}'

        features = {}
        for i, elem in enumerate(distances.tolist()):
            features[f'sim_k{i}'] = elem
        for i, elem in enumerate(scores):
            features[f'threat_k{i}'] = elem
            
    return features




