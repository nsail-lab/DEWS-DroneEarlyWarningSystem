import pandas as pd

import get_capabilities_features
import get_basic_features
import get_altitude_features
import get_speed_features
import get_asset_features
import get_noflyzones_features
from get_similarity_features import get_similarity_features_one_traj

# drop_out: ['capabilities','basic','altitude','speed','assets','noflyzones']
# early_n (int): extract features considering the first "early_n" points of the trajectory
# early_t (int): extract features considering the first "early_t" seconds of the trajectory
def get_features(group, drop_out = [], early_n = None, early_t = None): 
    
    assert (early_n is None) or (early_t is None), "Either early_n or early_t must be None"

    category = ['capabilities','basic','altitude','speed','assets','noflyzones']
    category = list(set(category) - set(drop_out))
 
    group = group.sort_values('timestamp')
    if early_n is not None:
        group = group.head(min(early_n, len(group)))
    
    if early_t is not None:
        unix_timestamp = group.timestamp.apply(lambda x: x.timestamp())
        delta_timestamp = unix_timestamp - unix_timestamp.iloc[0]
        group = group[delta_timestamp <= early_t]     

    feats = {}
    feats['t_start'] = group.timestamp.iloc[0]
    feats['vehicle_model'] = group.vehicle_model.iloc[0]
    feats['serial'] = group.clean_serial.iloc[0]

    if (category == 'all') or ('capabilities' in category):
        capabilities_feats = get_capabilities_features.get_capabilities_features(group)
        feats.update(capabilities_feats)
    
    if (category == 'all') or ('basic' in category):
        basic_feats = get_basic_features.get_basic_features(group)
        feats.update(basic_feats)

    if (category == 'all') or ('altitude' in category):
        altitude_feats = get_altitude_features.get_altitude_features(group)
        feats.update(altitude_feats)

    if (category == 'all') or ('speed' in category):
        speed_feats = get_speed_features.get_speed_features(group)
        feats.update(speed_feats)
    
    if (category == 'all') or ('assets' in category):
        assets_feats = get_asset_features.get_asset_features(group)
        feats.update(assets_feats)

    if (category == 'all') or ('noflyzones' in category):
        noflyzones_feats = get_noflyzones_features.get_noflyzones_feats(group)
        feats.update(noflyzones_feats)

    
    return pd.Series(feats)

# df_trajectories: dataframe with the trajectories from which to extract features
# df_reference: dataframe with the trajectories to determine the similarity from
# df_threat_scores: dataframe with the annotated threat scores
# k: number of trajectories to compute the similarity from
# metric: 'cosine', 'euclidean', 'minkowski', etc. 
# drop_out: ['self','cross']
def get_similarity_features(df_trajectories, df_reference, df_threat_scores, 
                            k=3, 
                            metric='cosine',
                            drop_out = []):
    
    category = ['self','cross'] 
    category = list(set(category) - set(drop_out))

    results = []
    
    for idx, traj_features in df_trajectories.iterrows():
        features = {}    
        features['id'] = idx
        if (category == 'all') or ('self' in category):
            df_ref = df_reference[(df_reference.vehicle_model == traj_features.vehicle_model) & \
                                  (df_reference.serial == traj_features.serial) & \
                                    (df_reference.t_start < traj_features.t_start)].copy()
            
            features_self = get_similarity_features_one_traj(traj_features,df_ref, df_threat_scores, 
                                                            k=k,metric=metric)
            features_self = {'self_' + key: value for key, value in features_self.items()}
            features.update(features_self)

        if (category == 'all') or ('cross' in category):
            df_ref = df_reference[~((df_reference.vehicle_model == traj_features.vehicle_model) & \
                                  (df_reference.serial == traj_features.serial)) & \
                                  (df_reference.t_start < traj_features.t_start)].copy()
            features_cross = get_similarity_features_one_traj(traj_features,df_ref, df_threat_scores, 
                                                            k=k,metric=metric)
            features_cross = {'cross_' + key: value for key, value in features_cross.items()}

            features.update(features_cross)

        results.append(features)

    df_sim_features = pd.DataFrame(results)
    return df_sim_features