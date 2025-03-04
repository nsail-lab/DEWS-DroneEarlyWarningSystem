import numpy as np

def get_altitude_features(group, n_bins=10):
    
    height = group['Height above take-off']

    
    h_cnts, bin_edges = np.histogram(height, bins=n_bins)
    h_cnts = h_cnts/h_cnts.sum()
    h_mean, h_std = height.mean(), height.std()
    h_start,h_end = height.iloc[0], height.iloc[-1]
    
    h_results = {'h_b' + str(k):v for k,v in zip(range(len(h_cnts)),h_cnts)}
 
    features = {
        'h_start': h_start,
        'h_end': h_end,
        'h_mean': h_mean,
        'h_std': h_std,
    }
    features.update(h_results)
    return features