import numpy as np

def get_noflyzones_feats(group, n_bins = 10):

    n_records = len(group)

    isin_noflyzone_mask = ~group.isin_nofly_zone.isna()

    enter_noflyzone = isin_noflyzone_mask.sum() > 0
    perc_noflyzone = isin_noflyzone_mask.sum()/n_records

    distances = group.distance_closest_noflyzone.to_numpy()
    distances[isin_noflyzone_mask] = 0

    nf_cnts, bin_edges = np.histogram(distances, bins=n_bins)
    nf_cnts = nf_cnts/nf_cnts.sum()
    
    nf_results = {'nf_b' + str(k):v for k,v in zip(range(len(nf_cnts)),nf_cnts)}

    nf_d_min = np.min(distances)
    nf_d_max = np.max(distances)
    nf_d_mean, nf_d_std = np.mean(distances), np.std(distances)

    features = {
        'enter_noflyzone': enter_noflyzone,
        'perc_noflyzone': perc_noflyzone,
        'nf_d_min': nf_d_min,
        'nf_d_max': nf_d_max,
        'nf_d_mean': nf_d_mean,
        'nf_d_std': nf_d_std,
    }

    features.update(nf_results)
    return features   