import numpy as np

def get_asset_features(group, n_bins = 10):
    group = group.infer_objects(copy=False)
    group = group.fillna(0)

    max_assets_value = group.isin_asset_value.max()
    

    ass_cnts, bin_edges = np.histogram(group.radius_0, bins=n_bins)
    ass_cnts = ass_cnts/ass_cnts.sum()
    ass_results = {'av_r0_b' + str(k):v for k,v in zip(range(len(ass_cnts)),ass_cnts)}
    ass_results.update({'av_r0_mean': group.radius_0.mean(),
                        'av_r0_std': group.radius_0.std()})
    
    ass_cnts, bin_edges = np.histogram(group.radius_1, bins=n_bins)
    ass_cnts = ass_cnts/ass_cnts.sum()
    ass_results.update({'av_r1_b' + str(k):v for k,v in zip(range(len(ass_cnts)),ass_cnts)})
    ass_results.update({'av_r1_mean': group.radius_1.mean(),
                        'av_r1_std': group.radius_1.std()})
    
    ass_cnts, bin_edges = np.histogram(group.radius_2, bins=n_bins)
    ass_cnts = ass_cnts/ass_cnts.sum()
    ass_results.update({'av_r2_b' + str(k):v for k,v in zip(range(len(ass_cnts)),ass_cnts)})
    ass_results.update({'av_r2_mean': group.radius_2.mean(),
                        'av_r2_std': group.radius_2.std()})
    
    features = {
        'av_max': max_assets_value,
    }
    features.update(ass_results)
    
    return features
