from utils import compute_geometric_distance
def get_basic_features(group):

    n_records = len(group)

    timestamps = group.timestamp
    t_start, t_end = timestamps.min(), timestamps.max()
    duration = (t_end-t_start).total_seconds() #/60    

    distance = compute_geometric_distance(group)

    source_type = 0 if group.source_type.mode()[0]=='RF' else 1

    return {
        'n_records':n_records,
        'duration':duration,
        'distance':distance,
        'communication_channel':source_type
    }
