
import numpy as np
import math
from haversine import haversine, Unit

def deg_to_rad(deg):
    return deg * (math.pi / 180)


def compute_average_speed_2points(lat1, lon1, alt1, time1, lat2, lon2, alt2, time2):
    # Convert degrees to radians
    phi1 = deg_to_rad(lat1)
    lambda1 = deg_to_rad(lon1)
    phi2 = deg_to_rad(lat2)
    lambda2 = deg_to_rad(lon2)
    
    # Earth's radius in km
    R = 6371.0
    
    # Cartesian coordinates for Point 1
    x1 = (R + alt1/1000) * math.cos(phi1) * math.cos(lambda1)
    y1 = (R + alt1/1000) * math.cos(phi1) * math.sin(lambda1)
    z1 = (R + alt1/1000) * math.sin(phi1)
    
    # Cartesian coordinates for Point 2
    x2 = (R + alt2/1000) * math.cos(phi2) * math.cos(lambda2)
    y2 = (R + alt2/1000) * math.cos(phi2) * math.sin(lambda2)
    z2 = (R + alt2/1000) * math.sin(phi2)
    
    # Euclidean distance between Point 1 and Point 2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    # Time difference
    time_difference = (time2 - time1).total_seconds() 
    
    # Calculate average speed
    average_speed = distance / time_difference if time_difference != 0 else 0
    
    return average_speed

def compute_speed(timestamps, longitudes, latitudes,altitudes):
    """
    Compute the speed between consecutive points in a trajectory.
    Inputs are lists of timestamps, longitudes, and latitudes.
    """
    
    speeds = []
    for i in range(1, len(timestamps)):
        timestamp1 = timestamps[i-1]
        timestamp2 = timestamps[i]
        lon1 = longitudes[i-1]
        lon2 = longitudes[i]
        lat1 = latitudes[i-1]
        lat2 = latitudes[i]
        alt1 = altitudes[i-1]
        alt2 = altitudes[i]
        speed = compute_average_speed_2points(lat1, lon1, alt1, timestamp1, lat2, lon2, alt2, timestamp2)

        speeds.append(speed)
    
    return speeds

def get_speed_features(group, n_bins = 10):

    if len(group) > 1:
        timestamps = group.timestamp
        heights = group['Height above take-off']
        longitudes = group.longitude
        latitudes = group.latitude
        speeds = compute_speed(timestamps.tolist(),
                            longitudes.tolist(),
                            latitudes.tolist(),
                            heights.tolist())
        
        sp_start, sp_end = speeds[0],speeds[-1]
        sp_mean,sp_std = np.mean(speeds),np.std(speeds)
        sp_cnts, bin_edges = np.histogram(speeds, bins=n_bins)
        sp_cnts = sp_cnts/sp_cnts.sum()
        sp_results = {'sp_b' + str(k):v for k,v in zip(range(len(sp_cnts)),sp_cnts)}
    else:
        sp_results = {'sp_b' + str(k):None for k,v in zip(range(n_bins),[0]*n_bins)}
        sp_start, sp_end, sp_mean, sp_std = None, None, None, None

    features = {
        'sp_start': sp_start,
        'sp_end': sp_end,
        'sp_mean': sp_mean,
        'sp_std': sp_std,
    }
    features.update(sp_results)

    return features