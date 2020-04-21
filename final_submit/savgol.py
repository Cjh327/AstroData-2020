from scipy.signal import savgol_filter


def savgol_smooth(features):
    features = savgol_filter(features, window_length=7, polyorder=3)
    print("smoothed features:", features.shape)
    return features
