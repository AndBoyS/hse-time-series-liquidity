import numpy as np


def get_best_cfs_features(df, features, target):
    '''
    Нахождение лучшего сабсета признаков на основе CFS - Correlation-based Feature Selection.
    По сути ищем признаки, сильно коррелирующие с таргетом, и слабо коррелирубщие между собой.
    Для нахождение сабсета используется жадный алгоритм, аналогичный sklearn.feature_selection.SequentialFeatureSelector с direction='forward'
    '''

    features_left = features.copy()
    features_subset = []
    best_merit = 0

    while features_left:

        best_feature = None

        for feature in features_left:
            current_subset = features_subset + [feature]
            merit = get_cfs(df, current_subset, target)

            if merit > best_merit:
                best_feature = feature
                best_merit = merit

        if best_feature is None:
            break
        else:
            features_left.remove(best_feature)
            features_subset.append(best_feature)

    return features_subset, best_merit


def get_cfs(df, subset, label):
    '''
    Значение CFS для конкретного сабсета признаков
    '''
    k = len(subset)

    all_corr = df[subset+[label]].corr()
    # average feature-class correlation
    rcf = (all_corr[label][subset]).abs().mean()

    # average feature-feature correlation
    corr = all_corr[subset].loc[subset]

    if corr.shape[0] > 1:
        corr.values[np.tril_indices_from(corr.values)] = np.nan

    corr = abs(corr)
    rff = corr.unstack().mean()

    return (k * rcf) / (k + k * (k-1) * rff) ** 0.5
