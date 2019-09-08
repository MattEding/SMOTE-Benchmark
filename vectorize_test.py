import argparse
import textwrap
from timeit import timeit

import numpy as np
import pandas as pd

from imblearn.datasets import fetch_datasets
import imblearn.datasets._zenodo as zenodo
from imblearn.over_sampling import SMOTE


def trial(name, sampling_strategy, k_neighbors, n_jobs):
    setup = f'''
    from imblearn.datasets import fetch_datasets
    from imblearn.over_sampling import SMOTE
    
    sampling_strategy = '{sampling_strategy}'
    k_neighbors = {k_neighbors}
    n_jobs = {n_jobs}
    dataset = fetch_datasets()['{name}']
    X, y = dataset.data, dataset.target
    smote = SMOTE(sampling_strategy, k_neighbors=k_neighbors, n_jobs=n_jobs, random_state=0)
    '''
    setup = textwrap.dedent(setup).strip()
    t = timeit('smote.fit_resample(X, y)', setup=setup, number=100)

    dataset = fetch_datasets()[name]
    X, y = dataset.data, dataset.target
    smote = SMOTE(sampling_strategy, k_neighbors=k_neighbors, n_jobs=n_jobs, random_state=0)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    idx = -len(X)
    X_new, y_new = X_resampled[idx:], y_resampled[idx:]
    return X_new, y_new, t


def all_trials(sampling_strategy, k_neighbors, n_jobs):
    time = []
    xstd = []
    ystd = []
    xmean = []
    ymean = []
    xmax = []
    ymax = []
    xmin = []
    ymin = []

    ids = range(1, 28)
    index = pd.Index([zenodo.MAP_ID_NAME[i] for i in range(1, 28)], name='Name')

    for i in ids:
        name = zenodo.MAP_ID_NAME[i]
        X, y, t = trial(name, sampling_strategy, k_neighbors, n_jobs)
        time.append(t)
        xstd.append(X.std())
        ystd.append(y.std())
        xmean.append(X.mean())
        ymean.append(y.mean())
        xmax.append(X.max())
        ymax.append(y.max())
        xmin.append(X.min())
        ymin.append(y.min())

    df = pd.DataFrame(
        dict(
            Time=time,
            X_STD=xstd,
            y_STD=ystd,
            X_Mean=xmean,
            y_Mean=ymean,
            X_Max=xmax,
            y_Max=ymax,
            X_Min=xmin,
            y_Min=ymin,
        ),
        index=index
    )
    return df


def main():
    parser = argparse.ArgumentParser(zenodo.__doc__)
    parser.add_argument('dataset', help='zenodo dataset name or ID')
    parser.add_argument('--n_jobs', '-j', type=int, help='n_jobs for SMOTE')
    choices = ['minority', 'not majority', 'not majority', 'all', 'auto', 'none']
    parser.add_argument('--sampling_strategy', '-s', default='auto', choices=choices, help='sampling_strategy for SMOTE')
    parser.add_argument('--k_neighbors', '-k', default=5, type=int, help='k_neighbors for SMOTE')
    parser.add_argument('--file', '-f', help='file to save results to')
    args = parser.parse_args()

    if args.dataset in ['0', 'all']:
        result = all_trials(args.sampling_strategy, args.k_neighbors, args.n_jobs)
    else:
        try:
            name = zenodo.MAP_ID_NAME[int(args.dataset)]
        except Exception:
            name = args.dataset
        
        X, y, t = trial(name, args.sampling_strategy, args.k_neighbors, args.n_jobs)
        result = pd.Series(
            dict(
                Time=t,
                X_STD=X.std(),
                y_STD=y.std(),
                X_Mean=X.mean(),
                y_Mean=y.mean(),
                X_Max=X.max(),
                y_Max=y.max(),
                X_Min=X.min(),
                y_Min=y.min(),
            )
        )
    
    if args.file is not None:
        result.to_pickle(f'{args.file}.pkl')
    else:
        print(result)


if __name__ == '__main__':
    main()
