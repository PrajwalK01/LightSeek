"""
LightSeek — Download More Quarters (W3 data boost)
Author: Prajwal K / Team Integral X
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightkurve as lk
import numpy as np
import pandas as pd

CONFIRMED_TARGETS = [
    ('Kepler-17',2),('Kepler-17',3),('Kepler-22',2),('Kepler-22',3),
    ('Kepler-62',2),('Kepler-186',2),('Kepler-442',2),('Kepler-452',2),
    ('Kepler-296',2),('Kepler-438',2),('Kepler-440',2),('Kepler-61',2),
    ('Kepler-174',2),('Kepler-283',2),('Kepler-298',2),('Kepler-314',2),
]

FP_TARGETS = [
    ('KIC 3544595',2),('KIC 4544670',2),('KIC 5090937',2),
    ('KIC 5376836',2),('KIC 5383248',2),('KIC 5459778',2),
    ('KIC 5652983',2),('KIC 5735762',2),('KIC 6021275',2),
    ('KIC 6278762',2),('KIC 6442340',2),('KIC 6521045',2),
    ('KIC 6603043',2),('KIC 6669809',2),('KIC 6851425',2),
    ('KIC 7021681',2),
]

CONFIRMED_DIR = 'data/processed/kepler/confirmed'
FP_DIR        = 'data/processed/kepler/false_positive'
MANIFEST      = 'data/processed/manifest.csv'


def download_one(target, quarter, save_dir, label, label_name):
    try:
        result = lk.search_lightcurve(target, mission='kepler', quarter=quarter)
        if len(result) == 0:
            raise ValueError('not found')
        lc = result[0].download()
        lc = lc.remove_nans().remove_outliers().normalize()
        time = lc.time.value
        flux = lc.flux.value
        flux = np.nan_to_num(flux, nan=1.0)
        t_uni = np.linspace(time.min(), time.max(), 1000)
        flux  = np.interp(t_uni, time, flux)
        safe  = target.replace(' ','_').replace('-','_') + f'_q{quarter}'
        os.makedirs(save_dir, exist_ok=True)
        np.save(f'{save_dir}/{safe}_flux.npy', flux)
        np.save(f'{save_dir}/{safe}_time.npy', t_uni)
        print(f'  Saved {target} Q{quarter}')
        return {
            'target': target, 'label': label,
            'label_name': label_name,
            'flux_path': f'{save_dir}/{safe}_flux.npy',
            'time_path': f'{save_dir}/{safe}_time.npy'
        }
    except Exception as e:
        print(f'  FAILED {target} Q{quarter}: {e}')
        return None


def main():
    records = []
    total = len(CONFIRMED_TARGETS) + len(FP_TARGETS)
    done  = 0

    print(f'\nDownloading {total} additional light curves...\n')

    for target, q in CONFIRMED_TARGETS:
        r = download_one(target, q, CONFIRMED_DIR, 1, 'CONFIRMED')
        if r: records.append(r)
        done += 1
        print(f'  Progress: {done}/{total}')

    for target, q in FP_TARGETS:
        r = download_one(target, q, FP_DIR, 0, 'FALSE POSITIVE')
        if r: records.append(r)
        done += 1
        print(f'  Progress: {done}/{total}')

    old      = pd.read_csv(MANIFEST)
    new      = pd.DataFrame(records)
    combined = pd.concat([old, new], ignore_index=True)
    combined.to_csv(MANIFEST, index=False)

    print(f'\nTotal dataset: {len(combined)} samples')
    print(f'Confirmed:     {len(combined[combined.label==1])}')
    print(f'False pos:     {len(combined[combined.label==0])}')


if __name__ == '__main__':
    main()