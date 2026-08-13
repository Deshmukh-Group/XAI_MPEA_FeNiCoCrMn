"""
Composition-only surrogate models for FeNiCoCrMn MPEAs: five at.% values in,
one property out, with no atomic-structure information.

    python ML_composition_only.py Bulk
    python ML_composition_only.py USFE
    python ML_composition_only.py USFE --skip-cv     # minutes instead of ~30

Run from this folder.  Redirect stdout to keep a log.

Input: data_pso_Mn.dat, the raw PSO log, 9996 rows x 11 columns.  Columns 1-5 are
the composition variables for Mn, Cr, Co, Fe, Ni, each stored minus its 5 at.%
lower bound; columns 6-11 are bulk modulus, C11, C12, C44, Young's modulus (GPa)
and USFE (eV per fault area).

Cleaning: +5 per element to recover at.%; drop rows whose composition does not sum
to 100 within 1e-4 at.% (the PSO wrote six decimals without renormalising); drop
rows carrying the failure sentinel 1000 or a negative value; drop every copy of a
repeated composition.  That last step leaves one row per composition, so the
random split below cannot put the same composition in both train and test.

Each run writes into a folder named after the target, so the two do not collide:

    Bulk/  or  USFE/
        scaler.pkl
        model_{RF,SGD,MLP,BR}.pkl
        results.csv
        fig_{NAME}_{train,test}.pdf
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from Models_Regressor import plot_regression_results, regressor_compare

COLS = ['c1', 'c2', 'c3', 'c4', 'c5',
        'Bulk', 'C11', 'C12', 'C44', 'Young', 'USFE']
ELEMS = ['Mn', 'Cr', 'Co', 'Fe', 'Ni']

# Keep only compositions that sum to 100 within this tolerance, i.e. 99.9999 to
# 100.0001 at.%.  The PSO did not renormalise, so a minority of rows overshoot.
SUM_TOL = 1e-4

# Fault area of the FeNiCoCrMn cell (not a copper cell); reproduces the
# manuscript's mean USFE of 345.65 mJ/m^2.
EV_TO_MJM2 = 1.6021766208e-19 / (44.054073e-10 * 25.43463e-10) * 1000

# axis label and unit conversion; add 'C11': ('$C_{11}$ (GPa)', 1.0) etc. as needed
TARGETS = {'Bulk': ('Bulk modulus (GPa)', 1.0),
           'USFE': ('USFE (mJ/m$^2$)', EV_TO_MJM2)}

ap = argparse.ArgumentParser()
ap.add_argument('target', choices=sorted(TARGETS))
ap.add_argument('--skip-cv', action='store_true', help='no cross-validation')
args = ap.parse_args()
label, convert = TARGETS[args.target]
outdir = args.target
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv('data_pso_Mn.dat', header=None, sep=r'\s+', names=COLS)
print(f'{len(df)} rows read')

df[ELEMS] = df[COLS[:5]].to_numpy() + 5
csum = df[ELEMS].sum(axis=1)
print(f'composition sums span {csum.min():.4f} to {csum.max():.4f} at.% '
      f'(the PSO wrote six decimals without renormalising)')

off_sum = (csum - 100).abs() > SUM_TOL
print(f'dropping {int(off_sum.sum())} rows whose composition does not sum to '
      f'100 +/- {SUM_TOL:g} at.%')
df = df[~off_sum]

bad = (df[COLS] == 1000).any(axis=1) | (df[COLS] < 0).any(axis=1)
print(f'dropping {int(bad.sum())} rows with the 1000 sentinel or a negative value')
df = df[~bad]

dup = df[ELEMS].round(6).duplicated(keep=False)
print(f'dropping {int(dup.sum())} rows sharing a repeated composition '
      f'({len(df[dup][ELEMS].drop_duplicates())} compositions evaluated twice)')
df = df[~dup]

X = df[ELEMS]
y = df[[args.target]] * convert
print(f'{len(X)} samples; {args.target} mean {y.to_numpy().mean():.3f} '
      f'std {y.to_numpy().std():.3f}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=34)
scaler = preprocessing.StandardScaler().fit(X_train)
joblib.dump(scaler, f'{outdir}/scaler.pkl')
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

regressor_compare(X_train, y_train, X_test, y_test,
                  skip_cv=args.skip_cv, tag=outdir)

rows = []
for name in ['RF', 'SGD', 'MLP', 'BR']:
    model = joblib.load(f'{outdir}/model_{name}.pkl')
    for split, Xs, ys in (('train', X_train, y_train), ('test', X_test, y_test)):
        pred = model.predict(Xs)
        mae = mean_absolute_error(ys, pred)
        r2 = r2_score(ys, pred)
        mse = mean_squared_error(ys, pred)
        print(f'{name:4s} {split:5s} MAE {mae:8.3f}  R2 {r2:7.4f}  MSE {mse:9.3f}')
        rows.append([name, split, mae, r2, mse])

        fig, ax = plt.subplots(figsize=(8, 8))
        score = (r'$R^2={:.3f}$' + '\n' + r'$MAE={:.2f}$').format(r2, mae)
        plot_regression_results(ax, np.ravel(ys), np.ravel(pred),
                                f'{name} ({split})', score, label)
        fig.savefig(f'{outdir}/fig_{name}_{split}.pdf', bbox_inches='tight')
        plt.close(fig)

pd.DataFrame(rows, columns=['model', 'split', 'MAE', 'R2', 'MSE']
             ).to_csv(f'{outdir}/results.csv', index=False)
print(f'wrote everything to {outdir}/')
