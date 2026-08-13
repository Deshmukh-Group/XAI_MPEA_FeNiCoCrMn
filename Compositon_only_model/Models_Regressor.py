import joblib
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import pandas as pd

import numpy as np
from numpy import absolute, mean, std


def regressor_compare(X_train: np.ndarray, y_train: pd.DataFrame, X_test: np.ndarray,
                      y_test: pd.DataFrame, cv_repeats: int = 3,
                      skip_cv: bool = False, tag: str = '') -> pd.DataFrame:
    """
    Lightweight script to test many models
    :param X_train: training split (after Preprocessing)
    :param y_train: training target vector
    :param X_test: test split (after Preprocessing)
    :param y_test: test target vector
    :param cv_repeats: repeats for RepeatedKFold.  The cross-validation dominates
                       the runtime, so lower this for a quick pass.
    :param skip_cv: fit and score only, with no cross-validation
    :param tag: output directory for the fitted models, e.g. 'Bulk' or 'USFE',
                so that runs for different targets do not overwrite each other.
                Must already exist.
    :return: DataFrame of cross-validation scores
    """
    prefix = f'{tag}/model_' if tag else 'model_'

    models = [
        ('RF', RandomForestRegressor()),
        ('SGD', SGDRegressor()),
        ('MLP', MLPRegressor(max_iter=10000)),
        ('BR', BayesianRidge()),
    ]

    results = []
    dfs = pd.DataFrame()
    for name, model in models:
        # define the direct MultiOutput wrapper model
        wrapper = MultiOutputRegressor(model)
        if skip_cv:
            print('Model_Validation: %s ; skipped (skip_cv)' % name)
            results.append([name, float('nan'), float('nan')])
        else:
            # define the evaluation procedure
            cv = RepeatedKFold(n_splits=10, n_repeats=cv_repeats, random_state=1)
            # evaluate the model and collect the scores
            n_scores = cross_val_score(wrapper, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
            # force the scores to be positive
            n_scores = absolute(n_scores)
            # summarize performance
            print('Model_Validation: %s ;' % name, 'MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
            results.append([name, mean(n_scores), std(n_scores)])

        df_temp = pd.DataFrame(results, columns=['Model_Name', 'MAE_mean', 'MAE_std'])
        dfs = pd.concat([dfs, df_temp], ignore_index=True)

        # Training and Testing
        wrapper.fit(X_train, y_train)
        y_pred = wrapper.predict(X_test)
        MAE = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
        R2 = r2_score(y_test, y_pred, multioutput='uniform_average')
        print('Model Name: %s ;' % name, 'Testing MAE: %.3f' % MAE, 'R2: %.3f' % R2)
        # save the model to disk
        joblib.dump(wrapper, prefix + name + '.pkl')

    return dfs


def plot_regression_results(ax, y_true, y_pred, title, scores,
                            axis_label='USFE (mJ/m$^2$)', err_max=None):
    """Scatter plot of the predicted vs true targets.

    :param axis_label: quantity and unit, e.g. 'Bulk modulus (GPa)'.  The colour
                       scale spans the absolute error, so its range has to match
                       the property: errors are ~0.1-3 GPa for the elastic
                       constants but ~5-15 mJ/m^2 for the USFE.
    :param err_max: upper limit of the error colour scale.  None picks it from
                    the data (95th percentile of the absolute error).
    """
    c = abs(y_pred - y_true)
    if err_max is None:
        err_max = float(np.percentile(c, 95)) if c.size else 1.0
    err_max = max(err_max, 1e-9)
    ax.plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--r", linewidth=2
    )
    sc = ax.scatter(y_true, y_pred, c=c, cmap='jet_r', vmin=0, vmax=err_max, s=30)
    ax.set_aspect('equal')  # Set axes to be square

    #ax.spines["top"].set_visible(False)
    #ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.spines["left"].set_position(("outward", 10))
    # ax.spines["bottom"].set_position(("outward", 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_xlabel(f"MD Calculated {axis_label}", fontsize=18)
    ax.set_ylabel(f"ML Predicted {axis_label}", fontsize=18)
    extra = plt.Rectangle(
        (0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0
    )
    # legend font size 16
    ax.legend([extra], [scores], loc="upper left", fontsize=20, handlelength=0)
    # title = title + "\n Evaluation in {:.2f} seconds".format(elapsed_time)
    ax.set_title(title)
    # Add color bar, ticks spanning the actual error range of this property
    cbar = plt.colorbar(sc, shrink=0.8, ticks=np.linspace(0, err_max, 6))
    cbar.ax.set_ylabel('|error|', fontsize=16)
    cbar.ax.tick_params(labelsize=16)
