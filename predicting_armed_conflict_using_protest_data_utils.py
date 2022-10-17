# Data utils 

import numpy as np  
import pandas as pd 
import logging
from typing import Tuple
import warnings
import statsmodels.api as sm  # type: ignore

from typing import Any, List

def prob_to_odds(p: Any, clip=True) -> Any:
    """ Cast probability into odds """

    if isinstance(p, list):
        p = np.array(p)

    if clip:
        offset = 1e-10
        offset = 1e-10
        upper = 1 - offset
        lower = 0 + offset
        p = np.clip(p, lower, upper)

    # Check for probs greq 1 because odds of 1 is inf which might break things
    if np.any(p >= 1):
        msg = "probs >= 1 passed to get_odds, expect infs"
        warnings.warn(msg)

    odds = p / (1 - p)
    return odds


def prob_to_logodds(p: Any) -> Any:
    """ Cast probability to log-odds """
    return np.log(prob_to_odds(p))


def odds_to_prob(odds: Any) -> Any:
    """ Cast odds ratio to probability """
    return odds / (odds + 1)


def logodds_to_prob(logodds: Any) -> Any:
    """ Cast logodds to probability """
    return odds_to_prob(np.exp(logodds))

def calibrate_prob(
    s_test_pred: pd.Series, s_calib_pred: pd.Series, s_calib_actual: pd.Series
) -> pd.Series:
    """ Calibrate s_test_pred

    First predictions are transformed into logodds.
    Then a logit model is fit on
    "actual_outcomes ~ alpha + beta*logodds(p_calib)".
    Then alpha and beta are applied to test predictions like
    A =  e^(alpha+(beta*p_test))
    p_test_calibrated = A/(A+1)

    See: https://en.wikipedia.org/wiki/Logistic_regression

    """

    def _get_scaling_params(
        s_calib_actual: pd.Series, s_calib: pd.Series
    ) -> Tuple[float, float]:
        """ Gets scaling params """

        y = np.array(s_calib_actual)
        intercept = np.ones(len(s_calib))
        X = np.array([intercept, s_calib]).T

        model = sm.Logit(y, X).fit(disp=0)
        beta_0 = model.params[0]
        beta_1 = model.params[1]

        return beta_0, beta_1

    def _apply_scaling_params(
        s_test: pd.Series, beta_0: float, beta_1: float
    ) -> pd.Series:
        """ Scale logodds in s_test using intercept and beta"""
        numerator = np.exp(beta_0 + (beta_1 * s_test))
        denominator = numerator + 1
        scaled_probs = numerator / denominator

        return scaled_probs

    def _check_inputs(
        s_test_pred: pd.Series,
        s_calib_pred: pd.Series,
        s_calib_actual: pd.Series,
    ) -> None:
        """ Check that inputs have valid names and could be proabilities """

        if (
            s_test_pred.min() < 0
            or s_test_pred.max() > 1
            or s_calib_pred.min() < 0
            or s_calib_pred.max() > 1
        ):
            raise RuntimeError(
                "Probabilities outside (0,1) range were passed to calibrate"
            )
            
        if not s_calib_pred.name != s_test_pred.name:
            warnings.warn(f"{s_calib_pred.name} != {s_test_pred.name}")
        if s_test_pred.isnull().sum() > 0:
            _log_missing_indices(s_test_pred)
            raise RuntimeError("Missing values in s_test_pred")
        if s_calib_pred.isnull().sum() > 0:
            _log_missing_indices(s_calib_pred)
            raise RuntimeError("Missing values in s_calib_pred")
        if s_calib_actual.isnull().sum() > 0:
            _log_missing_indices(s_calib_actual)
            raise RuntimeError("Missing values in s_calib_actual")

        if (
            not len(s_calib_pred) == len(s_calib_actual)
            or len(s_calib_pred.index.difference(s_calib_actual.index)) > 0
        ):
            raise RuntimeError(
                f"len(s_calib_pred): {len(s_calib_pred)} "
                f"len(s_calib_actual): {len(s_calib_actual)} "
                f"index diff: "
                f"{s_calib_pred.index.difference(s_calib_actual.index)}"
                f"s_calib_pred.head() : {s_calib_pred.head()}"
                f"s_calib_pred.tail() : {s_calib_pred.tail()}"
                f"s_calib_actual.head() : {s_calib_actual.head()}"
                f"s_calib_actual.tail() : {s_calib_actual.tail()}"
            )

    _check_inputs(s_test_pred, s_calib_pred, s_calib_actual)

    beta_0, beta_1 = _get_scaling_params(
        s_calib_actual=s_calib_actual,
        s_calib= prob_to_logodds(s_calib_pred.copy()),
    )
    if beta_1 < 0:
        warnings.warn(f"Beta_1 < 0. Very weak {s_calib_pred.name} ?")

    s_test_pred_scaled = _apply_scaling_params(
        prob_to_logodds(s_test_pred.copy()), beta_0, beta_1
    )
    return s_test_pred_scaled

def resampled_index(
    df,
    cols,
    share_positives,
    share_negatives,
    threshold,
    random_state,
):
    """ Resample a dataframe with respect to cols

    Resampling is a technique for changing the positive/negative balance
    of a dataframe. Positives are rows where any of the specified cols
    are greater than the threshold. Useful for highly unbalanced
    datasets where positive outcomes are rare.

    """

    # Negatives are rows where all cols are close to zero
    mask_negatives = np.isclose(df[cols], threshold).max(axis=1)
    # Positives are all the others
    mask_positives = ~mask_negatives

    df_positives = df.loc[mask_positives]
    df_negatives = df.loc[mask_negatives]

    len_positives = len(df_positives)
    len_negatives = len(df_negatives)

    n_positives_wanted = int(share_positives * len_positives)
    n_negatives_wanted = int(share_negatives * len_negatives)

    replacement_pos = share_positives > 1
    replacement_neg = share_negatives > 1
    df = pd.concat(
        [
            df_positives.sample(n=n_positives_wanted, replace=replacement_pos, random_state=random_state),
            df_negatives.sample(n=n_negatives_wanted, replace=replacement_neg, random_state=random_state),
        ]
    )
    
    
    return df

def find_index(dicts, key, value):
    class Null: pass
    for i, d in enumerate(dicts):
        if d.get(key, Null) == value:
            return i
    else:
        raise ValueError('no dict with the key and value combination found')
        
def RetrieveFromList_X(Datasets,name):
    
    df = Datasets[find_index(Datasets, 'Name', name)]['df']
    df = df.drop(Datasets[find_index(Datasets, 'Name', name)]['df'].filter(regex='dep').columns, axis=1)
    
    return df

def RetrieveFromList_y(Datasets,name,depv):
    
    df = Datasets[find_index(Datasets, 'Name', name)]['df']
    df = df[depv]
    
    return df