import numpy as np
from scipy.optimize import brentq
import pdb

"""
    Gets the value of lambda hat that controls the marginal risk for a monotone risk function.
    The calib loss table should be ordered from small loss to large loss
"""
def get_lhat(calib_loss_table, lambdas, alpha, B=1):
    n = calib_loss_table.shape[0]
    rhat = calib_loss_table.mean(axis=0)
    lhat_idx = max(np.argmax(((n/(n+1)) * rhat + B/(n+1) ) > alpha) - 1, 0) # Can't be -1.
    return lambdas[lhat_idx]

"""
    Gets the value of lambda hat that controls the marginal risk for a selective risk function.
    The risk should be approximately monotone non-increasing in lambda
"""
def get_lhat_selective(calib_losses, calib_goodnesses, lambdas, alpha, B=1):
    rhatplus = 0
    reversed_lambdas = lambdas[::-1]
    for i in range(lambdas.shape[0]):
        selected = calib_goodnesses > reversed_lambdas[i]
        curr_rhat = calib_losses[selected].mean()
        if np.isnan(curr_rhat):
            continue
        rhatplus = np.maximum(curr_rhat, rhatplus)
        if rhatplus + B/selected.sum() > alpha:
            break
    return reversed_lambdas[max(i-1,0)]
