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
    lhat_idx = max(np.argmax(((n/(n+1)) * rhat + B/(n+1) ) >= alpha) - 1, 0) # Can't be -1.
    return lambdas[lhat_idx]
