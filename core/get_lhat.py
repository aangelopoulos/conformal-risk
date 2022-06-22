import numpy as np
import pdb

"""
    calib loss table from small loss to large loss
"""
def get_lhat(calib_loss_table, lambdas_table, alpha):
    n = calib_loss_table.shape[0]
    rhat = calib_loss_table.mean(axis=0)
    lhat_idx = max(np.argmax(((n/(n+1)) * rhat + 1/(n+1) ) > alpha) - 1, 0) # Can't be -1.
    return lambdas_table[lhat_idx]
