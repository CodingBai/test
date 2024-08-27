from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import McmcDina

def CD_by_MCMC_DINA(score,q):
    q = np.transpose(q)
    print("Score's shape", score.shape)
    print("Q's.shape", q.shape)
    model = McmcDina(attrs=q, score=score, max_iter=5000, burn=2500)
    est_skills, est_no_s, est_g = model.mcmc()
    return est_skills