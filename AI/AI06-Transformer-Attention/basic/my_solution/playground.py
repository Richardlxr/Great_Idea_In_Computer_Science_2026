import numpy as np
from my_attention import experiment_causality, experiment_q5, experiment_single

np.set_printoptions(precision=6, suppress=True)
experiment_q5()
experiment_single()
experiment_causality()
