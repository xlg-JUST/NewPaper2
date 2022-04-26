import pandas as pd
from scipy.stats import ranksums
import numpy as np
path = r'Experiment Results/'


def significance(d1, d2):
    p_value = ranksums(d1, d2)[-1]
    m1, m2 = np.mean(d1), np.mean(d2)
    s1, s2 = np.std(d1), np.std(d2)
    s = np.sqrt((s1**2+s2**2)/2)
    cohen = (m1-m2)/s
    count1, count2 = 0, 0
    for elm1 in d1:
        for elm2 in d2:
            if elm1 > elm2:
                count1 += 1
            elif elm1 < elm2:
                count2 += 1
    cliff = (count1 - count2)/len(d1)/len(d2)
    return cohen, p_value, cliff


# Mix

#
# ours_def1 = [0.78119349,0.793233083,0.757352941,0.802238806,0.81409002,0.785578748,0.750413223,0.779411765,0.782945736,0.784090909]
#
# ours_ref1 = [0.648648649,0.666666667,0.670886076,0.649006623,0.680272109,0.656716418,0.657718121,0.687116564,0.657718121,0.641025641]
#
# ggnn_def1 = [0.737957114,0.763107682,0.762019139,0.727864991,0.729628868,0.736458472,0.749090698,0.709397166,0.724420421,0.713961261]
#
# ggnn_ref1 = [0.496594447,0.621161368,0.533286462,0.515672629,0.550514522,0.452126311,0.488502446,0.51157829,0.59597597,0.56211315]

# cohen, p_value, cliff = significance(ours_ref1, ggnn_ref1)
# print(cohen)
# print(p_value)
# print(cliff)
d1 = pd.read_csv(path+'WithinDe.csv', header=None).to_numpy()
ours = np.float32(d1[:, 0].reshape(-1))[:-1]


tar = ['Stanford', 'N-gramIDF', 'XGBoost', 'LSTM_Att', 'GGNN']
for i in range(len(tar)):
    cur = d1[:, i+1].reshape(-1)
    cur = cur[:-1]
    cohen, p_value, cliff = significance(ours, cur)
    print(tar[i])
    print(cohen, p_value, cliff)
