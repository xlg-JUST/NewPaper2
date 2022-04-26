from Tools import *

path = r'Experiment Results/'

# Mix
d1 = pd.read_csv(path+'Ours/Mix_project_Design.csv')

ours_def1 = [0.78119349,0.793233083,0.757352941,0.802238806,0.81409002,0.785578748,0.750413223,0.779411765,0.782945736,0.784090909]

ours_ref1 = [0.648648649,0.666666667,0.670886076,0.649006623,0.680272109,0.656716418,0.657718121,0.687116564,0.657718121,0.641025641]

ggnn_def1 = [0.737957114,0.763107682,0.762019139,0.727864991,0.729628868,0.736458472,0.749090698,0.709397166,0.724420421,0.713961261]

ggnn_ref1 = [0.496594447,0.621161368,0.533286462,0.515672629,0.550514522,0.452126311,0.488502446,0.51157829,0.59597597,0.56211315]

cohen, p_value, cliff = significance(ours_ref1, ggnn_ref1)
print(cohen)
print(p_value)
print(cliff)