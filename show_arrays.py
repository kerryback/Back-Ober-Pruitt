import pickle
import numpy as np

data = pickle.load(open('outputs/bgn_0_panel.pkl', 'rb'))
arr_tuple = data['arr_tuple']

print('arr_tuple contains 17 arrays:')
print()

names = [
    'r', 'mu', 'xi', 'sigmaj', 'chi', 'beta', 'corr_zj',
    'eret', 'ret', 'P', 'corr_zr', 'book', 'op_cash_flow',
    'loadings_mu_taylor', 'loadings_xi_taylor',
    'loadings_mu_proj', 'loadings_xi_proj'
]

for i, (name, arr) in enumerate(zip(names, arr_tuple)):
    if isinstance(arr, np.ndarray):
        print(f'  [{i:2d}] {name:20s}: shape={str(arr.shape):15s} dtype={arr.dtype}')
    else:
        print(f'  [{i:2d}] {name:20s}: {type(arr).__name__} = {arr}')

print()
print('Panel DataFrame columns:')
print(f'  {list(data["panel"].columns)}')
