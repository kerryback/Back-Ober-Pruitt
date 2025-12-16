"""Check memory usage of KP14 arrays."""
import pickle
import numpy as np

# Load the arrays file
with open('outputs/kp14_0_arrays.pkl', 'rb') as f:
    data = pickle.load(f)

arr_tuple = data['arr_tuple']
N = data['N']
T = data['T']

print(f"N = {N}, T = {T}")
print(f"Number of arrays in arr_tuple: {len(arr_tuple)}")
print("\nArray shapes and memory usage:")
print("="*70)

names = ['K', 'book', 'op_cashflow', 'x', 'z', 'eps', 'uj', 'chi',
         'rate', 'high', 'Et_G', 'EtA', 'alph', 'Et_z_alph',
         'price', 'ret', 'eret', 'lambda_f',
         'loadings_z_taylor', 'loadings_x_taylor',
         'loadings_z_proj', 'loadings_x_proj']

total_gb = 0
for i, arr in enumerate(arr_tuple):
    if isinstance(arr, np.ndarray):
        size_gb = arr.nbytes / 1e9
        total_gb += size_gb
        print(f"{names[i]:20s}: {str(arr.shape):30s} = {size_gb:6.2f} GB")
    else:
        print(f"{names[i]:20s}: (not an array, type={type(arr).__name__})")

print("="*70)
print(f"Total arr_tuple size: {total_gb:.2f} GB")
print(f"\nMemory needed for K**alpha: ~6.32 GB (additional)")
print(f"Total memory at error: {total_gb:.2f} + 6.32 = {total_gb + 6.32:.2f} GB")
