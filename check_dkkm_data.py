import pickle
import pandas as pd
import numpy as np

# Load all three DKKM files
files = ['outputs/bgn_0_dkkm_6.pkl', 'outputs/bgn_0_dkkm_36.pkl', 'outputs/bgn_0_dkkm_360.pkl']
all_data = []

for f in files:
    with open(f, 'rb') as file:
        data = pickle.load(file)
        df = data['dkkm_stats'].copy()
        df['panel'] = 0
        df['factors'] = data['nfeatures']
        all_data.append(df)
        print(f"\nFile: {f}")
        print(f"  nfeatures: {data['nfeatures']}")
        print(f"  Shape: {df.shape}")

combined = pd.concat(all_data, ignore_index=True)
print('\n' + '='*70)
print('COMBINED DATA')
print('='*70)
print(f'\nShape: {combined.shape}')
print(f'\nColumns: {list(combined.columns)}')
print(f'\nFirst 20 rows:')
print(combined.head(20))
print(f'\nUnique alpha values: {sorted(combined["alpha"].unique())}')
print(f'Unique factors values: {sorted(combined["factors"].unique())}')
print(f'\nValue counts for (alpha, factors):')
print(combined.groupby(['alpha', 'factors']).size())

# Now simulate the processing in analysis.py
print('\n' + '='*70)
print('AFTER PROCESSING (like analysis.py)')
print('='*70)

# Rename columns
combined = combined.rename(columns={"alpha": "kappa"})

# Calculate Sharpe
combined["sharpe"] = combined["mean"] / combined["stdev"]

# Calculate hjd_realized (already have hjd)
combined["hjd_realized"] = combined["hjd"] ** 2

# Aggregate to panel level
aggregated = combined.groupby(["kappa", "factors", "panel"])[["sharpe", "hjd_realized"]].mean().reset_index()
aggregated["hjd"] = np.sqrt(aggregated.hjd_realized)

print(f'\nAggregated shape: {aggregated.shape}')
print(f'\nAggregated data:')
print(aggregated)

# Filter like in create_dkkm_figure
lb, ub = 0.01, 0.1
filtered = aggregated[(aggregated['kappa'] >= lb) & (aggregated['kappa'] <= ub)].copy()
print(f'\nFiltered shape: {filtered.shape}')
print(f'\nFiltered data:')
print(filtered)

# Group by kappa and factors for plotting
mean_sharpe = filtered.groupby(['kappa', 'factors'])['sharpe'].mean().reset_index()
print(f'\nMean sharpe for plotting:')
print(mean_sharpe)
