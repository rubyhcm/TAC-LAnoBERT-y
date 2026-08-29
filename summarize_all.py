import pandas as pd
df = pd.read_csv('summary.csv')
print("--- ALL BGL_lanobert ---")
print(df[df['model'] == 'BGL_lanobert'].to_string())
print("\n--- ALL BGL_tac_v2_2epochs ---")
print(df[df['model'] == 'BGL_tac_v2_2epochs'].to_string())
