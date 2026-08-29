import os
import glob
import re
import pandas as pd

def parse_report(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    metrics = {}
    for line in content.split('\n'):
        if line.startswith('AUROC:'):
            metrics['AUROC'] = float(line.split(':')[1].strip())
        elif line.startswith('best_F1:'):
            metrics['best_F1'] = float(line.split(':')[1].strip())
        elif line.startswith('best_threshold:'):
            metrics['best_threshold'] = float(line.split(':')[1].strip())
        elif line.strip().startswith('1 ') or line.strip().startswith('1    '):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    metrics['Precision'] = float(parts[1])
                    metrics['Recall'] = float(parts[2])
                    metrics['F1-score'] = float(parts[3])
                except:
                    pass
    return metrics

data = []
for d in ['outputs/BGL_lanobert/results', 'outputs/BGL_tac_v2_2epochs/results']:
    model_name = d.split('/')[1]
    for filepath in glob.glob(f"{d}/*_report.txt"):
        filename = os.path.basename(filepath)
        metrics = parse_report(filepath)
        metrics['model'] = model_name
        metrics['file'] = filename.replace('_report.txt', '')
        data.append(metrics)

df = pd.DataFrame(data)
df = df.sort_values(by=['model', 'best_F1'], ascending=[True, False])
df.to_csv('summary.csv', index=False)
print("--- BGL_lanobert Top 10 by best_F1 ---")
print(df[df['model'] == 'BGL_lanobert'][['file', 'best_F1', 'Precision', 'Recall', 'AUROC']].head(10).to_string())
print("\n--- BGL_tac_v2_2epochs Top 10 by best_F1 ---")
print(df[df['model'] == 'BGL_tac_v2_2epochs'][['file', 'best_F1', 'Precision', 'Recall', 'AUROC']].head(10).to_string())
