import json
from pathlib import Path
p = Path(r'c:\iris classification\notebooks\01_iris_classification.ipynb')
d = json.loads(p.read_text(encoding='utf-8'))
print('cells', len(d['cells']))
for i, c in enumerate(d['cells'][:5], 1):
    print(i, c['cell_type'], c.get('source', [])[:1])
