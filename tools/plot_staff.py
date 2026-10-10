"""Plot saved Staff evidence; no model, retrieval or evaluation is run."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
context = json.loads((ROOT/'traces/staff_context.json').read_text())
first = json.loads((ROOT/'traces/staff_test.json').read_text())
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
policies = context['policies']
axes[0].bar(list(policies), [r['context_tokens'] for r in policies.values()])
axes[0].set(title='Same captured pools: all 20/20', ylabel='Selected excerpt word tokens')
rows = first['metrics']['results']
confidence = sum(r['prediction']['confidence'] for r in rows)/len(rows)
accuracy = sum(r['correct'] for r in rows)/len(rows)
axes[1].plot([0, 1], [0, 1], '--', color='gray')
axes[1].scatter([confidence], [accuracy], s=70)
axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean confidence (one occupied bin)',
            ylabel='Observed accuracy', title='Original held-out run: overconfident')
risk = [r for r in first['risk_coverage'] if r['risk'] is not None]
axes[2].scatter([r['coverage'] for r in risk], [r['risk'] for r in risk])
axes[2].set(xlim=(0, 1.05), ylim=(0, 1), xlabel='Coverage', ylabel='Error rate',
            title='Constant confidence cannot rank risk')
fig.tight_layout()
fig.savefig(ROOT/'plots/staff_evaluation.png', dpi=140)
plt.close(fig)
