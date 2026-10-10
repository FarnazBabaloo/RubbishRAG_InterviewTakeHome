"""Compare saved probabilities on the same 24 audit questions; no retrieval."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'traces/route_calibration_audit.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4))
metrics=['decision_brier','decision_ece'];x=[0,1]
for offset,key,label in [(-.18,'raw','Previous confidence'),(.18,'calibrated','Fitted groups')]:
 axes[0].bar([i+offset for i in x],[r[key][m] for m in metrics],width=.36,label=label)
axes[0].set_xticks(x,['Decision Brier','Decision ECE']);axes[0].set(title='Same 24 audit queries; lower is better',ylabel='Error');axes[0].legend()
for key,label in [('raw','Previous confidence'),('calibrated','Fitted groups')]:
 points=[p for p in r[key]['risk_coverage'] if p['risk'] is not None]
 axes[1].plot([p['coverage'] for p in points],[p['risk'] for p in points],'o-',label=label)
axes[1].set(xlabel='Coverage',ylabel='Observed error rate',title='Selective risk (small, same-author sample)',ylim=(-.02,.4));axes[1].legend()
fig.tight_layout();fig.savefig(ROOT/'plots/route_calibration.png',dpi=140);plt.close(fig)
