"""Turn results/*.json into LaTeX table rows that writeup/main.tex \\input's."""
import json
from data import ROOT

R = ROOT / "results"
if (R / "p5_classification.json").exists():
    d = json.loads((R / "p5_classification.json").read_text())
    rows = [f"{k} & {v['test_acc']:.4f} & {v.get('test_bce', float('nan')):.4f} \\\\"
            .replace("nan", "--") for k, v in d.items() if not k.startswith("_")]
    (R / "p5_class_table.tex").write_text("\n".join(rows) + "\n")
if (R / "p5_regression.json").exists():
    d = json.loads((R / "p5_regression.json").read_text())
    rows = [f"{k} & {v['test_mae']:.4f} \\\\" for k, v in d.items() if not k.startswith("_")]
    (R / "p5_reg_table.tex").write_text("\n".join(rows) + "\n")
print("tables written")
