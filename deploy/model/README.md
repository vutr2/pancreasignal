---
library_name: sklearn
pipeline_tag: tabular-classification
tags:
  - medical
  - pancreatic-cancer
  - biomarkers
  - urine
  - logistic-regression
metrics:
  - roc_auc
---

# PancreaSignal urine panel (logistic regression)

Estimates the probability of **pancreatic ductal adenocarcinoma (PDAC)** from three urine proteins (LYVE1, REG1B, TFF1), urine creatinine, age and sex. The comparison is PDAC vs everyone else in the training study: healthy controls and patients with benign hepatobiliary disease. This is the model served by the PancreaSignal API and agent.

**Live demo:** [pancreasignal Space](https://huggingface.co/spaces/trungvu0307/pancreasignal) · **Code, evaluation and agent:** [github.com/vutr2/pancreasignal](https://github.com/vutr2/pancreasignal)

> Research prototype. Not a medical device and not for clinical use. It was trained on case-control data, so its output is not a real-world risk.

## Performance on unseen hospitals

Leave-one-center-out: each of the hospitals BPTB, LIV and ESP is held out in turn, the model is trained on the others and scored on the held-out one. The 95% CI comes from 1,000 bootstrap resamples of the pooled out-of-fold predictions.

| Metric | Value | 95% CI |
|---|---|---|
| AUROC | **${auroc}** | ${auroc_lo} – ${auroc_hi} |
| Sensitivity at 90% specificity | ${sens90} | |
| Sensitivity at 95% specificity | ${sens95} | |

- **Inside each held-out hospital**, AUROC is 0.834 (BPTB), 0.800 (LIV) and 0.819 (ESP, only 6 negatives, so very noisy).
- **The model in this repository** is the same recipe refitted on all ${n_train} patients. The numbers above estimate how that recipe does on a hospital it has never seen.

Full tables, baselines and paired comparisons: [RESULTS.md](https://github.com/vutr2/pancreasignal/blob/main/RESULTS.md).

## How to use

The model is a scikit-learn `StandardScaler → LogisticRegression` pipeline. `model_light.json` stores its fitted numbers, so it loads in any scikit-learn version without unpickling anything:

```python
import json

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

with open(hf_hub_download("${repo_id}", "model_light.json")) as f:
    light = json.load(f)
names = light["feature_order"]  # LYVE1, REG1B, TFF1, age, sex

# Rebuild the fitted pipeline from its numbers
scaler = StandardScaler()
scaler.mean_ = np.array(light["scaler"]["mean"])
scaler.scale_ = np.array(light["scaler"]["scale"])
scaler.var_ = scaler.scale_**2
scaler.n_features_in_, scaler.feature_names_in_ = len(names), np.array(names, dtype=object)
logreg = LogisticRegression()
logreg.coef_ = np.array([light["logreg"]["coef"]])
logreg.intercept_ = np.array([light["logreg"]["intercept"]])
logreg.classes_ = np.array(light["logreg"]["classes"])
logreg.n_features_in_ = len(names)
model = Pipeline([("scaler", scaler), ("logreg", logreg)])

# One synthetic patient. Proteins in ng/mL, creatinine in mg/mL (the training data's units).
urine = {"LYVE1": 5.6, "REG1B": 123.0, "TFF1": 723.0}
creatinine, age, male = 0.724, 67, True

row = {m: np.log1p(v / creatinine) for m, v in urine.items()}  # creatinine-normalised, then log1p
row.update(age=age, sex=1.0 if male else 0.0)
X = pd.DataFrame([row])[names]
print(round(model.predict_proba(X)[0, 1], 3))  # ${example_probability}
```

- **Inputs:** urine LYVE1, REG1B and TFF1 in ng/mL; urine creatinine in mg/mL; age in years; sex.
- **Features:** each protein divided by creatinine, then `log1p`; plus age and sex (1 = male).
- **Output:** the probability of PDAC vs the rest of the training population.

`model_light.json` also holds the training ranges and percentiles. The project code ([api/light.py](https://github.com/vutr2/pancreasignal/blob/main/src/pancreasignal/api/light.py)) uses them to flag inputs outside the training data. It also flags values more than 10× outside the training range, which are more likely unit mistakes than real patients. Check the units: a value 1,000× too low still looks normal to the model.

`urine_v1.joblib` is the original pickled bundle: a dict with the pipeline, its config, the ranges, the percentiles and the training metadata. It needs **scikit-learn ${sklearn_version}** exactly, and you should only unpickle files you trust. `urine_v1.meta.json` gives its SHA-256. Both files contain the same numbers, and the project's tests check that the rebuilt pipeline predicts exactly what the pickled one does.

## Training

- **Data:** the urine panel of Debernardi et al. 2020: ${n_train} patients from four hospitals (BPTB, LIV, ESP, UCL). ${n_pdac} have PDAC. The other ${n_rest} are healthy controls or have benign hepatobiliary disease.
- **Model:** `StandardScaler` then `LogisticRegression` (L2, C = 1.0, lbfgs, no class weights, no extra calibration step). Logistic regression beat XGBoost on unseen hospitals by +0.041 AUROC (+0.015 to +0.066).
- **Reproducibility:** fitted with scikit-learn ${sklearn_version} and Python ${python_version} from commit [`${commit_short}`](https://github.com/vutr2/pancreasignal/commit/${commit}). The SHA-256 of the data files is in `urine_v1.meta.json`.

## Limitations

- **Case-control data, not a screening population.** ${pdac_share} of the training patients have PDAC, far more than in any real clinic. The probability is only meaningful relative to this study. Recalibrate it on local data before you read it as a risk.
- **One risk scale does not transfer between hospitals.** The PDAC share is 20% at BPTB and 70% at LIV. Models trained without BPTB predict an average risk of 0.43 there; models trained without LIV predict 0.47.
- **At LIV, the proteins add nothing over age** (AUROC difference +0.000, −0.081 to +0.080). Every benign patient at LIV has pancreatitis, the hardest condition to tell apart from PDAC.
- **Plasma CA19-9 does better on the same patients:** AUROC 0.890 vs 0.780 for PDAC vs benign disease.
- **Few early cancers:** only 16 patients have stage I disease.

## Citation

Data from the supplementary material of the article below (CC BY 4.0), via [Kaggle](https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer).

> Debernardi S, O'Brien H, Algahmdi AS, et al. *A combination of urinary biomarker panel and PancRISK score for earlier detection of pancreatic cancer: A case–control study.* PLOS Medicine 2020;17(12):e1003489. [doi:10.1371/journal.pmed.1003489](https://doi.org/10.1371/journal.pmed.1003489)
