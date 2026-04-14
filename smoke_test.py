# smoke_test.py
from dpmm.pipelines import MSTPipeline
import fairlearn
from aif360.sklearn.preprocessing import Reweighing
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds
from aif360.sklearn.metrics import statistical_parity_difference
import folktables
import xgboost
import numpy as np
import pandas as pd
import sklearn
import scipy

print("fairlearn:", fairlearn.__version__)
print("folktables:", folktables.__version__)
print("xgboost:", xgboost.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)
print("scipy:", scipy.__version__)
print("All imports OK")

# MSTPipeline functional test
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "age": rng.integers(18, 90, 100).astype(float),
    "income": rng.choice([0, 1], 100),
    "race": pd.Categorical(rng.choice(["A", "B", "C"], 100)),
})

domain = {
    "age": {"lower": 18.0, "upper": 90.0},
    "income": {"lower": 0, "upper": 1},
    "race": {"categories": ["A", "B", "C"]},
}

model = MSTPipeline(epsilon=1.0, delta=1e-5, proc_epsilon=0.1)
model.fit(df, domain=domain)
synth = model.generate(n_records=100)
print(synth.head())
print("MSTPipeline smoke test passed")