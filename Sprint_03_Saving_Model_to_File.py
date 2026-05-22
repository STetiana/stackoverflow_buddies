# Add the following code to the last cell of your model training notebook.
# Run after normal model training is done to export pipeline containing
# model to a file for loading to a data app later

import joblib
import json
import sklearn
from datetime import date

# pipeline is the fitted pipeline containing a model e.g. TabPFNRegressor - change name before use!
joblib.dump(pipeline, "pipeline_tabpfn_v3.joblib")

# save metadata (versions, model version) - change specifics before use!
meta = {
    "saved_on": "2026-05-21",
    "scikit_learn_version": sklearn.__version__,
    "tabpfn_model_version": "v3"
}
with open("pipeline_meta.json", "w") as f:
    json.dump(meta, f)