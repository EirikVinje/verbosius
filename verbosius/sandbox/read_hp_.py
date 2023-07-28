import pickle
import optuna

import numpy as np
import plotly.express as px


study = optuna.load_study(study_name="complete_pipeline_with_validation_actual", storage="sqlite:///imdb_tm_pipe.db")
for study_ in study.best_trials:
    print(study_.values)
    print(study_.params)

