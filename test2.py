# main.py
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from src.data_loader import allen_api
from src.Kfold_pooler import StimulusGroupKFoldSplitterStep
from src.pipeline_steps import AnalysisPipeline
from src.utils import make_container_dict

def main():
    boc = allen_api.get_boc()
    
    # Suppose you have a dictionary of container -> session -> experiment IDs
    eid_dict = make_container_dict(boc)  # from your original snippet

    # The same mapping you mentioned for sessions to stimuli
    stimulus_session_dict = {
        'three_session_A': ['natural_movie_one', 'natural_movie_three'],
        'three_session_B': ['natural_movie_one'],
        'three_session_C': ['natural_movie_one', 'natural_movie_two'],
        'three_session_C2': ['natural_movie_one', 'natural_movie_two']
    }

    # Build pipeline with just this one step
    pipeline = AnalysisPipeline([
        StimulusGroupKFoldSplitterStep(
            boc=boc,
            eid_dict=eid_dict,
            stimulus_session_dict=stimulus_session_dict,
            n_splits=3
        )
    ])
    
    container_id = 511498742
    session = 'three_session_A'
    stimulus = 'natural_movie_three'
    
    folds = pipeline.run((container_id, session, stimulus))
    for i, (X_train, X_test) in enumerate(folds, start=1):
        print(f"Fold {i}: train={X_train.shape}, test={X_test.shape}")

if __name__ == "__main__":
    main()
