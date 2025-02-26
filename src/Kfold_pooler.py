import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from .pipeline_steps import PipelineStep  # or wherever your base class is

class StimulusGroupKFoldSplitterStep:
    """
    A pipeline step that:
      1) Looks up the experiment ID (eid) from eid_dict using (container_id, session)
      2) Fetches dff_traces from the Allen Brain Observatory
      3) Creates one row in X per row (trial) in the stimulus table (max pooling over [start, end))
      4) Uses GroupKFold to group by frame index, so all trials for the same frame
         appear together in train/test.
      5) Returns a list of folds, where each fold is a tuple:
            (X_train, frames_train, X_test, frames_test)
         That way, you know exactly which frame index each trial corresponds to.
    """

    def __init__(
        self,
        boc,
        eid_dict,
        stimulus_session_dict,
        n_splits=5
    ):
        """
        :param boc: BrainObservatoryCache instance (from your AllenAPI singleton)
        :param eid_dict: dict mapping container_id -> { session: eid, ... }
        :param stimulus_session_dict: dict mapping session -> list of valid stimuli
        :param n_splits: number of cross-validation folds
        """
        self.boc = boc
        self.eid_dict = eid_dict
        self.stimulus_session_dict = stimulus_session_dict
        self.n_splits = n_splits

    def process(self, data):
        """
        :param data: (container_id, session, stimulus)
        :return: a list of folds, each fold is (X_train, frames_train, X_test, frames_test)
        """
        container_id, session, stimulus = data
        
        # 1) Validate the stimulus for the given session
        valid_stimuli = self.stimulus_session_dict.get(session, [])
        if stimulus not in valid_stimuli:
            raise ValueError(
                f"Stimulus '{stimulus}' not valid for session '{session}'. "
                f"Valid stimuli: {valid_stimuli}"
            )

        # 2) Get the experiment ID (EID)
        session_eid = self.eid_dict[container_id][session]

        # 3) Fetch data
        dataset = self.boc.get_ophys_experiment_data(session_eid)
        dff_traces = dataset.get_dff_traces()[1]  # shape => (n_neurons, n_timepoints)

        # 4) Stimulus table for this specific stimulus
        stim_table = dataset.get_stimulus_table(stimulus)

        # 5) Build X and store frames so each row = one trial
        X_list = []
        frame_list = []  # keep track of the frame index for each row
        groups = []      # used for GroupKFold

        for _, row_ in stim_table.iterrows():
            start_t = row_['start']
            end_t = row_['end']
            frame_idx = row_['frame']
            
            # Time indices for this trial
            time_indices = range(start_t, end_t)
            
            if len(time_indices) == 0:
                # fallback to a zero vector if no timepoints found
                trial_vector = np.zeros(dff_traces.shape[0])
            else:
                relevant_traces = dff_traces[:, time_indices]
                # Max pool
                trial_vector = np.max(relevant_traces, axis=1)
            
            X_list.append(trial_vector)
            frame_list.append(frame_idx)
            groups.append(frame_idx)  # group = frame, so all trials of the same frame stay together

        X = np.vstack(X_list)                # shape => (#trials, #neurons)
        frames = np.array(frame_list)        # shape => (#trials,)
        groups = np.array(groups)            # same shape (#trials,)

        # 6) GroupKFold to keep all trials of the same frame in the same fold
        folds = []
        gkf = GroupKFold(n_splits=self.n_splits)
        for train_idx, test_idx in gkf.split(X, groups=groups):
            # Subset data
            X_train = X[train_idx]
            X_test = X[test_idx]
            
            # Subset frames
            frames_train = frames[train_idx]
            frames_test = frames[test_idx]
            
            folds.append((X_train, frames_train, X_test, frames_test))

        return folds
