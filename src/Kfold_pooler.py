import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from .pipeline_steps import PipelineStep  # or wherever your base class is


class StimulusGroupKFoldSplitterStep(PipelineStep):
    """
    A pipeline step that:
      1) Looks up the experiment ID (eid) from eid_dict using (container_id, session)
      2) Fetches dff_traces from the Allen Brain Observatory
      3) Loops over each row (trial) in the stimulus table:
         - Pools neural data from [start, end) for that row,
         - Creates one row in X for each trial
      4) Uses GroupKFold, grouping by 'frame', so all trials of the same frame
         end up in the same train/test fold.

    The result is a list of (X_train, X_test) pairs, where X_* has shape
    (num_trials_in_fold, num_neurons).
    """

    def __init__(
        self,
        boc,
        eid_dict,
        stimulus_session_dict,
        n_splits=5
    ):
        """
        :param boc: BrainObservatoryCache instance (from the AllenAPI singleton)
        :param eid_dict: dict mapping container_id -> { session: eid, ... }
        :param stimulus_session_dict: dict mapping a session type to valid stimuli
               e.g. {
                 'three_session_A': ['natural_movie_one', 'natural_movie_three'],
                 ...
               }
        :param n_splits: number of cross-validation folds
        """
        self.boc = boc
        self.eid_dict = eid_dict
        self.stimulus_session_dict = stimulus_session_dict
        self.n_splits = n_splits

    def process(self, data):
        """
        :param data: a tuple (container_id, session, stimulus)
        :return: A list of (X_train, X_test) pairs for each fold
        """
        container_id, session, stimulus = data

        # 1) Validate that this stimulus is valid for the given session
        valid_stimuli = self.stimulus_session_dict.get(session, [])
        if stimulus not in valid_stimuli:
            raise ValueError(
                f"Stimulus '{stimulus}' not valid for session '{session}'. "
                f"Valid: {valid_stimuli}"
            )

        # 2) Find the experiment ID from eid_dict
        session_eid = self.eid_dict[container_id][session]

        # 3) Fetch data from Allen Brain Observatory
        dataset = self.boc.get_ophys_experiment_data(session_eid)
        # dff_traces => shape (n_neurons, n_timepoints)
        dff_traces = dataset.get_dff_traces()[1]

        # 4) Grab the stimulus table for this specific stimulus
        movie_stim_table = dataset.get_stimulus_table(stimulus)

        # 5) Build X by taking one row per trial
        #    We'll also build a 'groups' array so that all rows with the same 'frame'
        #    go to the same train/test fold.
        X_list = []
        frame_groups = []  # same length as X_list, storing the 'frame' index

        for _, row_ in movie_stim_table.iterrows():
            start_t = row_['start']
            end_t = row_['end']
            frame_idx = row_['frame']

            # Collect all timepoints for this trial
            time_indices = range(start_t, end_t)
            if len(time_indices) == 0:
                # If no timepoints found, skip or store zeros
                # We'll store zeros here just as a fallback
                trial_vector = np.zeros(dff_traces.shape[0])
            else:
                # shape => (n_neurons, #timepoints_in_this_trial)
                relevant_traces = dff_traces[:, time_indices]
                # Max pool across time
                trial_vector = np.max(relevant_traces, axis=1)

            X_list.append(trial_vector)
            frame_groups.append(frame_idx)

        # 6) Stack into final array (num_trials, num_neurons)
        X = np.vstack(X_list)  # shape (num_rows_in_table, n_neurons)
        groups = np.array(frame_groups)  # shape (num_rows_in_table,)

        # 7) GroupKFold ensures that all rows with the same frame index 
        #    end up in the same fold
        folds = []
        gkf = GroupKFold(n_splits=self.n_splits)
        for train_idx, test_idx in gkf.split(X, groups=groups):
            X_train = X[train_idx]
            X_test = X[test_idx]
            folds.append((X_train, X_test))

        return folds
