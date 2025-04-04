from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pickle
import torch
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
import numpy as np
import pickle
from pathlib import Path

class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    @abstractmethod
    def process(self, data):
        """Process the input data and return the transformed output."""
        pass

class AllenStimuliFetchStep(PipelineStep):
    """
    Fetches data from the Allen Brain Observatory.
    The session IDs are hard-coded since the stimuli are always the same.
    """
    # Hard-coded sessions
    SESSION_A = 501704220
    SESSION_B = 501559087
    SESSION_C = 501474098

    def __init__(self, boc):
        """
        :param boc: BrainObservatoryCache instance (via the AllenAPI singleton).
        """
        self.boc = boc

    def process(self, data):
        """
        Expects data to be either None or have (container_id, session, stimulus).
        We fetch a dictionary of raw stimuli arrays, store them in data['raw_data_dct'].
        """
        if isinstance(data, tuple):
            container_id, session, stimulus = data
            data = {'container_id': container_id, 'session': session, 'stimulus': stimulus}
        elif data is None:
            data = {}

        raw_data_dct = {}

        movie_one_dataset = self.boc.get_ophys_experiment_data(self.SESSION_A)
        raw_data_dct['natural_movie_one'] = movie_one_dataset.get_stimulus_template('natural_movie_one')

        movie_two_dataset = self.boc.get_ophys_experiment_data(self.SESSION_C)
        raw_data_dct['natural_movie_two'] = movie_two_dataset.get_stimulus_template('natural_movie_two')

        movie_three_dataset = self.boc.get_ophys_experiment_data(self.SESSION_A)
        raw_data_dct['natural_movie_three'] = movie_three_dataset.get_stimulus_template('natural_movie_three')

        natural_images_dataset = self.boc.get_ophys_experiment_data(self.SESSION_B)
        raw_data_dct['natural_scenes'] = natural_images_dataset.get_stimulus_template('natural_scenes')

        data['raw_data_dct'] = raw_data_dct
        return data



class ImageToEmbeddingStep(PipelineStep):
    """
    Converts images into embeddings using a specified Hugging Face model + processor,
    checking if a cached version already exists, and saves to disk if not.
    """

    def __init__(self, processor, model, embedding_cache_dir: str):
        """
        :param processor: A Hugging Face image processor (ViTFeatureExtractor, etc.)
        :param model: A Hugging Face ViT or similar model that outputs pooler_output
        :param embedding_cache_dir: Directory to load/save embeddings
        """
        self.processor = processor
        self.model = model
        self.embedding_cache_dir = Path(embedding_cache_dir)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        # Model prefix for filenames
        self.model_prefix = (
            model.name_or_path.replace('/', '_')
            if hasattr(model, 'name_or_path') else "unnamed_model"
        )
        self.embeddings_file = self.embedding_cache_dir / f"{self.model_prefix}_embeddings.pkl"
        print('BOOM',self.embeddings_file)

    def process(self, data):
        """
        data['raw_data_dct'] => dict with {stim_name: np.array_of_images, ...}
        We compute or load embeddings, then store the file path in data['embedding_file'].
        """
        raw_data_dct = data['raw_data_dct']

        if self.embeddings_file.exists():
            print(f"Found existing embeddings for model {self.model_prefix}. Using file:\n {self.embeddings_file}")
            data['embedding_file'] = str(self.embeddings_file)
            return data

        print(f"No cache found for model {self.model_prefix}. Computing now...")
        embeddings_dict = {}
        for stim_name, frames_array in raw_data_dct.items():
            embeddings = self._process_stims(frames_array)
            embeddings_dict[stim_name] = embeddings

        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"Saved embeddings to {self.embeddings_file}")

        data['embedding_file'] = str(self.embeddings_file)
        return data

    def _process_stims(self, frames_array):
        """
        Convert each frame in frames_array to embeddings (n_frames, embedding_dim).
        """
        n_frames = len(frames_array)
        frames_3ch = np.repeat(frames_array[:, None, :, :], 3, axis=1)

        with torch.no_grad():
            inputs = self.processor(images=frames_3ch[0], return_tensors="pt")
            outputs = self.model(**inputs)
            embedding_dim = outputs.pooler_output.squeeze().shape[-1]

        all_embeddings = np.empty((n_frames, embedding_dim), dtype=np.float32)
        for i in range(n_frames):
            inputs = self.processor(images=frames_3ch[i], return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_vector = outputs.pooler_output.squeeze().numpy()
            all_embeddings[i, :] = cls_vector

        return all_embeddings

    
class StimulusGroupKFoldSplitterStep(PipelineStep):
    def __init__(self, boc, eid_dict, stimulus_session_dict, n_splits=10):
        """
        :param boc: Allen BrainObservatoryCache
        :param eid_dict: container_id -> { session: eid }
        :param stimulus_session_dict: e.g. {'three_session_A': [...], ...}
        :param n_splits: how many CV folds
        """
        self.boc = boc
        self.eid_dict = eid_dict
        self.stimulus_session_dict = stimulus_session_dict
        self.n_splits = n_splits

    def process(self, data):
        """
        data requires 'container_id', 'session', 'stimulus'.
        Creates data['folds'] => list of (X_train, frames_train, X_test, frames_test).
        """
        container_id = data['container_id']
        session = data['session']
        stimulus = data['stimulus']
        
        valid_stims = self.stimulus_session_dict.get(session, [])
        if stimulus not in valid_stims:
            raise ValueError(f"Stimulus '{stimulus}' not valid for session '{session}'. "
                             f"Valid: {valid_stims}")

        session_eid = self.eid_dict[container_id][session]

        dataset = self.boc.get_ophys_experiment_data(session_eid)
        
        #dff_traces = dataset.get_dff_traces()[1]  # shape (n_neurons, n_timepoints)
        dff_traces= self.boc.get_ophys_experiment_events(ophys_experiment_id=session_eid)
        #dff_traces = dataset

        stim_table = dataset.get_stimulus_table(stimulus)
        print(stim_table)


        X_list, frame_list, groups = [], [], []

        for _, row_ in stim_table.iterrows():
            if row_['frame']!=-1:
                start_t, end_t = row_['start'], row_['end']
                frame_idx = row_['frame']
                time_indices = range(start_t, end_t)

                if len(time_indices) == 0:
                    trial_vector = np.zeros(dff_traces.shape[0])
                else:
                    relevant_traces = dff_traces[:, time_indices]
                    #trial_vector = np.max(relevant_traces, axis=1)
                    threshold = 0.0  # or pick something domain-appropriate
                    trial_vector = np.max(relevant_traces, axis=1)

                    # Convert to binary: 1 if above threshold, else 0
                    trial_vector = (trial_vector > threshold).astype(float)
                
                X_list.append(trial_vector)
                frame_list.append(frame_idx)
                groups.append(frame_idx)
            else:
                pass

        X = np.vstack(X_list)
        print(X.shape)
        frames = np.array(frame_list)
        groups = np.array(groups)

        folds = []
        gkf = GroupKFold(n_splits=self.n_splits)
        for train_idx, test_idx in gkf.split(X, groups=groups):
            X_train, X_test = X[train_idx], X[test_idx]
            frames_train, frames_test = frames[train_idx], frames[test_idx]
            folds.append((X_train, frames_train, X_test, frames_test))

        data['folds'] = folds
        return data


class MergeEmbeddingsStep(PipelineStep):
    """
    Reads the embedding file from data['embedding_file'],
    merges it with each fold in data['folds'], resulting in data['merged_folds'].
    """

    def __init__(self):
        # If you prefer, you can pass an argument here, e.g. `embedding_file`, 
        # but in this design, we read it from data.
        pass

    def process(self, data):
        """
        We expect:
          data['embedding_file'] -> path to a pickle file containing a dict: {stim_name: 2D array of embeddings}
          data['folds'] -> list of (X_train, frames_train, X_test, frames_test)
          data['stimulus'] -> e.g. 'natural_movie_one'
        
        We'll create data['merged_folds'] = list of (Xn_train, Xe_train, Xn_test, Xe_test, frames_train, frames_test).
        """
        embedding_file = data['embedding_file']
        stimulus = data['stimulus']
        folds = data['folds']

        # Load embeddings
        with open(embedding_file, 'rb') as f:
            all_stim_embeddings = pickle.load(f)

        # e.g. shape (#frames_in_stim, embedding_dim)
        # Note: we assume the indexing in all_stim_embeddings[stimulus]
        # matches the 'frame_idx' from the Allen table.
        embed_array = all_stim_embeddings[stimulus]
        #print(embed_array.shape)

        merged_folds = []
        for (Xn_train, frames_train, Xn_test, frames_test) in folds:
            # Build Xe_train from embed_array
            Xe_train = np.array([embed_array[f_idx] for f_idx in frames_train], dtype=np.float32)
            Xe_test  = np.array([embed_array[f_idx] for f_idx in frames_test], dtype=np.float32)

            merged_folds.append((Xn_train, Xe_train, Xn_test, Xe_test, frames_train, frames_test))

        data['merged_folds'] = merged_folds
        return data

    
'''

class DataSplitterStep(PipelineStep):
    """Splits data into train/test or k-folds."""
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        
    def process(self, data):
        # data might be (X, y)
        # Implement cross-validation logic or hold-out split
        # Return a list of (train_data, test_data) pairs or something similar
        folds = self._split_into_folds(data)
        return folds
    
    def _split_into_folds(self, data):
        # Example: K-Fold splitting
        # ...
        return [/* (train, test), (train, test), ... */]

class RegressionFittingStep(PipelineStep):
    """Fits a regression model (e.g., Zero-Inflated Gamma, linear regression)."""
    
    def __init__(self, model):
        """
        model can be any scikit-learn like regressor or 
        a custom class implementing .fit() and .predict().
        """
        self.model = model
    
    def process(self, data):
        # data might be (train_data, test_data) or a single dataset
        # Fit the model here
        # Return the fitted model or predictions
        X_train, y_train, X_test, y_test = data  # Example unpacking
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        return (self.model, predictions, y_test)

class StatisticalAnalysisStep(PipelineStep):
    """Performs statistical tests, significance analysis, etc."""
    
    def process(self, data):
        # data might be (model, predictions, y_test)
        # Compute p-values, confidence intervals, etc.
        model, predictions, y_test = data
        
        analysis_results = self._compute_statistics(predictions, y_test)
        return analysis_results
    
    def _compute_statistics(self, preds, actual):
        # e.g., R^2, p-values, etc.
        # ...
        return {
            "r2": ...,
            "p_value": ...,
            # more metrics...
        }
'''

import pymc as pm
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BayesianHypothesisTestingStep(PipelineStep):
    """
    A pipeline step that performs Bayesian hypothesis testing across neurons,
    comparing log-likelihoods between a trained and untrained model.
    
    Expects data['logliks'] to be of shape (n_neurons, n_folds, 2),
    where the last dimension is [trained_LLs, untrained_LLs].
    
    Returns data['bayesian_results'] with the posterior summary for each neuron's delta.
    """
    
    def __init__(self, draws=2000, tune=1000, target_accept=0.9, random_seed=42):
        """
        :param draws: Number of MCMC draws
        :param tune: Number of tuning (warm-up) steps
        :param target_accept: Target acceptance rate for sampling
        :param random_seed: Random seed for reproducibility
        """
        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept
        self.random_seed = random_seed

    def process(self, data):
        # 1) Extract log-likelihoods: shape (n_neurons, n_folds, 2)
        logliks = data.get('logliks', None)
        if logliks is None:
            raise ValueError("data['logliks'] is missing. "
                             "Expected shape (n_neurons, n_folds, 2).")
        
        n_neurons, n_folds, _ = logliks.shape
        
        # 2) Compute the difference for each neuron and fold: d_ij = L_trained - L_untrained
        differences = logliks[..., 0] - logliks[..., 1]  # shape (n_neurons, n_folds)
        
        # We'll flatten these so we can use indexes in a hierarchical model.
        # i.e. "neuron_idx" says which neuron each row belongs to.
        # This is a standard approach in PyMC for partial pooling.
        neuron_idx = np.repeat(np.arange(n_neurons), n_folds)
        diff_flat = differences.flatten()
        
        # 3) Build and sample from the Bayesian model
        with pm.Model() as model:
            # --- Hyperpriors for the per-neuron deltas ---
            mu_delta = pm.Normal("mu_delta", mu=0.0, sigma=1.0)
            sigma_delta = pm.HalfCauchy("sigma_delta", beta=2.5)
            
            # Each neuron i has a random effect delta_i
            delta_i = pm.Normal("delta_i",
                                mu=mu_delta,
                                sigma=sigma_delta,
                                shape=n_neurons)
            
            # Common measurement noise across folds
            #   (If you have reason to believe each neuron or fold has its own variance,
            #    you can extend this model to handle that.)
            sigma_lik = pm.HalfCauchy("sigma_lik", beta=2.5)
            
            # Observed differences
            pm.Normal("obs",
                      mu=delta_i[neuron_idx],
                      sigma=sigma_lik,
                      observed=diff_flat)
            
            # 4) Sample from the posterior
            trace = pm.sample(draws=self.draws,
                              tune=self.tune,
                              target_accept=self.target_accept,
                              random_seed=self.random_seed,
                              cores=1)  # or cores > 1 if available
        
        # 5) Summarize the posterior for each neuron's delta_i
        # This returns a DataFrame with mean, sd, hdi, etc.
        posterior_summary = pm.summary(trace, var_names=["delta_i"], hdi_prob=0.95)
        
        # 'posterior_summary' has an index like "delta_i__0", "delta_i__1", ...
        # We'll re-map that to actual neuron indices.
        # Alternatively, we can just keep them as is and rely on ordering.
        # Let's reorganize so each row corresponds to the same neuron index.

        # posterior_summary is a DataFrame with row index "delta_i__X"
        # We'll parse out X:
        new_index = [int(rowname.split('__')[-1]) for rowname in posterior_summary.index]
        posterior_summary.index = pd.Index(new_index, name='neuron_index')
        posterior_summary = posterior_summary.sort_index()

        # Optionally, compute posterior probability of "delta_i > 0" for each neuron
        # We'll do that by checking how many samples exceed 0 in the chain:
        delta_samples = trace.posterior["delta_i"].stack(draws=("chain", "draw"))
        # delta_samples will be shape (n_neurons, total_draws)
        prob_greater_than_zero = (delta_samples > 0).mean(dim="draws").values
        
        # Insert that as a new column
        posterior_summary["P(delta>0)"] = prob_greater_than_zero
        
        # Store the results
        data['bayesian_results'] = posterior_summary
        
        return data


import pymc as pm
import numpy as np
import pandas as pd

class BayesianHypothesisTestingAveragedStep(PipelineStep):
    """
    A pipeline step that performs Bayesian hypothesis testing across neurons
    using averaged log-likelihood differences per neuron while still 
    accounting for within-neuron variability.
    
    Expects:
      - data['avg_logliks']: shape (n_neurons,)
      - data['var_logliks']: shape (n_neurons,) (per-neuron variance)
      - data['num_trials']: shape (n_neurons,) (how many trials per neuron)
      
    Returns:
      - data['bayesian_results']: posterior summaries of delta_i.
    """
    
    def __init__(self, draws=2000, tune=1000, target_accept=0.9, random_seed=42):
        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept
        self.random_seed = random_seed

    def process(self, data):
        avg_logliks = data.get('avg_logliks', None)
        var_logliks = data.get('var_logliks', None)
        num_trials = data.get('num_trials', None)

        if avg_logliks is None or var_logliks is None or num_trials is None:
            raise ValueError("Missing required data: 'avg_logliks', 'var_logliks', 'num_trials'")

        n_neurons = len(avg_logliks)

        # Compute adjusted per-neuron variance for the mean
        adjusted_variance = var_logliks / num_trials  # Prevents overconfidence
        adjusted_std = np.sqrt(adjusted_variance)

        with pm.Model() as model:
            # Global mean and variance
            mu_delta = pm.Normal("mu_delta", mu=0.0, sigma=1.0)
            sigma_delta = pm.HalfCauchy("sigma_delta", beta=2.5)
            
            # Per-neuron effect
            delta_i = pm.Normal("delta_i", mu=mu_delta, sigma=sigma_delta, shape=n_neurons)

            # Observed averages with appropriate uncertainty
            pm.Normal("obs", mu=delta_i, sigma=adjusted_std, observed=avg_logliks)
            
            # Sample
            trace = pm.sample(draws=self.draws, tune=self.tune, target_accept=self.target_accept, random_seed=self.random_seed, cores=1)

        # Summarize results
        posterior_summary = pm.summary(trace, var_names=["delta_i"], hdi_prob=0.95)
        posterior_summary.index = pd.Index(range(n_neurons), name='neuron_index')

        # Compute posterior probability P(delta > 0)
        delta_samples = trace.posterior["delta_i"].stack(draws=("chain", "draw"))
        prob_greater_than_zero = (delta_samples > 0).mean(dim="draws").values
        posterior_summary["P(delta>0)"] = prob_greater_than_zero

        data['bayesian_results'] = posterior_summary
        return data



class AnalysisPipeline:
    """Executes a series of PipelineStep objects in order."""
    
    def __init__(self, steps):
        self.steps = steps
    
    def run(self, data):
        for step in self.steps:
            data = step.process(data)
        return data
