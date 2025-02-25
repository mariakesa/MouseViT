from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pickle
import torch


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

    def process(self, data=None):
        """
        data is ignored. Returns a dictionary with the requested stimuli arrays.
        Structure:
            {
                'natural_movie_one': np.array,
                'natural_movie_two': np.array,
                'natural_movie_three': np.array,
                'natural_scenes': np.array
            }
        """
        raw_data_dct = {}

        # You can fetch from whichever session has the stimuli you want:
        movie_one_dataset = self.boc.get_ophys_experiment_data(self.SESSION_A)
        raw_data_dct['natural_movie_one'] = movie_one_dataset.get_stimulus_template('natural_movie_one')

        movie_two_dataset = self.boc.get_ophys_experiment_data(self.SESSION_C)
        raw_data_dct['natural_movie_two'] = movie_two_dataset.get_stimulus_template('natural_movie_two')

        movie_three_dataset = self.boc.get_ophys_experiment_data(self.SESSION_A)
        raw_data_dct['natural_movie_three'] = movie_three_dataset.get_stimulus_template('natural_movie_three')

        natural_images_dataset = self.boc.get_ophys_experiment_data(self.SESSION_B)
        raw_data_dct['natural_scenes'] = natural_images_dataset.get_stimulus_template('natural_scenes')

        return raw_data_dct


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

        # Create the cache directory if it doesn't exist
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        # Use the model name for the cache filename, replacing '/' with '_'
        # so "google/vit-base-patch16-224" becomes "google_vit-base-patch16-224.pkl"
        self.model_prefix = model.name_or_path.replace('/', '_') if hasattr(model, 'name_or_path') else "unnamed_model"
        self.embeddings_file = self.embedding_cache_dir / f"{self.model_prefix}_embeddings.pkl"

    def process(self, raw_data_dict):
        """
        :param raw_data_dict: e.g. {'natural_movie_one': np.array, 'natural_movie_two': np.array, ...}
        Returns a dict { 'natural_movie_one': embeddings, ... }
        """
        # If embeddings exist, load from pickle
        if self.embeddings_file.exists():
            print(f"Found existing embeddings for model {self.model_prefix}. Loading from cache...")
            with open(self.embeddings_file, 'rb') as f:
                embeddings_dict = pickle.load(f)
            return embeddings_dict

        # Otherwise, compute embeddings
        print(f"No cache found for model {self.model_prefix}. Computing now...")
        embeddings_dict = {}
        for stim_name, frames_array in raw_data_dict.items():
            embeddings = self._process_stims(frames_array)
            embeddings_dict[stim_name] = embeddings

        # Save to disk
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)

        print(f"Saved embeddings to {self.embeddings_file}")
        return embeddings_dict

    def _process_stims(self, frames_array):
        """
        Iterate over each frame, run the HF model, and return a 2D array: (n_frames, embedding_dim).
        """
        n_frames = len(frames_array)
        # Some HF ViT models expect 3-channel (RGB).
        # Allen stimuli might be single-channel, so replicate the channel dimension:
        frames_3ch = np.repeat(frames_array[:, None, :, :], 3, axis=1)

        # Determine embedding dimension by embedding a single frame
        with torch.no_grad():
            inputs = self.processor(images=frames_3ch[0], return_tensors="pt")
            outputs = self.model(**inputs)
            embedding_dim = outputs.pooler_output.squeeze().shape[-1]

        # Allocate memory
        all_embeddings = np.empty((n_frames, embedding_dim), dtype=np.float32)

        for i in range(n_frames):
            inputs = self.processor(images=frames_3ch[i], return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_vector = outputs.pooler_output.squeeze().numpy()
            all_embeddings[i, :] = cls_vector

        return all_embeddings
    
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
class AnalysisPipeline:
    """Executes a series of PipelineStep objects in order."""
    
    def __init__(self, steps):
        self.steps = steps
    
    def run(self, data):
        for step in self.steps:
            data = step.process(data)
        return data
