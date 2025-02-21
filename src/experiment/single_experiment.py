from src.data_processing.load_data import load_experiment_data
from src.data_processing.align_stimuli import align_stimulus_and_neural
from src.feature_extraction.vit_embeddings import extract_vit_embeddings
from src.modeling.zig_model import fit_zero_inflated_gamma
from src.evaluation.metrics import compute_metrics
from src.visualization.plot_neurons import plot_results

class SingleExperiment:
    def __init__(self, exp_id, vit_model):
        self.exp_id = exp_id
        self.vit_model = vit_model
        self.data = None
        self.embeddings = None
        self.model_results = None
        self.evaluation = None

    def load_data(self):
        self.data = load_experiment_data(self.exp_id)

    def extract_features(self):
        images = self.data['stimulus_images']
        self.embeddings = extract_vit_embeddings(images, self.vit_model)

    def fit_model(self):
        self.model_results = fit_zero_inflated_gamma(self.data['neural_activity'], self.embeddings)

    def evaluate(self):
        self.evaluation = compute_metrics(self.model_results)

    def visualize(self):
        plot_results(self.exp_id, self.model_results)

    def __call__(self):
        """ Run the full pipeline in order. """
        self.load_data()
        self.extract_features()
        self.fit_model()
        self.evaluate()
        self.visualize()
        return self.evaluation  # Return results for analysis
