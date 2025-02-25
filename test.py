from src.data_loader import allen_api
from dotenv import load_dotenv
from transformers import ViTModel
from src.pipeline_steps import AnalysisPipeline, ImageToEmbeddingStep
import os
from transformers import ViTImageProcessor, ViTForImageClassification

from src.data_loader import allen_api  # your singleton
from src.pipeline_steps import AllenStimuliFetchStep, ImageToEmbeddingStep, AnalysisPipeline

load_dotenv()

'''

boc = allen_api.get_boc()

print(boc)

vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

pipeline = AnalysisPipeline([
        ImageToEmbeddingStep(vit_model)])
'''

# main.py
from dotenv import load_dotenv
load_dotenv()  # ensure environment variables are loaded before using them

def main():
    # 1. Get BOC from AllenAPI
    boc = allen_api.get_boc()

    # 2. Build HF processor + model
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

    # 3. Where to cache embeddings
    # e.g. from environment "HGMS_TRANSF_EMBEDDING_PATH"
    embedding_cache_dir = os.environ.get('TRANSF_EMBEDDING_PATH', 'embeddings_cache')

    # 4. Build the pipeline
    pipeline = AnalysisPipeline([
        AllenStimuliFetchStep(boc),                          # fetch raw stimuli
        ImageToEmbeddingStep(processor, vit_model, embedding_cache_dir),  # compute or load embeddings
        # You can add subsequent steps here (e.g., cross-validation, regression, etc.)
    ])

    # 5. Run the pipeline
    final_result = pipeline.run(None)

    # final_result is a dict: {'natural_movie_one': <ndarray>, 'natural_movie_two': <ndarray>, ...}
    print("Embeddings shapes:")
    for stim_name, embed_array in final_result.items():
        print(f"{stim_name}: {embed_array.shape}")

if __name__ == "__main__":
    main()

