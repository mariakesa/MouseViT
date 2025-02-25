def main():
    # 1. Choose an embedding model (Strategy pattern, see below)
    vit_model = SomeHuggingFaceViTModel(...)
    
    # 2. Choose a regression model (could also be a strategy)
    zig_model = ZeroInflatedGammaModel(...)
    
    # 3. Build pipeline
    pipeline = AnalysisPipeline([
        ImageToEmbeddingStep(vit_model),
        DataSplitterStep(n_splits=5),
        RegressionFittingStep(zig_model),
        StatisticalAnalysisStep(),
    ])
    
    # 4. Run pipeline
    raw_images_and_labels = ...  # e.g., from Allen API
    results = pipeline.run(raw_images_and_labels)
    
    print(results)  # e.g., { 'r2': 0.85, 'p_value': 0.01, ... }