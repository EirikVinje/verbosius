from verbosius.preprocessing.stage_preprocess import stage_preprocess
from verbosius.trainingdata.stage_trainingdata import stage_trainingdata
from verbosius.xai_transformer.stage_transformer import stage_transformer


def test_custom_model():
    
    # Stage preprocess
    stage_preprocess("sst2", "data/sst2", "data/sst2_preprocessed")
    
    # Stage trainingdata
    stage_trainingdata("sst2", "data/sst2_preprocessed", 0, "data/sst2_trainingdata")
    
    # Stage transformer
    stage_transformer("sst2", "data/sst2_trainingdata", "data/sst2_transformer", "true", (0, -1))
    
    # Test transformer