from preprocessing.stage_preprocess import stage_preprocess
from trainingdata.stage_trainingdata import stage_trainingdata
from xai_transformer.stage_transformer import stage_transformer





def test_custom_model():
    
    stage_preprocess(dataset="imdb", 
                     chunk_size=25000, 
                     chunk_amount_per_mix=2, 
                     input="", 
                     output="/home/bigtech/data/verbosius/imdb/preprocess", 
                     test_size=0.4,
                     seed=42,
                     shuffle=True,
                     chunk_size_test=-1,
                     validation=True)
    
    stage_trainingdata(dataset="imdb",
                       input="/home/bigtech/data/verbosius/imdb/preprocess",
                       chunkdist_n=1,
                       output="/home/bigtech/data/verbosius/imdb/trainingdata")
    
    stage_transformer(dataset="imdb",
                      input="/home/bigtech/data/verbosius/imdb/trainingdata",
                      output="/home/bigtech/data/verbosius/imdb/models",
                      save_model=True,
                      chunkdist_n=(1,2))
    

if __name__ == "__main__":
    test_custom_model()
    
