from preprocessing.stage_preprocess import stage_preprocess
from trainingdata.stage_trainingdata import stage_trainingdata
from xai_transformer.stage_transformer import stage_transformer


def test_custom_model():
    
    # Stage preprocess
    stage_preprocess(dataset="imdb", 
                     chunk_size=100, 
                     chunk_amount_per_mix=1, 
                     input="", 
                     output="/home/bigtech/data/verbosius/imdb/preprocess", 
                     test_size=0.4,
                     seed=42,
                     shuffle=True,
                     chunk_size_test=-1,
                     validation=True)

    assert False, "Stop here"

    stage_preprocess(dataset="sst5", 
                     chunk_size=30, 
                     chunk_amount_per_mix=1, 
                     input="", 
                     output="/home/bigtech/data/verbosius/sst5/preprocess", 
                     test_size=0.3,
                     seed=42, 
                     shuffle=True,
                     chunk_size_test=-1,
                     validation=False)
    
    # Stage trainingdata
    stage_trainingdata(dataset="imdb",
                       input="/home/kolla/data/verbosius/imdb/preprocess",
                       chunkdist_n=0,
                       output="/home/kolla/data/verbosius/imdb/trainingdata",
                       )
    
    
    # Stage transformer
    stage_transformer(dataset="imdb",
                      input="/home/kolla/data/verbosius/imdb/trainingdata",
                      output="/home/kolla/data/verbosius/imdb/models",
                      save_model=False,
                      chunkdist_n=(0,-1)
                      )

    



if __name__ == "__main__":
    test_custom_model()
    
