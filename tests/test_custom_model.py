from preprocessing.stage_preprocess import stage_preprocess
from trainingdata.stage_trainingdata import stage_trainingdata
from xai_transformer.stage_transformer import stage_transformer


def test_custom_model():
    
    # Stage preprocess
    stage_preprocess(dataset="imdb", 
                     chunk_size=10, 
                     chunk_amount_per_mix=1, 
                     input="", 
                     output="/home/kolla/data/verbosius/imdb/preprocess", 
                     test_size=0.4,
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
    
    assert False, "Stop here"

    # Stage transformer
    stage_transformer(dataset="imdb",
                      input="/home/kolla/data/verbosius/imdb/trainingdata",
                      output="/home/kolla/data/verbosius/imdb/models",
                      save_model=False,
                      chunkdist_n=(0,-1)
                      )

    



if __name__ == "__main__":
    test_custom_model()
    
