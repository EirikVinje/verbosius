from preprocessing.stage_preprocess import stage_preprocess
from trainingdata.stage_trainingdata import stage_trainingdata
from xai_transformer.stage_transformer import stage_transformer


def test_custom_model():
    
    # Stage preprocess
    stage_preprocess(dataset="imdb", 
                     batch_size=100, 
                     batch_amount_per_mix=1, 
                     input="", 
                     output="/home/kolla/data/verbosius/imdb/preprocess", 
                     test_size=0.3, 
                     use_test_set=True, 
                     seed=42, 
                     shuffle=True,
                     batch_size_test=-1,
                     batch_amount_test=-1,)
    
    assert False, "Stop here"

    # Stage trainingdata
    stage_trainingdata(dataset="imdb",
                       input="/home/kolla/data/verbosius/imdb/preprocess",
                       batchdist_n=0,
                       output="/home/kolla/data/verbosius/imdb/trainingdata",
                       )
    
    # Stage transformer
    stage_transformer(dataset="imdb",
                      input="/home/kolla/data/verbosius/imdb/trainingdata",
                      output="/home/kolla/data/verbosius/imdb/models",
                      save_model=False,
                      batchdist_n=(0,-1)
                      )

    



if __name__ == "__main__":
    test_custom_model()
    
