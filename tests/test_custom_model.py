from preprocessing.stage_preprocess import stage_preprocess



def test_custom_model():
    
    # Stage preprocess
    stage_preprocess(dataset="imdb", 
                     chunk_size=100, 
                     chunk_amount_per_mix=1, 
                     input="", 
                     output="/home/bigtech/data/verbosius/imdb/preprocess", 
                     test_size=0.3, 
                     use_test_set=True, 
                     seed=42, 
                     shuffle=True,
                     chunk_size_test=-1,
                     chunk_amount_test=-1,)
    
if __name__ == '__main__':
    test_custom_model()