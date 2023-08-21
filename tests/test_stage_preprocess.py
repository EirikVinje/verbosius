from preprocessing.stage_preprocess import stage_preprocess

def test_stage_preprocess():
    
    data = stage_preprocess(dataset="imdb", 
                     input="/home/bigtech/data/verbosius/imdb/testing/chunker", 
                     output="/home/bigtech/data/verbosius/imdb/testing/preprocess", 
                     chunk_n=0,
                     return_data=False)


if __name__ == "__main__":
    test_stage_preprocess()