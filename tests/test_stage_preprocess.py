from preprocessing.stage_preprocess import stage_preprocess

def test_stage_preprocess():
    
    stage_preprocess(dataset="imdb", 
                     input="/home/bigtech/data/verbosius/imdb_testing/chunking", 
                     output="/home/bigtech/data/verbosius/imdb_testing/preprocess", 
                     chunkdist_n=283764)


if __name__ == "__main__":
    test_stage_preprocess()