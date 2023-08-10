from preprocessing.stage_preprocess import stage_preprocess
from trainingdata.stage_trainingdata import stage_trainingdata
from xai_transformer.stage_transformer import stage_transformer
from chunking.stage_chunks import stage_chunks

def test_pipeline():
    
    stage_chunks(dataset="imdb",
                chunk_size=100,
                chunk_amount=2,
                input="",
                output="/home/kolla/data/verbosius/imdb/chunker",
                test_size=0.4,
                validation=True,
                seed=42,
                shuffle=True)

    stage_preprocess(dataset="imdb", 
                     input="/home/kolla/data/verbosius/imdb/chunker", 
                     output="/home/kolla/data/verbosius/imdb/preprocess", 
                     chunk_n=0)
    
    stage_trainingdata(dataset="imdb",
                       input="/home/kolla/data/verbosius/imdb/preprocess",
                       chunkdist_n=0,
                       output="/home/kolla/data/verbosius/imdb/trainingdata")
    
    stage_transformer(dataset="imdb",
                      input="/home/kolla/data/verbosius/imdb/trainingdata",
                      output="/home/kolla/data/verbosius/imdb/models",
                      save_model=True,
                      chunkdist_n=(0,-1))
    

if __name__ == "__main__":
    test_pipeline()
    
