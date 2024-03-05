from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

import config as config
from chunking.stage_chunks import stage_chunks
from preprocess.stage_preprocess import stage_preprocess
from trainingdata.stage_trainingdata import stage_trainingdata


def test_pipeline():

    config.ERROR_MAX_DF = 0.67
    config.ERROR_MAX_FEATURES = 25
    config.ERROR_MIN_DF = 1
    config.ERROR_NUMBER_OF_CLAUSES = 100
    config.ERROR_S = 5.0
    config.ERROR_T = 100
    config.MAX_DF = 0.95
    config.MAX_FEATURES = 50
    config.MIN_DF = 1
    config.NUMBER_OF_CLAUSES = 200
    config.S = 5.0
    config.SKB_score_func = f_classif
    config.STOPWORDS = None
    config.T = 200

    config.N_JOBS = 5
    config.EARLY_STOP_ACC=1.0
    config.CV_MAX_FEATURES=5000
    config.N_GRAM_RANGE=(1, 2)
    config.LITERAL_BUDGET=6
    config.ERROR_LITERAL_BUDGET = 6

    chunkdist_n=1 
    dataset="amazon"
    chunk_size=500
    chunk_amount=5
    size="small"

    stage_chunks(dataset, chunk_size, chunk_amount, chunkdist_n, size)
    stage_preprocess(dataset, chunkdist_n)
    stage_trainingdata(dataset, chunkdist_n)
    

if __name__ == "__main__":
    test_pipeline()
    
