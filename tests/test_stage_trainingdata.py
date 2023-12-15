from trainingdata.stage_trainingdata import stage_trainingdata
import numpy as np


def test_stage_trainingdata():

    stage_trainingdata(dataset="imdb",
                       input="/home/bigtech/data/verbosius/imdb_testing/preprocess",
                       output="/home/bigtech/data/verbosius/imdb_testing/trainingdata",
                       chunkdist_n=0)

if __name__ == "__main__":
    test_stage_trainingdata()