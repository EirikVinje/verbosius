from trainingdata.stage_trainingdata import stage_trainingdata


def test_stage_trainingdata():

    # dataset : str, input : str, chunkdist_n : int, output : str, error_chunk : bool = False
    stage_trainingdata(dataset="imdb",
                       input="/home/bigtech/data/verbosius/imdb/testing/preprocess",
                       chunkdist_n=0,
                       output="/home/bigtech/data/verbosius/imdb/testing/trainingdata",
                       error_chunk=True)


if __name__ == "__main__":
    test_stage_trainingdata()