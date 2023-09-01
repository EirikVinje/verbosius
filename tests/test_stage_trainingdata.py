from trainingdata.stage_trainingdata import stage_trainingdata


def test_stage_trainingdata():

    # stage_trainingdata(dataset : str, input : str, output : str, chunkdist_n : int, n_badtexts : int = 2000, error_chunk : bool = False)
    stage_trainingdata(dataset="imdb",
                       input="/home/bigtech/data/verbosius/imdb/testing/preprocess",
                       output="/home/bigtech/data/verbosius/imdb/testing/trainingdata",
                       chunkdist_n=0,
                       n_badtexts=25,
                       error_chunk=True,
                       )


if __name__ == "__main__":
    test_stage_trainingdata()