from trainingdata.stage_trainingdata import stage_trainingdata


def test_stage_trainingdata():

    stage_trainingdata(dataset="imdb",
                       input="/home/bigtech/data/verbosius/imdb/testing/preprocess",
                       chunkdist_n=0,
                       output="/home/bigtech/data/verbosius/imdb/testing/trainingdata",
                       return_data=False,
                       get_bad_x=True)

if __name__ == "__main__":
    test_stage_trainingdata()