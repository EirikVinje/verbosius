from xai_transformer.stage_transformer import stage_transformer


def test_stage_transformer():

    # stage_transformer(dataset : str, input : str, output : str, save_model : str, chunkdist_n : int):
    stage_transformer(dataset="imdb",
                      train_val_input="/home/bigtech/data/verbosius/imdb_testing/trainingdata",
                      test_input="/home/bigtech/data/verbosius/imdb_testing/chunker",
                      model_output="/home/bigtech/data/verbosius/imdb_testing/models",
                      chunkdist_n=283764)

if __name__ == "__main__":
    test_stage_transformer()