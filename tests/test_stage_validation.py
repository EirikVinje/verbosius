from xai_validation.stage_validation import stage_validation


def test_stage_validation():

    # stage_validation(model_path : str, model_name : str, batch_size_pred : int):
    stage_validation(model_path="/home/bigtech/data/verbosius/imdb/models",
                     model_name="imdb_model_dist_92384576",
                     batch_size_pred=32)
    

if __name__ == "__main__":
    test_stage_validation()