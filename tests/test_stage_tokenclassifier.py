from tokenclassifier.stage_tokenclassifier import stage_tokenclassifier




def main():

    stage_tokenclassifier(dataset="imdb",
                            train_val_input="/home/bigtech/data/verbosius/imdb_testing/trainingdata",
                            test_input="/home/bigtech/data/verbosius/imdb_testing/chunker",
                            model_output="/home/bigtech/data/verbosius/imdb_testing/models",
                            chunkdist_n=0)
    

if __name__ == "__main__":
    main()