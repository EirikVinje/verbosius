from chunking.stage_chunks import stage_chunks


def test_stage_chunks():

    stage_chunks(dataset="imdb",
                 chunk_size=500,
                 chunk_amount=3,
                 input="",
                 output="/home/bigtech/data/verbosius/imdb_testing/chunking",
                 chunkdist_n=283764)

if __name__ == "__main__":
    test_stage_chunks()