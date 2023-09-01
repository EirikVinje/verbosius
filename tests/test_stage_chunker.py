from chunking.stage_chunks import stage_chunks


def test_stage_chunks():

    data = stage_chunks(dataset="imdb",
                 chunk_size=100,
                 chunk_amount=3,
                 input="",
                 output="/home/bigtech/data/verbosius/imdb/testing/chunker",
                 test_size=0.4,
                 shuffle=True)

if __name__ == "__main__":
    test_stage_chunks()