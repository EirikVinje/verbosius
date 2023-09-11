import chunking.chunker_functions as chunker_functions


def test_chunk():
    ds = chunker_functions.dataset("imdb")
    ds = ds(two_cat=True)
    chunked_data = chunker_functions.chunk_data_multiclass(dataset = ds,
                                                    n_chunks_per_mix=4,
                                                    chunk_size = int(25000/4),
                                                    path = "/home/tobxtra/data/verbosius/",
                                                    test_size=.2,
                                                    validation = True,
                                                    val_size=0.2,
                                                    shuffle=True,
                                                    seed=42)
    
    assert len(chunked_data[0]) == 4, "Chunked train data should have 4 chunks"
    assert len(chunked_data[2]) == 4, "Chunked test data should have 4 chunks"
    assert len(chunked_data[4]) == 4, "Chunked val data should have 4 chunks"

    assert len(chunked_data[0][0]) == (25000/4)*0.8
    assert len(chunked_data[4][0]) == (25000/4)*0.2
    



def test_chunk_supersample():
    ds = chunker_functions.dataset("imdb")
    ds = ds(two_cat=True)
    chunked_data = chunker_functions.chunk_data_multiclass_supersample(dataset = ds,
                                                    n_chunks_per_mix=4,
                                                    chunk_size = int(25000/4),
                                                    path = "/home/tobxtra/data/verbosius/",
                                                    test_size=.2,
                                                    validation = True,
                                                    val_size=0.2,
                                                    shuffle=True,
                                                    seed=42)
    
    assert len(chunked_data[0]) == 4, "Chunked train data should have 4 chunks"
    assert len(chunked_data[2]) == 4, "Chunked val data should have 4 chunks"

    assert len(chunked_data[0][0]) == (25000/4)*0.8
    assert len(chunked_data[2][0]) == (25000/4)*0.2


    chunked_data = chunker_functions.chunk_data_multiclass_supersample(dataset = ds,
                                                    n_chunks_per_mix=4,
                                                    chunk_size = 8000,
                                                    path = "/home/tobxtra/data/verbosius/",
                                                    test_size=.2,
                                                    validation = True,
                                                    val_size=0.2,
                                                    shuffle=True,
                                                    seed=42)

    assert len(chunked_data[0]) == 4, "Chunked train data should have 4 chunks"
    assert len(chunked_data[2]) == 4, "Chunked val data should have 4 chunks"

    assert len(chunked_data[0][0]) == 8000*0.8, len(chunked_data[0][0])
    assert len(chunked_data[2][0]) == 8000*0.2, len(chunked_data[2][0])
    


def test_skips_supersample_lower_chunk_size():
    ds = chunker_functions.dataset("imdb")
    ds = ds(two_cat=True)
    chunked_data = chunker_functions.chunk_data_multiclass_supersample(dataset = ds,
                                                    n_chunks_per_mix=4,
                                                    chunk_size = 3000,
                                                    path = "/home/tobxtra/data/verbosius/",
                                                    test_size=.2,
                                                    validation = True,
                                                    val_size=0.2,
                                                    shuffle=True,
                                                    seed=42)

    assert len(chunked_data[0]) == 4, "Chunked train data should have 4 chunks"
    assert len(chunked_data[2]) == 4, "Chunked val data should have 4 chunks"

    assert len(chunked_data[0][0]) == 3000*0.8, len(chunked_data[0][0])
    assert len(chunked_data[2][0]) == 3000*0.2, len(chunked_data[2][0])

if __name__ == "__main__":
    # test_chunk()
    test_chunk_supersample()
    test_skips_supersample_lower_chunk_size()
    print(f"Passed all tests in {__file__}")