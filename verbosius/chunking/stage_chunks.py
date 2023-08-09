
import chunking.chunker_functions as chunker_functions


def stage_chunks(dataset : str, input : str, chunk_amount_per_mix : int, chunk_size : int, chunk_size_test : int, test_size : float, validation : bool, shuffle : bool, seed : int):

    ds = chunker_functions.dataset(dataset)
    ds = ds(two_cat=True)
    chunked_data = chunker_functions.chunk_data_multiclass(dataset = ds,
                                                    n_chunks_per_mix=chunk_amount_per_mix,
                                                    chunk_size = chunk_size,
                                                    path = input,
                                                    test_chunk_size=chunk_size_test,
                                                    test_size=test_size,
                                                    validation = validation,
                                                    val_chunk_size=-1,
                                                    val_size=.2,
                                                    shuffle=shuffle,
                                                    seed=seed)
    
    
    
