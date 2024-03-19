from chunking.chunker_functions import chunk_data_multiclass_supersample
from chunking.chunker_functions import chunk_data_multiclass
from chunking.get_data import dataset


def test_chunk_data_multliclass_supersample():

    
    ds = dataset("amazon")
    ds = ds(two_cat=True, size="small")
