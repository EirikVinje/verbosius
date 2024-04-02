from chunker import Chunker
from preprocess.preprocess import Preprocess
from weighter import Weighter
from train_eval_tokenize.trainingdata import Trainingdata
import config

def make_data():

    config.root = "/home/bigtech/data/verbosius/testing/root" 
    part_n = 1234

    # Chunker(size="big", part_n=part_n, n_chunks=10, progress_bar=True, force_write=True).run()
    # Preprocess(part_n=part_n, progress_bar=True, force_write=True).run()
    # Weighter(part_n=part_n, progress_bar=True, force_write=True).run()
    
    trainingdata = Trainingdata(part_n=part_n, progress_bar=True, force_write=True)

    print(trainingdata._get_class_balance())

    trainingdata.run()


if __name__ == "__main__":
    make_data()