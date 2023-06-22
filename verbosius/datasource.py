

class IMDB:

    def __init__(self, two_cat : bool, batch : tuple):
        
        self.two_cat = two_cat
        self.batch = batch

    def load_data(self, path: str):

        print("Loading IMDB data")


class RottenTomatoes:

    def __init__(self, two_cat : bool, batch : tuple):
        
        self.two_cat = two_cat
        self.batch = batch

    def load_data(self, path: str):

        return


class Amazon:

    def __init__(self, two_cat : bool, batch : tuple) -> None:
        
        self.two_cat = two_cat
        self.batch = batch

    def load_data(self, path: str):

        return 
    

def dataset(dataset : str):

    if dataset == "imdb":
        return IMDB
    elif dataset == "rottentomatoes":
        return RottenTomatoes
    elif dataset == "amazon":
        return Amazon
    else:
        raise ValueError("No such dataset exists")