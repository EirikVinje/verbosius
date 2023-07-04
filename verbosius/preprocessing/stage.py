import pickle
import os
import time

def stage_data(cleaned_x, split_x, token_x, lemma_x, token_ids_x, y):

    data_dicts = []
    for i in range(len(y)):

        instance = {"cleaned_text": cleaned_x[i],
                    "split_text": split_x[i],
                    "tokens": token_x[i],
                    "lemmas": lemma_x[i],
                    "token_ids": token_ids_x[i],
                    "label": y[i]}

        data_dicts.append(instance)
    
    return data_dicts


def write_data(data_dicts, path, name, timestamp : bool = False):

    root = os.path.expanduser('~')
    path = os.path.join(root, "projects", path)

    dir = os.listdir(path)
    n = len(dir)

    if timestamp:
        localtime = f"{time.localtime().tm_year}{time.localtime().tm_mon}{time.localtime().tm_mday}_{time.localtime().tm_hour}{time.localtime().tm_min}"
        with open(f"{path}/{name}_n{n}_{localtime}.pkl", "wb") as f:
            pickle.dump(data_dicts, f)

    else:
        with open(f"{path}/{name}_n{n}.pkl", "wb") as f:
            pickle.dump(data_dicts, f)




if __name__ == "__main__":

    print("--- Staging data module ---")
