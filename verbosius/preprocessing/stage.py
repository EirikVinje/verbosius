import pickle
import os

def stage_data(cleaned_x, split_x, token_x, lemma_x, token_ids_x, y, orig_labels):

    data_dicts = []
    for i in range(len(y)):

        instance = {"cleaned_text": cleaned_x[i],
                    "split_text": split_x[i],
                    "tokens": token_x[i],
                    "lemmas": lemma_x[i],
                    "token_ids": token_ids_x[i],
                    "label": y[i],
                    "orig_label": orig_labels[i]}

        data_dicts.append(instance)

    return data_dicts


def write_data(data, path):

    dir = os.listdir(path)
    n = len(dir)
        
    with open(f"{path}/batch_{n}.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":

    print("--- Staging data module ---")
