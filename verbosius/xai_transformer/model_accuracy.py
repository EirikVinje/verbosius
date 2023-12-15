import torch
import numpy as np
from sklearn.metrics import accuracy_score

import chunking.get_data as gd
import xai_validation.helper_functions_xaival as hf_xaival
import config as config
import xai_transformer.helper_functions as hf


def load_test(dataset):

    test = gd.dataset(dataset)(two_cat=True).load_test()
    
    new_test_x = hf_xaival.tokenize_to_model([text for text, _ in test], config.tokenizer, config.device)

    test_x = {"input_ids": [], "attention_mask": [], "targets": []}
    
    test_x = hf.extend_test(test_x, new_test_x)
    test_y = [label for _, label in test]

    test_x = hf.Test_Dataset(**test_x)

    return test_x, test_y


def accuracy(model_path, test_x, test_y):

    model = torch.load(model_path)

    model.eval()

    with torch.no_grad():

        preds = model(**test_x)

        preds = preds.logits

        preds = torch.argmax(preds, dim=2)

        preds = preds.detach().cpu().numpy()

        true = test_y.detach().cpu().numpy()

        preds = np.concatenate(preds)
        true = np.concatenate(true)

        acc = accuracy_score(true, preds)

        return acc
    

if __name__ == "__main__":

    test_x, test_y = load_test()