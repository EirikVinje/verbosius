import json
import os

import torch
from termcolor import colored
import numpy as np

import config as config
from xai_validation.helper_functions_xaival import tokenize_to_model

def main():

    user = os.environ.get("USER")
    dataset = "amazon"
    dist_n = 10

    with open("/home/tobxtra/projects/verbosius/verbosius/tokenclassifier/pred_docs.json", "r") as f:
        testdata = json.load(f)
    
    device = config.device
    tokenizer = config.tokenizer

    model = torch.load(f"/home/{user}/data/verbosius/{dataset}/models/{dataset}_model_dist_{dist_n}/model").to(device=device)

    testdata = tokenizer(testdata, padding="longest", truncation=True, return_tensors="pt").to(device=device)

    res = model(input_ids=testdata["input_ids"], attention_mask=testdata["attention_mask"])

    logits = res["logits"]
    token_predictions = []
    for logit in logits:
    
        token_preds = torch.argmax(logit, dim=1)
        token_predictions.append(token_preds)
    

    print()
    for token_preds, input_ids in zip(token_predictions, testdata["input_ids"]):

        input_ids = input_ids.cpu().numpy()
        token_preds = token_preds.cpu().numpy()

        is_token = np.where(input_ids > 3)[0]
        input_ids = input_ids[is_token]
        token_preds = token_preds[is_token]

        for i in range(token_preds.shape[0]):
            
            token = input_ids[i]
            pred = token_preds[i]

            token = tokenizer.convert_ids_to_tokens(int(token))
            
            if pred == 0:
                pred = colored(token[1:], "white")
            
            elif pred == 1:
                pred = colored(token[1:], "red")

            elif pred == 2:
                pred = colored(token[1:], "green")
            
            print(pred, end=" ")

        print(end="\n\n")


if __name__ == "__main__":
    main()