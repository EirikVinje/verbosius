import numpy as np
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import verbosius.preprocessing.datasource as datasource
import verbosius.preprocessing.preprocess as preprocess
import verbosius.preprocessing.stage as stage
import verbosius.trainingdata.generate_trainingdata as gen_data
import verbosius.xai_transformer.helper_functions as hf
import verbosius.config as config


def make_vocabulary(token_predictions, input_ids, attention_mask, tokenizer):
    
    vocabulary = []
    
    for token_preds, ids, masks in zip(token_predictions, input_ids, attention_mask):
        
        token_preds = token_preds[0].cpu().numpy() 
        ids = ids[0].cpu().numpy()
        masks = masks[0].cpu().numpy()
        
        masks = np.where(masks == 1)[0][1:-1]
        
        tokens = tokenizer.convert_ids_to_tokens(ids)
        tokens = np.array(tokens)
        tokens = tokens[masks]
        
        token_preds = np.array(token_preds)
        token_preds = token_preds[masks]
        
        tokens_conc = []
        indexes = []
        for j, token in enumerate(tokens):

            if token[0] == "Ġ":
                
                indexes.append(j)
                temp = [token]

                for k in range(j+1, len(tokens)):

                    if tokens[k][0] != "Ġ":
                        temp.append(tokens[k])

                    else:
                        break

                word = "".join(temp)

                tokens_conc.append(word)
    
        token_preds = token_preds[indexes]
            
        tokens = tokens_conc
        tokens = [token[1:] for token in tokens]
        tokens = np.array(tokens)
        
        not_empty_string = np.where(tokens != "")
        tokens = tokens[not_empty_string]
        token_preds = token_preds[not_empty_string]
        
        not_neutral = np.where(token_preds != 0)
        
        tokens = tokens[not_neutral]
        
        for token in tokens:
            vocabulary.append(token)
        
    vocabulary = set(vocabulary)
    vocabulary = list(vocabulary)
    
    return vocabulary


def validate():

    ds = datasource.dataset("rottentomatoes")
    ds = ds(two_cat=True)
    
    batched_data = datasource.batch_data_multiclass(dataset = ds,
                                                    n_batches_per_mix = 1,
                                                    batch_size = 500,
                                                    path = "/home/kolla/data/rottentomatoes_raw",
                                                    test = True,
                                                    use_test_set=True,
                                                    test_batch_size=2000,
                                                    test_batches_per_mix=1,
                                                    test_size=100,
                                                    shuffle=True,
                                                    seed=42)

    train_x = batched_data[0][0]
    train_y = batched_data[1][0]
    test_x = batched_data[2][0]
    test_y = batched_data[3][0]

    train_y = np.array(train_y).astype(int)
    test_y = np.array(test_y).astype(int)

    train_x_cleaned = preprocess.clean_text(train_x)
    test_x_cleaned = preprocess.clean_text(test_x)

    model = torch.load("/home/kolla/data/verbosius/imdb/models/imdb_model_0")

    print("Model loaded")

    token_preds = []
    input_ids = []
    attention_masks = []

    with torch.no_grad():

        for tx in tqdm(train_x_cleaned):
            
            tx = hf.tokenize_to_model(tx, config.tokenizer, config.device)
            
            res = model(input_ids=tx["input_ids"], 
                         attention_mask=tx["attention_mask"], 
                         targets=tx["targets"])

            logits = res["logits"]
            tok_pred = torch.argmax(logits[0], dim=2)

            token_preds.append(tok_pred)
            input_ids.append(tx["input_ids"])
            attention_masks.append(tx["attention_mask"])


    vocabulary = make_vocabulary(token_preds, input_ids, attention_masks, config.tokenizer)

    vectorizer = CountVectorizer(binary=True, vocabulary=vocabulary)

    train_x_bin = vectorizer.fit_transform(train_x_cleaned)
    test_x_bin = vectorizer.transform(test_x_cleaned)
    
    logreg = LogisticRegression(max_iter=1000, 
                                penalty='l2', 
                                random_state=42, 
                                C=0.092705530127623, 
                                tol=0.748258213506498)

    logreg.fit(train_x_bin, train_y)
    log_res =  accuracy_score(test_y, logreg.predict(test_x_bin))

    print("Logistic Regression accuracy / explanation score: ", log_res)


if __name__ == "__main__":

    validate()