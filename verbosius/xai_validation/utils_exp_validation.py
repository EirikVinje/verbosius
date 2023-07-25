import numpy as np
import torch
import datasets

import config as config


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


def get_prediction_outputs(model, x, batch_size_pred):

    token_preds = []
    input_ids = []
    attention_masks = []

    with torch.no_grad():
        
        j = 0
        for i in range(batch_size_pred, len(x), batch_size_pred):
            
            if i + batch_size_pred > len(x):
                i = -1

            tx = tokenize_to_model(x[j:i], config.tokenizer, config.device)
            
            res = model(input_ids=tx["input_ids"], 
                         attention_mask=tx["attention_mask"], 
                         targets=tx["targets"])

            logits = res["logits"]
            tok_pred = torch.argmax(logits[0], dim=2)

            token_preds.append(tok_pred)
            input_ids.append(tx["input_ids"])
            attention_masks.append(tx["attention_mask"])

            j = i

    return token_preds, input_ids, attention_masks


def tokenize_to_model(data, tokenizer, device):
    

    tokenized_inputs = tokenizer(data, 
                                 truncation=True, 
                                 padding="longest", 
                                 return_tensors='pt',
                                 max_length=512,
                                 )
    
    targets = []
    
    for i in range(len(data)):
    
        word_ids = tokenized_inputs.word_ids(batch_index=0)      
        previous_word_idx = None
        target_ids = []
        
        for word_idx in word_ids:  
            
            if word_idx is None:
                target_ids.append(0)

            elif word_idx != previous_word_idx: 
                target_ids.append(1)

            else:
                target_ids.append(0)

            previous_word_idx = word_idx
        targets.append(target_ids)
        
    tokenized_inputs["targets"] = torch.tensor(targets).to(device = device)
    tokenized_inputs["input_ids"]= tokenized_inputs["input_ids"].to(device = device) 
    tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].to(device = device) 
    
    return tokenized_inputs