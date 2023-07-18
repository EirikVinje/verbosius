import evaluate
import torch

import numpy as np

from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModel


class CustomModel(nn.Module):
    def __init__(self, num_labels, num_seq_labels, neutral_weight, loss_weight=1): 
        super(CustomModel,self).__init__() 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_weight = loss_weight
        
        #Load Model with given checkpoint and extract its body
        self.model = model = AutoModel.from_pretrained('distilroberta-base')
        #self.token_model = None
        #self.token_model = AutoModelForTokenClassification.from_pretrained('distilroberta-base')
        
        self.classifier = nn.Linear(768, num_labels) 
        self.seq_classifier = nn.Linear(768, num_seq_labels)
        self.to_evidence = nn.Sequential(nn.Linear(2, 1),
                                         nn.Sigmoid())
        self.cel = nn.CrossEntropyLoss(weight=torch.tensor([neutral_weight, 1.0, 1.0]).to(self.device))
        self.seq_cel = nn.CrossEntropyLoss()
        self.temp_evidence = None

    def forward(self, input_ids=None, attention_mask=None, labels=None, targets=None, sentiment=None):
        #print('inputids:',input_ids.size())
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        #print("1",outputs.last_hidden_state.size())
        #token_outputs = self.token_model(input_ids=input_ids, attention_mask=attention_mask)
        #print('2',token_outputs.logits.size())
        #print(outputs[1:][0])
        e = outputs.last_hidden_state
        #e_tokens = token_outputs.last_hidden_state
        
        #token_labels_pred = token_outputs.logits
        token_labels_pred = self.classifier(e)
        
        #print('outputs',e[:,1:,:])
        evidence = torch.relu(self.to_evidence(token_labels_pred[:, :, 1:3]) - 0.1)
        #self.temp_evidence = evidence.detach()
        
        
        #print('evidence:',evidence.size())
        #print('targets:',targets.size())
        
        
        w_targets = (e*evidence.expand_as(e))*targets.unsqueeze(2).expand_as(e)
        #print('w_targets',w_targets)
        
        
        #print('sentiment', sentiment.size())
        seq_label_pred = self.seq_classifier(w_targets.sum(dim=1))
        #print('seq_label_pred', seq_label_pred.size())
        
        #print('token_labels_pred', token_labels_pred.size())
        logits = (token_labels_pred, seq_label_pred)
        #print(logits)
        
        if labels is not None:
            seq_loss = self.seq_cel(seq_label_pred, sentiment)
            #print('seq_loss',seq_loss.size())
            #print('labels size',labels.size())
            #print('token_labels_pred', token_labels_pred.size())
            #print('labels', labels.size())
            
            token_loss = self.cel(token_labels_pred.view(-1, 3), labels.view(-1))
            #print('token_loss', token_loss.size())

            loss = (seq_loss)+(token_loss*self.loss_weight)
            #oss = seq_loss
            return TokenClassifierOutput(loss=loss, logits=logits)
        
        else:
            return TokenClassifierOutput(logits=logits)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels, targets, sentiment):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.targets = targets
        self.sentiment = sentiment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = self.labels[idx]
        targets = self.targets[idx]
        sentiment = self.sentiment[idx]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'targets': targets,
            'sentiment': sentiment
        }


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits[1], axis=1)
    token_predictions = np.argmax(logits[0], axis=2)
    
    nz = (predictions!=0).sum()
    token_negative = (token_predictions==1).sum()
    token_neutral = (token_predictions==0).sum()
    token_positive = (token_predictions==2).sum()

    seqeval = evaluate.load("seqeval")
    
    label_list = ["neutral", "positive", "negative"]

    true_predictions = [

        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]

        for prediction, label in zip(token_predictions, labels[0])

    ]

    true_labels = [

        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]

        for prediction, label in zip(token_predictions, labels[0])

    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    output = {
        "token_accuracy": results["overall_accuracy"],
        "sequence_accuracy": (metric.compute(predictions=predictions, references=labels[1]))['accuracy'],
        "nz": nz,
        "token_neutral": token_neutral,
        "token_negative": token_negative,
        "token_positive": token_positive
    }
    #df = pd.DataFrame(output, index=[0])
    #df.to_csv("eval_results.csv", mode="a", header=not os.path.exists("eval_results.csv"))
    return output


def tokenize_and_align_labels(data, tokenizer, device):
    
    Y = np.array([i['sentiment'] for i in data])

    examples = data

    tokenized_inputs = tokenizer([example["tokens"] for example in examples], 
                                 truncation=True, 
                                 padding=True, 
                                 return_tensors='pt',
                                 is_split_into_words=True,
                                 max_length=512
                                 )
    
    labels = []
    targets = []
    
    for i, label in enumerate([example["labels"] for example in examples]):
        
        word_ids = tokenized_inputs.word_ids(batch_index=i)  
        
        previous_word_idx = None

        label_ids = []
        target_ids = []
        
        
        for word_idx in word_ids:  

            if word_idx is None:
                target_ids.append(0)
                label_ids.append(-100)

            elif word_idx != previous_word_idx:  
                target_ids.append(1)
                label_ids.append(label[word_idx])

            else:
                target_ids.append(0)
                label_ids.append(-100)

            previous_word_idx = word_idx
        targets.append(target_ids)
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = np.array(labels,dtype=np.int8)
    
    tokenized_inputs["targets"] = np.array(targets,dtype=np.int8)

    output = {}
    output["input_ids"] = tokenized_inputs["input_ids"].to(device = device) 
    output["attention_mask"] = torch.tensor(tokenized_inputs["attention_mask"], dtype=torch.int8).to(device = device)
    output["labels"] = torch.tensor(tokenized_inputs["labels"], dtype=torch.int8).to(device = device)
    output["targets"] = torch.tensor(tokenized_inputs["targets"], dtype=torch.int8).to(device = device)
    output["sentiment"] = torch.tensor(Y, dtype=torch.int8).to(device = device)
    # print("SENTIMENT SIZE", len(output['sentiment']))
    return output