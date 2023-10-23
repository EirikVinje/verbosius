import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModel
from transformers import Trainer


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
            
            #loss = token_loss

            #oss = seq_loss
            return TokenClassifierOutput(loss=loss, logits=logits)
        
        else:
            return TokenClassifierOutput(logits=logits)
        

