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
        self.seq_model = AutoModel.from_pretrained('distilroberta-base')
        #self.token_model = None
        #self.token_model = AutoModelForTokenClassification.from_pretrained('distilroberta-base')
        self.token_model = AutoModel.from_pretrained('distilroberta-base')

        self.classifier = nn.Linear(768, num_labels) 
        self.seq_classifier = nn.Linear(768, num_seq_labels)
        self.to_evidence = nn.Sequential(nn.Linear(2, 1),
                                         nn.Sigmoid())
        
        self.cel = nn.CrossEntropyLoss(weight=torch.tensor([neutral_weight, 1.0, 1.0]).to(self.device))
        self.seq_cel = nn.CrossEntropyLoss()
        self.temp_evidence = None

    def forward(self, input_ids=None, attention_mask=None, labels=None, targets=None, sentiment=None):
        
        outputs_token = self.token_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        outputs_seq = self.seq_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)

        token_e = outputs_token.last_hidden_state
        
        token_labels_pred = self.classifier(token_e)
        
        evidence = torch.relu(self.to_evidence(token_labels_pred[:, :, 1:3]) - 0.1)
        
        seq_e = outputs_seq.last_hidden_state

        evidence_weighted = seq_e*(evidence.expand_as(token_e)).detach()
        
        w_targets = evidence_weighted*targets.unsqueeze(2).expand_as(token_e)
        
        seq_label_pred = self.seq_classifier(w_targets.sum(dim=1))
        
        logits = (token_labels_pred, seq_label_pred)
        
        if labels is not None:
            seq_loss = self.seq_cel(seq_label_pred, sentiment)
            
            token_loss = self.cel(token_labels_pred.view(-1, 3), labels.view(-1))
        
            loss = (seq_loss)+(token_loss*self.loss_weight)
            
            return TokenClassifierOutput(loss=loss, logits=logits)
        
        else:
            return TokenClassifierOutput(logits=logits)
        

