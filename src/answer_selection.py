import logging
import gc
import torch
import sys
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from fastprogress import master_bar, progress_bar
import random
import json
from torch import linalg as LA
from transformers import BertForQuestionAnswering
from transformers import  BertTokenizer,BertModel





FP16 = False
BATCH_SIZE = 16
SEED = 42
WARMUP_PROPORTION = 0.1
PYTORCH_PRETRAINED_BERT_CACHE = "../cache"
LOSS_SCALE = 0.
MAX_SEQ_LENGTH = 300
logger = logging.getLogger("SQ-50-bert-regressor")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(device, n_gpu, FP16))
random.seed(SEED)
np.random.seed(SEED)
dataset = "../"+sys.argv[1]+"/"
torch.manual_seed(SEED)
if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)
vec = np.memmap(dataset+'entity2vec.bin', dtype='float32', mode='r',offset=0)



#model
class AnswerSelection(BertPreTrainedModel):

    def __init__(self,config):
        super(AnswerSelection, self).__init__(config)
        
        self.bert_subject = BertModel.from_pretrained('bert-base-uncased')
        self.bert_relation = BertModel.from_pretrained('bert-base-uncased')
        self.bert_object = BertModel.from_pretrained('bert-base-uncased')
        self.bert_attention = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.loss_fct = torch.nn.MSELoss()

    def forward(self, input_ids, attention_mask,segment_ids,subject_ids,subject_mask,output_ids,output_attention_mask,targets=None):

        subject = self.bert_subject(subject_ids,subject_mask).pooler_output
        objectt = self.bert_object(output_ids,output_attention_mask).pooler_output

        attention_logits = self.bert_attention(input_ids=input_ids, token_type_ids=segment_ids,attention_mask=attention_mask)
        start_index = torch.argmax(attention_logits.start_logits,dim=-1)
        end_index = torch.argmax(attention_logits.end_logits,dim=-1)

        final_input = torch.tensor([[]]).to('cuda')
        final_mask = torch.tensor([[]]).to('cuda')

        sep =torch.tensor([102]).to('cuda')
        one = torch.tensor([1]).to('cuda')
        
        
        for i,seg,mask,s,e in zip(input_ids,segment_ids,attention_mask,start_index,end_index):
            fsplit = 0
            for index in range(0,len(i)):
                if i[index]==102:
                    fsplit = index-1
                    break
            first_part_i = i[0:fsplit+1]
            first_part_seg = seg[0:fsplit+1]
            first_part_mask = mask[0:fsplit+1]

            second_part_i = torch.cat((i[s:e+1],sep))
            second_part_seg = torch.cat((seg[s:e+1],one))
            second_part_mask = torch.cat((mask[s:e+1],one))


            concat_i = torch.cat((first_part_i,second_part_i))
            concat_seg = torch.cat((first_part_seg,second_part_seg))
            concat_mask = torch.cat((first_part_mask,second_part_mask))
            

            padding = torch.tensor([0] * (MAX_SEQ_LENGTH-len(concat_i))).to('cuda')
            concat_i = torch.cat((concat_i,padding))
            concat_seg = torch.cat((concat_seg,padding))
            concat_mask = torch.cat((concat_mask,padding))

            concat_i =concat_i.narrow(0,0,MAX_SEQ_LENGTH)
            concat_seg =concat_seg.narrow(0,0,MAX_SEQ_LENGTH)
            concat_mask =concat_mask.narrow(0,0,MAX_SEQ_LENGTH)
            
            concat_i = concat_i[None,:]
            concat_mask = concat_mask[None,:]

            if final_input.size()[1]==0:
                final_input = concat_i
                final_mask = concat_mask
            else:

                final_input = torch.cat((final_input,concat_i),dim=0)
                final_mask = torch.cat((final_mask,concat_mask),dim=0)

        relation = self.bert_relation(final_input, final_mask).pooler_output

        subject = torch.FloatTensor(subject.to('cpu')).to('cuda')
        objectt = torch.FloatTensor(objectt.to('cpu')).to('cuda')
        relation = torch.FloatTensor(relation.to('cpu')).to('cuda')

        
        Es = LA.norm(subject+relation-objectt,dim=1)
        Es = Es+ 0.0001
        score = -1 * torch.log(Es)
        
        
        if targets is not None:
            loss = self.loss_fct(score.view(-1), targets.view(-1))
            loss = torch.FloatTensor(loss.to('cpu')).to('cuda')
            return loss
        else:
            return score 



class InputExample(object):

    def __init__(self,question,subject,answer,relation,score):

        self.question = question
        self.subject = subject
        self.answer = answer
        self.relation = relation
        self.score = score


class InputFeatures(object):
    def __init__(self,input_ids,input_mask,segment_ids,subject_ids,subject_mask,output_ids,output_mask,score):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.output_ids = output_ids
        self.output_mask = output_mask
        self.segment_ids = segment_ids
        self.subject_ids = subject_ids
        self.subject_mask = subject_mask
        self.score = score



class DataProcessor:

    
    def __init__(self):

        x=2

    def get_train_examples(self):
        question = []
        subject = []
        answer = []
        relation = []
        score = []
        for line in data_train:
            line=json.loads(line)
            
            for triple in line['triples']:
                
                   question.append(line['question'][:100]+" "+triple['relations'][0])
                   subject.append(triple['subject'])
                   answer.append(triple['object'])
                   relation.append(triple['relations'][0])
                   score.append(triple['relation_object_score'])
        score=(score-np.min(score))/(np.max(score)-np.min(score))
        return self._create_examples(question,subject,answer,relation,score)

        

    def get_test_examples(self,data):

        question = []
        subject = []
        answer = []
        relation = []
        score = []
        value = data['triples']
        
        for triple in value:
            question.append(data['question'][:100]+" "+triple['relations'][0])
            answer.append(triple['object'])
            subject.append(triple['subject'])
            relation.append(triple['relations'][0])
            score.append(triple['prune_score'])
        examples = self._create_examples(question,subject,answer,relation,score)
        
        return examples


    def _create_examples(self,question,subject,answer,relation,score):
        
        examples = []
        for (i, (question,subject,answer,relation,score)) in enumerate(zip(question,subject,answer,relation,score)):
            
            examples.append(InputExample(question=question,subject=subject,answer=answer,relation=relation,score=score))
            
        return examples


n = 0
def convert_examples_to_features(examples, max_seq_length, tokenizer,bertTokenizer):
    
    features = []
    for (ex_index, example) in enumerate(examples):

        #-----------------------------question------------------------------

        encoding = bertTokenizer.encode_plus(text=example.question,text_pair=example.relation,max_length=max_seq_length,truncation=True)
        question_input_ids = encoding['input_ids']
        question_input_mask = encoding['attention_mask']
        question_input_segment = encoding['token_type_ids']

        if len(question_input_ids) > max_seq_length:
            question_input_ids = question_input_ids[:(max_seq_length)]
            question_input_mask = question_input_mask[:(max_seq_length)]
            question_input_segment = question_input_segment[:(max_seq_length)]
            
        padding = [0] * (max_seq_length - len(question_input_ids))
        question_input_ids += padding
        question_input_mask += padding
        question_input_segment += padding

        #-----------------------------subject----------------------------

        subject_ids = tokenizer.encode(example.subject,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        if len(subject_ids) > max_seq_length:
            subject_ids = subject_ids[:(max_seq_length)]
        subject_mask = [1] * len(subject_ids)
        padding = [0] * (max_seq_length - len(subject_ids))
        subject_ids += padding
        subject_mask += padding
        
            


        #-----------------------------answer------------------------------

        answer_input_ids = tokenizer.encode(example.answer,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        if len(answer_input_ids) > max_seq_length:
            answer_input_ids = answer_input_ids[:(max_seq_length)]
        answer_input_mask = [1] * len(answer_input_ids)
        padding = [0] * (max_seq_length - len(answer_input_ids))
        answer_input_ids += padding
        answer_input_mask += padding
        
        


        #--------------------------all to gether----------------------------

        input_ids=question_input_ids
        input_mask=question_input_mask
        input_segment=question_input_segment
        output_ids=answer_input_ids
        output_mask=answer_input_mask
        
        features.append(

                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=input_segment,
                              output_ids=output_ids,
                              output_mask=output_mask,
                              subject_ids=subject_ids,
                              subject_mask=subject_mask,
                              score=example.score))

    return features

class FreezableBertAdam(BertAdam):
    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    continue
                lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr 

def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())

def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b

def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))

def count_model_parameters(model):
    logger.info(
        "# of paramters: {:,d}".format(
            sum(p.numel() for p in model.parameters())))
    logger.info(
        "# of trainable paramters: {:,d}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bertTokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = AnswerSelection.from_pretrained('bert-base-uncased')
model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

def get_optimizer(num_train_optimization_steps: int, learning_rate: float):
    grouped_parameters = [
       x for x in optimizer_grouped_parameters if any([p.requires_grad for p in x["params"]])
    ]
    for group in grouped_parameters:
        group['lr'] = learning_rate
    
    optimizer = FreezableBertAdam(grouped_parameters,
                             lr=learning_rate, warmup=WARMUP_PROPORTION,
                             t_total=num_train_optimization_steps)

    return optimizer


def train(model: nn.Module, num_epochs: int, learning_rate: float):
    num_train_optimization_steps = len(train_dataloader) * num_epochs 
    optimizer = get_optimizer(num_train_optimization_steps, learning_rate)
    assert all([x["lr"] == learning_rate for x in optimizer.param_groups])
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0  
    model.train()
    mb = master_bar(range(num_epochs))
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0    
    for _ in mb:
        for step, batch in enumerate(progress_bar(train_dataloader, parent=mb)):
            try:
                batch = tuple(t.to(device) for t in batch)
                b_all_input_ids,b_all_input_masks,b_all_input_segments,b_all_subject_ids,b_all_subject_masks,b_all_output_ids,b_all_output_masks,score = batch
                loss = model(b_all_input_ids, attention_mask=b_all_input_masks,segment_ids=b_all_input_segments,subject_ids=b_all_subject_ids,subject_mask=b_all_subject_masks,output_ids=b_all_output_ids,output_attention_mask=b_all_output_masks,targets=score)
                if n_gpu > 1:
                    loss = loss.mean() 
                if FP16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if tr_loss == 0:
                    tr_loss = loss.item()
                else:
                    tr_loss = tr_loss * 0.9 + loss.item() * 0.1
                nb_tr_examples += b_all_input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                mb.child.comment = f'loss: {tr_loss:.4f} lr: {optimizer.get_lr()[0]:.2E}'
            except:
                print('h')
    logger.info("  train loss = %.4f", tr_loss) 
    return tr_loss

    

dev_dir = dataset + "pruned_dev.json"
test_dir = dataset + "pruned_test.json"
train_dir = dataset + "pruned_train.json"
output_test_file = dataset + 'final_attention1.txt'
f_out = open(output_test_file,'w')   
data_train = open(train_dir, "r")
data_test = open(test_dir, "r")
train_examples = DataProcessor().get_train_examples()
train_features = convert_examples_to_features(train_examples, MAX_SEQ_LENGTH, tokenizer,bertTokenizer)
del train_examples


gc.collect()



all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_input_segment = torch.tensor([f.segment_ids for f in train_features],dtype=torch.long)
all_subject_ids = torch.tensor([f.subject_ids for f in train_features], dtype=torch.long)
all_subject_mask = torch.tensor([f.subject_mask for f in train_features], dtype=torch.long)
all_output_ids = torch.tensor([f.output_ids for f in train_features], dtype=torch.long)
all_output_mask = torch.tensor([f.output_mask for f in train_features], dtype=torch.long)
all_score = torch.tensor([f.score for f in train_features], dtype=torch.float)
train_data = TensorDataset(all_input_ids, all_input_mask,all_input_segment,all_subject_ids,all_subject_mask,all_output_ids, all_output_mask,all_score)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

set_trainable(model, True)
set_trainable(model.bert_relation.embeddings, False)
set_trainable(model.bert_relation.encoder, False)
set_trainable(model.bert_subject.embeddings, False)
set_trainable(model.bert_subject.encoder, False)
set_trainable(model.bert_object.embeddings, False)
set_trainable(model.bert_object.encoder, False)
count_model_parameters(model)
train(model, num_epochs = 2, learning_rate = 5e-4)

model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = "../model/"+sys.argv[1]+"embedding_transe_attention.pth"
torch.save(model_to_save.state_dict(), output_model_file)
gc.collect()



set_trainable(model.bert_relation.encoder.layer[11], True)
set_trainable(model.bert_relation.encoder.layer[10], True)
set_trainable(model.bert_subject.encoder.layer[11], True)
set_trainable(model.bert_subject.encoder.layer[10], True)
set_trainable(model.bert_object.encoder.layer[11], True)
set_trainable(model.bert_object.encoder.layer[10], True)

count_model_parameters(model)
train(model, num_epochs = 2, learning_rate = 5e-5)

model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = "../model/"+sys.argv[1]+"embedding_transe_attention.pth"
torch.save(model_to_save.state_dict(), output_model_file)



set_trainable(model, True)
count_model_parameters(model)
train(model, num_epochs = 1, learning_rate = 1e-5)

model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = "../model/"+sys.argv[1]+"embedding_transe_attention.pth"
torch.save(model_to_save.state_dict(), output_model_file)
gc.collect()


placeholder = sys.argv[1]
model = AnswerSelection.from_pretrained('bert-base-uncased',cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
model.load_state_dict(torch.load("../model/"+placeholder+"embedding_transe_attention.pth"))
model.to(device)
model.eval()


BATCH_SIZE=100
counter=0
total_count=0
model.eval()
with open(dev_dir, 'r') as f:
    for line in f:
      total_count += 1
      try:
        data=json.loads(line)
        test_examples = DataProcessor().get_test_examples(data)
       
        test_features = convert_examples_to_features(test_examples, MAX_SEQ_LENGTH, tokenizer,bertTokenizer)
        if len(test_examples)!=0:
          all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
          all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
          all_input_segment = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
          all_subject_ids = torch.tensor([f.subject_ids for f in test_features], dtype=torch.long)
          all_subject_mask = torch.tensor([f.subject_mask for f in test_features], dtype=torch.long)
          all_output_ids = torch.tensor([f.output_ids for f in test_features], dtype=torch.long)
          all_output_mask = torch.tensor([f.output_mask for f in test_features], dtype=torch.long)
          all_score = torch.tensor([f.score for f in test_features], dtype=torch.float)
          test_data = TensorDataset(all_input_ids, all_input_mask,all_input_segment,all_subject_ids,all_subject_mask, all_output_ids, all_output_mask,all_score)
          test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)#not used each line is has 100 sample so itis batched naturally

          

          input_ids = all_input_ids.to(device)
          input_mask = all_input_mask.to(device)
          input_segments = all_input_segment.to(device)
          all_subject_ids = all_subject_ids.to(device)
          all_subject_mask = all_subject_mask.to(device)
          all_output_ids = all_output_ids.to(device)
          all_output_mask = all_output_mask.to(device)
          score = all_score.to(device)
          
          with torch.no_grad():
            neg = model(input_ids, attention_mask=input_mask,segment_ids=input_segments,subject_ids=all_subject_ids,subject_mask=all_subject_mask,output_ids=all_output_ids,output_attention_mask=all_output_mask)
          neg = neg.cpu().detach().numpy()
          score = score.cpu().detach().numpy()
          
          answer = False
          max_score = float('-inf')
          for triple,ne in zip(data["triples"],neg):
              if ne.item() > max_score:
                answer  = triple['answer']
                max_score = ne.item()
              triple["ranking_score"]=ne.item()
          if answer:
            counter += 1
          f_out.write(json.dumps(data)+"\n")
      except:
        print('here')    
print(counter/total_count *100)
f_out.write(str(counter/total_count *100)+"\n")
f_out.close()







