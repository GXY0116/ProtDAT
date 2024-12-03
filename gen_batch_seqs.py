import argparse
import torch
import time
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from transformers import AutoTokenizer, AutoModel
from Decoder import ProtDAT_Decoder

parser = argparse.ArgumentParser(description='GENERATE PRAMETERS')

parser.add_argument('--d-model', default=768, type=int, help='the dimension of embeddings')
parser.add_argument('--cross-vocab-size', default=50, type=int, help='cross embedding size')
parser.add_argument('--layer', default=12, type=int, help='layer of bi_cross_gpt')
parser.add_argument('--num-head', default=12, type=int, help='the number of multi head')
parser.add_argument('--model-des-path', default='model/pubmedbert', type=str, help='model to embed descriptions')
parser.add_argument('--tokenizer-des-path', default='model/pubmedbert', type=str, help='tokenizer to descriptions')
parser.add_argument('--tokenizer-seq-path', default='model/esm1b', type=str, help='tokenizer to protein sequences')
parser.add_argument('--checkpoint-path', default='model/state_dict.pth', type=str, help='checkpoint path')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout')

parser.add_argument('--gen-batch-size', default=6, type=int, help='generate batch size')
parser.add_argument('--des-test-dir',type=str, default='data/description_case', help='path to test descriptions dataset')
parser.add_argument('--pro-test-dir', type=str, default='data/sequence_case', help='path to test protein dataset')


def load_model(args, model_path, device):
    tokenizer_des = AutoTokenizer.from_pretrained(args.tokenizer_des_path)
    tokenizer_seq = AutoTokenizer.from_pretrained(args.tokenizer_seq_path)
    model = ProtDAT_Decoder(d_model=args.d_model, des_vocab_size=tokenizer_des.vocab_size, seq_vocab_size=tokenizer_seq.vocab_size, 
                         cross_vocab_size=args.cross_vocab_size, layer=args.layer, head_num=args.num_head, dropout=args.dropout, device=device).to(device)
    model = load_checkpoint(model, model_path, device)
    return tokenizer_des, tokenizer_seq, model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        if any(key.startswith('module.') for key in checkpoint['model_state_dict'].keys()):
            checkpoint['model_state_dict'] = {key.replace('module.', ''): value for key, value in checkpoint['model_state_dict'].items()}
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state_dict loaded successfully.")
    except Exception as e:
        print("Error loading model state_dict:", e)
    print("Checkpoint loaded successfully.")
    return model


def generate_attention_mask(d_mask, seq, n_heads, cross_vocab_size, device):
    with torch.no_grad():
        des_count = (d_mask != 0).sum(dim=1)
        des_mask = torch.ones((d_mask.shape[0], d_mask.shape[1], d_mask.shape[1]), dtype=torch.bool).to(device)
        cross_mask_des = torch.ones((d_mask.shape[0], cross_vocab_size, d_mask.shape[1]), dtype=torch.bool).to(device)
        for i in range(des_mask.shape[0]):
            des_mask[i,:,des_count[i]:] = False
            cross_mask_des[i,:,des_count[i]:] = False   
        des_mask = des_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
        cross_mask_des = cross_mask_des.unsqueeze(1).expand(-1, n_heads, -1, -1)
        
        seq_mask = torch.ones((seq.shape[0], seq.shape[1], cross_vocab_size+seq.shape[1]), dtype=torch.bool).to(device)
        for i in range(seq_mask.shape[0]):
            tmp = torch.tril(seq_mask[i,:,cross_vocab_size:]) != 0
            seq_mask[i,:,cross_vocab_size:] = tmp
        seq_mask = seq_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
    return [des_mask, cross_mask_des, seq_mask]


def top_p_sample(logits, top_p=0.85, previous_token=None, repetition_penalty=1.2, temperature=0.8):
    
    logits = logits / temperature
    next_tokens = []
    
    for i in range(logits.shape[0]):
        if previous_token is not None:
            if logits[i, previous_token[i]] <= 0:
                logits[i, previous_token[i]] *= repetition_penalty
            else:
                logits[i, previous_token[i]] /= repetition_penalty

        prob = F.softmax(logits[i], dim=-1)
        sorted_probs, sorted_indices = torch.sort(prob, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
            
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.gather(sorted_indices, -1, next_token)
        next_tokens.append(next_token)

    next_tokens = torch.stack(next_tokens).view(-1, 1)
    
    return next_tokens


def generate_batch_seq(args, model, des, seq, d_mask, length, tokenizer_seq=None, 
                       top_p=None, repetition_penalty=None, temperature=None, num_sample=1, device=None):
    if length < 1:
        raise ValueError("Length must be >= 1")
    if seq is None:
        seq = torch.tensor([[0]]).repeat(des.shape[0], 1).to(device)  # Initialize the sequence as <cls>

    batch_size = des.shape[0]
    all_samples = []

    for _ in range(num_sample):  # generate num_sample sequences
        current_seq = seq.clone()
        previous_token = None

        for _ in range(length):
            mask = generate_attention_mask(d_mask=d_mask, seq=current_seq, n_heads=args.num_head, cross_vocab_size=args.cross_vocab_size, device=device)
            s_emb = model(des=des, seq=current_seq, mask=mask)[2][:, -1, :]
            pre_emb = model.emb2seq(s_emb)

            idx_next = top_p_sample(
                logits=pre_emb, 
                top_p=top_p,  
                repetition_penalty=repetition_penalty, 
                temperature=temperature,
                previous_token=previous_token
            ).to(device)

            previous_token = idx_next
            current_seq = torch.cat((current_seq, idx_next), dim=1)
            
        final_decoded_sequences = []
        for i in range(batch_size):
            token_ids = current_seq[i].tolist()
            eos_position = token_ids.index(2) if 2 in token_ids else len(token_ids)
            final_decoded_sequences.append(tokenizer_seq.decode(token_ids[:eos_position]).replace('<cls>', '').replace(' ', '').replace('<eos>', ''))

        all_samples.append(final_decoded_sequences)

    return all_samples


class GenerateDataset(Dataset):
    def __init__(self, data_des, data_seq):
        self.data_des = data_des
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_des)

    def __getitem__(self, idx):
        des_item = self.data_des[idx]
        seq_item = self.data_seq[idx]

        return {
            'des_item': des_item,
            'seq_item': seq_item,
        }
  
    
def main():
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    tokenizer_des, tokenizer_seq, model = load_model(args, model_path=args.checkpoint_path, device=device)
    model_des = AutoModel.from_pretrained(args.model_des_path).to(device)
    model_des.eval()
    model.to(device)
    model.eval()
    
    data_des = pd.read_pickle(args.des_test_dir)
    data_seq = pd.read_pickle(args.pro_test_dir)
    generate_data = GenerateDataset(data_des=data_des, data_seq=data_seq)
    generate_data_loader = DataLoader(generate_data, batch_size=args.gen_batch_size, shuffle=False)
    print(len(generate_data_loader))
    
    seq_list = []
    begin_time = time.time()
    
    for batch in generate_data_loader:
        des_batch, seq_batch = batch['des_item'], batch['seq_item']
        tokenized_des = tokenizer_des(des_batch, max_length=512, padding=True, truncation=True,
                                          return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            embedding_des = model_des(input_ids=tokenized_des['input_ids'].to(device), 
                                      attention_mask=tokenized_des['attention_mask'].to(device)).last_hidden_state
        tokenized_seqs = tokenizer_seq(seq_batch, max_length=1024, padding=True, truncation=True,
                                          return_tensors="pt", add_special_tokens=True)
        
        start_time = time.time()
        
        seq = generate_batch_seq( 
            args=args,
            model=model,
            des=embedding_des,
            # seq=None, # only protein description
            seq=tokenized_seqs['input_ids'][...,:1].to(device), # with sequence fragment prompt
            d_mask=tokenized_des['attention_mask'],
            length=500, tokenizer_seq=tokenizer_seq,
            top_p=0.85, repetition_penalty=1.2, temperature=1.0,
            num_sample=1, device=device
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        seq_list.append(seq[0])  # only when num_sample=1
        
        for i in range(len(seq[0])):
            print(f"Generated Sequence: {seq[0][i]}")
            print(f"Origin Sequence: {seq_batch[i]}")
        print(f"Time for Current Sequence: {elapsed_time:.2f} seconds\n")
    
    total_end_time = time.time()
    
    print(f'Total Elapsed Time: {total_end_time - begin_time:.2f} seconds')

if __name__ == '__main__':
    main()