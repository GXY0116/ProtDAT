import argparse
import torch
import time
import pandas as pd
import torch.nn.functional as F

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


def generate_attention_mask(seq, n_heads, cross_vocab_size, device=None):
    with torch.no_grad():   
        seq_mask = torch.ones((seq.shape[0], seq.shape[1], cross_vocab_size+seq.shape[1]), dtype=torch.bool).to(device)
        for i in range(seq_mask.shape[0]):
            tmp = torch.tril(seq_mask[i,:,cross_vocab_size:]) != 0
            seq_mask[i,:,cross_vocab_size:] = tmp
        seq_mask = seq_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
    return [None, None, seq_mask] 


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


def generate_single_seq(args, model, des, seq, length, tokenizer_seq=None, top_p=None, repetition_penalty=None, temperature=None, num_sample=1, device=None):
    if length < 1:
        raise ValueError("Length must be >= 1")
    if seq is None:
        seq = torch.tensor([[0]]).to(device) # Initialize the sequence as <cls>
    
    all_decoded_sequences = []
    
    for _ in range(num_sample):  # generate num_sample sequences
        current_seq = seq.clone()
        previous_token = None
        
        for _ in range(length):
            mask = generate_attention_mask(seq=current_seq, n_heads=args.num_head, cross_vocab_size=args.cross_vocab_size, device=device)
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
            
            if idx_next.item() == 2:
                break
        
        token_ids = current_seq[0].tolist()
        decoded_seq = tokenizer_seq.decode(token_ids)
        processed_seq = decoded_seq.replace('<cls>', '').replace(' ', '').replace('<eos>', '')
        all_decoded_sequences.append(processed_seq)
        
    return all_decoded_sequences
    
    
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
    
    seq_list = []
    begin_time = time.time()
    
    for i in range(len(data_des)):
        tokenized_seqs = tokenizer_seq(data_seq[i], padding=False, return_tensors="pt", add_special_tokens=True)
        tokenized_des = tokenizer_des(data_des[i], padding=False, return_tensors="pt", add_special_tokens=True)
        
        with torch.no_grad():
            embedding_des = model_des(input_ids=tokenized_des['input_ids'][..., :512].to(device), attention_mask=None).last_hidden_state
        
        start_time = time.time()
        
        seq = generate_single_seq(
            args=args,
            model=model,
            des=embedding_des,
            # seq=None, # only protein description
            seq=tokenized_seqs['input_ids'][...,:1].to(device), # with sequence fragment prompt
            length=500, tokenizer_seq=tokenizer_seq,
            top_p=0.85, repetition_penalty=1.2, temperature=1.0,
            num_sample=1, device=device
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        seq_list.append(seq[0])  # only when num_sample=1
        
        print(f"Iteration {i+1}:")
        print(f"Generated Sequence: {seq}")
        print(f"Origin Sequence: {data_seq[i]}")
        print(f"Total Sequences Generated: {len(seq_list)}")
        print(f"Time for Current Sequence: {elapsed_time:.2f} seconds\n")
    
    total_end_time = time.time()
    
    print(f'Total Elapsed Time: {total_end_time - begin_time:.2f} seconds')

if __name__ == '__main__':
    main()