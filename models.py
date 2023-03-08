import numpy as np 
import torch 
from torch import nn 

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SASRec(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(SASRec, self).__init__()

        self.num_users = num_users 
        self.num_items = num_items 
        self.device = args.device 

        self.item_emb = nn.Embedding(self.num_items + 1, args.hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(args.max_length, args.hidden_dim)
        self.dropout = nn.Dropout(p=args.dr_rate)

        self.layernorm = nn.LayerNorm(args.hidden_dim, eps=1e-8)

        self.MHALayer = nn.ModuleList()
        self.FFNLayer = nn.ModuleList()
        for _ in range(args.num_layers):
            self.MHALayer.append(nn.MultiheadAttention(args.hidden_dim, args.num_heads, args.dr_rate))
            self.FFNLayer.append(PointWiseFeedForward(args.hidden_dim, args.dr_rate))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            try:
                nn.init.xavier_normal_(m.data)
            except:
                pass 

    def seq2feats(self, log_seqs):
        '''
        Input: item sequence
        Output: item embedding
        '''

        seqs = self.item_emb(log_seqs)
        seqs *=self.item_emb.embedding_dim ** 0.5 # sqrt(d)

        # seqs.shape: (N, S) B indicates batch size, S indicates the number of historical interaction sequence length or maximum sequence length
        # you can use a recursive function (sine or cosine function) instead of an increasing function such as range(seqs.shape[1])
        position_emb = np.tile(range(seqs.shape[1]), [seqs.shape[0], 1]) 
        '''
        np.tile([0, 1, 2], [3, 1])
        outputs:
            [[1, 2, 3], 
             [1, 2, 3], 
             [1, 2, 3]]

        '''

        seqs += self.pos_emb(torch.LongTensor(position_emb).to(self.device)) # addition position embedding and item embedding 
        seqs = self.dropout(seqs)

        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device)) # causality 


        for MHA, FFN in zip(self.MHALayer, self.FFNLayer):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.layernorm(seqs)
            mha_outs, _ = MHA(Q, seqs, seqs, attn_mask=attention_mask)

            seqs = Q + mha_outs # skip connection
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.layernorm(seqs)
            seqs = FFN(seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        feats = self.layernorm(seqs)

        return feats 

    def forward(self, user_ids, sequence, pos_seqs, neg_seqs):
        feats = self.seq2feats(sequence)

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (feats * pos_embs).sum(dim=-1)
        neg_logits = (feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits 
    
    def predict(self, user_ids, sequence, item_idx):
        feats = self.seq2feats(sequence)

        final_feat = feats[:, -1, :] # only use last QKV
        
        item_embs = self.item_emb(item_idx)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)  # F*M

        return logits 