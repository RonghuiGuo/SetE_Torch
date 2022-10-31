import torch
import torch.nn as nn


class SetE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.instance_embed = nn.Embedding(args.instance_num, args.emb_dim)
        self.concept_embed = nn.Embedding(args.concept_num, args.emb_dim)
        self.relation_embed = nn.Embedding(args.relation_num, args.emb_dim*2)

    def init_weights(self):
        for name, param in self.named_parameters():
            if param.data.dim() > 1:
                nn.init.xavier_uniform_(param.data)

    def forward(self, flag, data_pos, data_neg):
        if flag == 'instanceOf':
            embeds_pos = [self.instance_embed(
                data_pos[0]), self.concept_embed(data_pos[1])]
            embeds_neg = [self.instance_embed(
                data_neg[0]), self.concept_embed(data_neg[1])]
            return embeds_pos, embeds_neg
        elif flag == 'triple':
            embeds_pos = [self.instance_embed(data_pos[0]), self.instance_embed(
                data_pos[1]), self.relation_embed(data_pos[2])]
            embeds_neg = [self.instance_embed(data_neg[0]), self.instance_embed(
                data_neg[1]), self.relation_embed(data_neg[2])]
            return embeds_pos, embeds_neg
