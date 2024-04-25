from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

class FILIPInfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, batch_tokens1, batch_tokens2, labels, epoch=100):
        return self.info_nce(batch_tokens1, batch_tokens2, labels, epoch)
    
    def getmeanmax(self, tensor, dim):
        maxsim = torch.max(tensor, dim=dim)
        meanmax = maxsim.values.mean(dim=-1)
        idx_to_keep = torch.transpose(torch.diagonal(maxsim.indices), 0, 1)

        return meanmax, idx_to_keep

    def info_nce(self, batch_tokens1, batch_tokens2, labels, epoch):
   
        batch_tokens1, batch_tokens2 = F.normalize(batch_tokens1, dim=-1), F.normalize(batch_tokens2, dim=-1)
        labels = labels.squeeze(1)
        mask_labels = (labels[:, None] == labels[None, :]).type(torch.float)
        # mask_labels = (mask_labels/(epoch/10+1.1)).fill_diagonal_(1.)
        mask_labels = mask_labels.fill_diagonal_(1.)
        # print(labels)
        # print(mask_labels)
        # print(batch_tokens1.shape)

        batch, n_patch, _ = batch_tokens1.size()

        attn_tokens = rearrange(torch.cat((batch_tokens1, batch_tokens2)), 
                                'b p i -> (b p) i').cuda(non_blocking=True)

        token_wise_sim = torch.einsum('ki, mi -> km', attn_tokens.chunk(2))
        token_wise_sim = rearrange(token_wise_sim, '(b1 p1) (b2 p2) -> b1 b2 p1 p2', b1=batch, b2=batch, p1=n_patch)

        sim_1_to_2, idx_to_keep1_2 = self.getmeanmax(token_wise_sim, dim=-1)
        sim_2_to_1, idx_to_keep2_1 = self.getmeanmax(token_wise_sim, dim=-2)
        # print(sim_1_to_2)

        # labels = torch.arange(len(batch_tokens1)).cuda(non_blocking=True)

        # loss_1_to_2b = F.binary_cross_entropy_with_logits(sim_1_to_2 / self.temperature, 
        #                                                  mask_labels, 
        #                                                  reduction=self.reduction).cuda(non_blocking=True)
        # print(sim_2_to_1)
        # print(loss_1_to_2b)
        # loss_2_to_1b = F.binary_cross_entropy_with_logits(sim_2_to_1 / self.temperature, 
        #                                                  mask_labels, 
        #                                                  reduction=self.reduction).cuda(non_blocking=True)

        loss_1_to_2 = F.cross_entropy(sim_1_to_2 / self.temperature, 
                                      torch.arange(len(batch_tokens1)).cuda(non_blocking=True), 
                                      reduction=self.reduction).cuda(non_blocking=True)
        # loss_2_to_1 = F.cross_entropy(sim_2_to_1 / self.temperature, labels, reduction=self.reduction).cuda(non_blocking=True)

        # return (loss_1_to_2b + loss_1_to_2) / 2, idx_to_keep1_2, idx_to_keep2_1
        return loss_1_to_2, idx_to_keep1_2, idx_to_keep2_1