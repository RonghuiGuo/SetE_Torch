import torch


def SetE_loss(args, flag, embeds_pos, embeds_neg):
    if flag == 'instanceOf':
        f_pos = torch.sum(embeds_pos[0]*embeds_pos[1], dim=-1)
        f_neg = torch.sum(embeds_neg[0]*embeds_neg[1], dim=-1)
        return torch.max(torch.zeros_like(f_pos), args.B_t - f_pos).mean() + \
            torch.max(torch.zeros_like(f_neg), f_neg - args.B_t).mean()

    elif flag == 'triple':
        g_pos = torch.sum(
            torch.cat([embeds_pos[0], embeds_pos[1]], dim=1)*embeds_pos[2], dim=-1)
        g_neg = torch.sum(
            torch.cat([embeds_neg[0], embeds_neg[1]], dim=1)*embeds_neg[2], dim=-1)
        return torch.max(torch.zeros_like(g_pos), args.B_r - g_pos).mean() + \
            torch.max(torch.zeros_like(g_neg), g_neg - args.B_r).mean()
