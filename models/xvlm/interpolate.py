import torch
from scipy import interpolate
import numpy as np

def interpolate_relative_pos_embed(rel_pos_bias, dst_num_pos, param_name=''):
    # from: https://github.com/microsoft/unilm/blob/8a0a1c1f4e7326938ea7580a00d56d7f17d65612/beit/run_class_finetuning.py#L348

    # rel_pos_bias: relative_position_bias_table
    src_num_pos, num_attn_heads = rel_pos_bias.size()

    num_extra_tokens = 0
    src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
    dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
    if src_size != dst_size:
        print("Position interpolate %s from %dx%d to %dx%d" % (param_name, src_size, src_size, dst_size, dst_size))

        # extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
        # rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

        def geometric_progression(a, r, n):
            return a * (1.0 - r ** n) / (1.0 - r)

        left, right = 1.01, 1.5
        while right - left > 1e-6:
            q = (left + right) / 2.0
            gp = geometric_progression(1, q, src_size // 2)
            if gp > dst_size // 2:
                right = q
            else:
                left = q

        # if q > 1.090307:
        #     q = 1.090307

        dis = []
        cur = 1
        for i in range(src_size // 2):
            dis.append(cur)
            cur += q ** (i + 1)

        r_ids = [-_ for _ in reversed(dis)]

        x = r_ids + [0] + dis
        y = r_ids + [0] + dis

        t = dst_size // 2.0
        dx = np.arange(-t, t + 0.1, 1.0)
        dy = np.arange(-t, t + 0.1, 1.0)

        # print("Original positions = %s" % str(x))
        # print("Target positions = %s" % str(dx))

        all_rel_pos_bias = []

        for i in range(num_attn_heads):
            z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
            f = interpolate.interp2d(x, y, z, kind='cubic')
            all_rel_pos_bias.append(
                torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

        rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

    return rel_pos_bias

def interpolate_pos_embed(pos_embed_checkpoint, num_patches, num_extra_tokens=1):
    # num_patches = visual_encoder.num_patch_embed
    # num_extra_tokens = visual_encoder.num_pos_embed - visual_encoder.num_patch_embed

    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        # print('reshape position embedding from %d to %d' % (orig_size ** 2, new_size ** 2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint
