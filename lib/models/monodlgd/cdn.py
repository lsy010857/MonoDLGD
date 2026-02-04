import torch
from torch.nn import functional as F

# Reference: https://github.com/happinesslz/SEED/blob/main/pcdet/models/model_utils/cdn.py

def prepare_for_cdn(dn_args, training, num_queries, num_classes,code_size=6,ema_min_bbox=None,ema_max_bbox=None,ema_momentum_bbox=0.9,ema_min_depth=None,ema_max_depth=None,ema_momentum_depth=0.9):

    if training:
        (target_label,targets_bbox, targets_error, target_3dcenter,dn_number,label_noise_ratio, box_noise_scale, depth_noise_scale,box_uncertainty,depth_uncertainty,nknown_bid,nmap_known_indice)=dn_args

        known = [(torch.ones_like(t)).cuda() for t in target_label]
        batch_size = len(known)
        known_num = [sum(k) for k in known]

        labels = torch.cat([t for t in target_label])
        boxes = torch.cat([t for t in targets_bbox])
        error= torch.cat([t for t in targets_error])



        batch_idx = torch.cat([torch.full_like(t.long(), i) for i, t in enumerate(target_label)])
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)

        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_error = error.repeat(2 * dn_number, 1)



        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()
        known_error_expand = known_error.clone()



        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned = known_labels_expaned.to(new_label.dtype)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)


        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)



        ########################333
        box_uncertainty_=None
        if box_uncertainty is not None:
            if len(nknown_bid):
                box_uncertainty_=box_uncertainty[(nknown_bid.long(), nmap_known_indice)]
        depth_uncertainty_=None
        if depth_uncertainty is not None:
            if len(nknown_bid):
                depth_uncertainty_=depth_uncertainty[(nknown_bid.long(), nmap_known_indice)]




        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs[:, :4])

            known_bbox_[:, 0] = known_bboxs[:, 0] - known_bboxs[:, 2]  # tx
            known_bbox_[:, 1] = known_bboxs[:, 1] - known_bboxs[:, 4]  # ty
            known_bbox_[:, 2] = known_bboxs[:, 0] + known_bboxs[:, 3]  # bx
            known_bbox_[:, 3] = known_bboxs[:, 1] + known_bboxs[:, 5]  # by

            diff = torch.zeros_like(known_bboxs[:, :4])
            diff[:, 0] = known_bboxs[:, 2]
            diff[:, 1] = known_bboxs[:, 4]
            diff[:, 2] = known_bboxs[:, 3]
            diff[:, 3] = known_bboxs[:, 5]


            if  box_uncertainty_ is  None:
                rand_part = torch.rand_like(known_bboxs[:, :4])

            else:
                #############################################################################################3
                box_uncertainty_reordered = torch.cat(  # l,r,t,b -->l,t,r,b   (tx,ty,bx,by 순으로 노이즈를 주기 떄문)
                    [box_uncertainty_[:, 0:1],  # (104,1)
                     box_uncertainty_[:, 2:3],  # (104,1)
                     box_uncertainty_[:, 1:2],  # (104,1)
                     box_uncertainty_[:, 3:4]],  # (104,1)
                    dim=1
                )
                sigma_raw = torch.exp(box_uncertainty_reordered)
                sigma_inv = 1 / (sigma_raw + 1e-8)

                min_val = sigma_inv.min(dim=0, keepdim=True).values
                max_val = sigma_inv.max(dim=0, keepdim=True).values
                #add ema
                if ema_min_bbox is None:
                    ema_min_bbox = min_val
                    ema_max_bbox = max_val
                else:
                    ema_min_bbox = ema_momentum_bbox * ema_min_bbox + (1 - ema_momentum_bbox) * min_val
                    ema_max_bbox = ema_momentum_bbox * ema_max_bbox + (1 - ema_momentum_bbox) * max_val

                rand_part = ((sigma_inv - ema_min_bbox) / (ema_max_bbox - ema_min_bbox))

                #############################################################################################3

            rand_sign = torch.randint_like(known_bboxs[:, :4], low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale  #tx ,ty,bx,by update
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)

            known_bbox_expand[:, 0] =(known_bbox_[:, 0]+known_bbox_[:, 2])*0.5
            known_bbox_expand[:, 1] = (known_bbox_[:, 1]+known_bbox_[:, 3])*0.5
            known_bbox_expand[:, 2] = known_bbox_expand[:, 0] - known_bbox_[:, 0]  # l'
            known_bbox_expand[:, 3] = known_bbox_[:, 2] - known_bbox_expand[:, 0]  # r'
            known_bbox_expand[:, 4] = known_bbox_expand[:, 1] - known_bbox_[:, 1]  # t'
            known_bbox_expand[:, 5] = known_bbox_[:, 3] - known_bbox_expand[:, 1]  # b'

            #l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            #t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

        if depth_noise_scale>0:
            if depth_uncertainty_ is  None:
                rand_part=torch.rand_like(known_error)
            else:
                sigma_raw = torch.exp(depth_uncertainty_)
                sigma_inv = 1 / (sigma_raw + 1e-8)
                min_val = sigma_inv.min()
                max_val = sigma_inv.max()

                if ema_min_depth is None:
                    ema_min_depth = min_val
                    ema_max_depth = max_val
                else:
                    ema_min_depth = ema_momentum_depth * ema_min_depth + (1 - ema_momentum_depth) * min_val
                    ema_max_depth = ema_momentum_depth * ema_max_depth + (1 - ema_momentum_depth) * max_val

                rand_part = ((sigma_inv - ema_min_depth) / (ema_max_depth - ema_min_depth))

            rand_sign = torch.randint_like(known_error, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part *= rand_sign
            known_error_expand = known_error_expand + torch.mul(rand_part, known_error).cuda() * depth_noise_scale




        m = known_labels_expaned.long().to("cuda")
        input_label_embed = F.one_hot(m, num_classes=num_classes).float()
        input_bbox_embed = known_bbox_expand
        input_error_embed = known_error_expand

        padding_label = torch.zeros(pad_size, num_classes).cuda()
        padding_bbox = torch.zeros(pad_size, code_size).cuda()
        padding_error = torch.zeros(pad_size, 1).cuda()


        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
        input_query_error = padding_error.repeat(batch_size, 1, 1)



        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
            input_query_error[(known_bid.long(), map_known_indice)] = input_error_embed


        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct GTs
        attn_mask[pad_size:, :pad_size] = True
        # gt cannot see queries
        attn_mask[:pad_size, pad_size:] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i: single_pad * 2 * (i + 1), single_pad * 2 * (i + 1): pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i: single_pad * 2 * (i + 1), : single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i: single_pad * 2 * (i + 1), single_pad * 2 * (i + 1): pad_size] = True
                attn_mask[single_pad * 2 * i: single_pad * 2 * (i + 1), : single_pad * 2 * i] = True


        dn_meta = {
            "pad_size": pad_size,
            "num_dn_group": dn_number,
        }

    else:
        input_query_label = None
        input_query_error = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None



    return (input_query_label, input_query_bbox, input_query_error, attn_mask, dn_meta,known_bid,map_known_indice,ema_min_bbox,ema_max_bbox,ema_min_depth,ema_max_depth)


