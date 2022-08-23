import torch
import torch.nn.functional as F

def maskpool(mask,x):
    '''
    mask: B X C_M X  H X W (1 hot encoding of mask)
    x: B X C X H X W (normal mask)
    '''
    _,c_m,_,_ = mask.shape
    b,c,h,w = x.shape
    mask = mask.view(b,c_m,h*w)
    mask_area = mask.sum(dim=-1,keepdims=True)
    mask = mask / torch.maximum(mask_area, torch.ones_like(mask))
    x = x.permute(0,2,3,1).reshape(b,h*w,c) # B X HW X C
    x = torch.matmul(mask.view(b,c_m,h*w).to('cuda'), x)
    return x,mask

def to_binary_mask(label_map,c_m=-1,resize_to=None):
    b,h,w = label_map.shape
    if c_m==-1:
        c_m = torch.max(label_map).item()+1
    label_map_one_hot = F.one_hot(label_map,c_m).permute(0,3,1,2).float()
    if resize_to is not None:
        label_map_one_hot = F.interpolate(label_map_one_hot,resize_to, mode='bilinear',align_corners=False)
        label_map = torch.argmax(label_map_one_hot,1)
        h,w = resize_to
    label_map_one_hot = label_map_one_hot.reshape(b,c_m,h*w)
    return label_map_one_hot.view(b,c_m,h,w)

def refine_mask(src_label,target_label,mask_dim,src_dim=16):
        # B X H_mask X W_mask
        n_tgt = torch.max(target_label).item()+1
        slic_mask = to_binary_mask(target_label,n_tgt,(mask_dim,mask_dim))  # B X 100 X H_mask X W_mask
        masknet_label_map = to_binary_mask(src_label,src_dim,(mask_dim,mask_dim)) # binary B X 16 X H_mask X W_mask
        pooled,_ =maskpool(slic_mask,masknet_label_map) # B X NUM_SLIC X N_MASKS
        pooled_ids = torch.argmax(pooled,-1) # B X NUM_SLIC  X 1 => label map
        converted_idx = torch.einsum('bchw,bc->bchw',slic_mask ,pooled_ids).sum(1).long().detach() #label map in hw space
        return converted_idx