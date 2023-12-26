import torch
import torch.nn.functional as F

from third_party.gather_layer import GatherLayer
from training.criterion import nt_xent

def supcon( features, labels=None, mask=None):

    contrast_mode = 'all'
    temperature = 0.07
    base_temperature = 0.07
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]

    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)


    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]

    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)

    # for numerical stability
    import numpy as np
    tensor = anchor_dot_contrast

    # 假设tensor是一个PyTorch张量
    '''min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)*10'''

    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    # logits=normalized_tensor
    # 假设matrix是一个PyTorch张量

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    count = torch.sum(mask == 1).item()

    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )

    mask = mask * logits_mask

    count = torch.sum(mask == 1).item()

    # compute log_prob

    exp_logits = torch.exp(logits) * logits_mask

    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos

    loss = loss.view(anchor_count, batch_size).mean()

    return loss
def supcon_real(out1,out2,others,others2,labels):

    _out1 = [out1,  others]
    _out2 = [out2, others2]
    outputs1 = torch.cat(_out1, dim=0)
    outputs2 = torch.cat(_out1, dim=0)
    features = torch.cat([outputs1.unsqueeze(1), outputs2.unsqueeze(1)], dim=1)
    n=others.size(0)
    labels_fake= torch.arange(201, 201 + n).view(-1)

    _labels=[labels,labels_fake]
    out_labels=torch.cat(_labels, dim=0)
    #outputs = torch.cat(features, dim=0)
    return supcon(features,out_labels)
def supconfake(out1,out2):


    features = torch.cat([out1.unsqueeze(1), out2.unsqueeze(1)], dim=1)
    n=out1.size(0)
    labels_fake= torch.arange(201, 201 + n).view(-1)


    #outputs = torch.cat(features, dim=0)
    return supcon(features,labels_fake)

def supcon_fake(out1, out2, others, temperature, distributed=False):
    if distributed:
        out1 = torch.cat(GatherLayer.apply(out1), dim=0)
        out2 = torch.cat(GatherLayer.apply(out2), dim=0)
        others = torch.cat(GatherLayer.apply(others), dim=0)
    N = out1.size(0)

    _out = [out1, out2, others]
    outputs = torch.cat(_out, dim=0)
    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature
    sim_matrix.fill_diagonal_(-5e4)

    mask = torch.zeros_like(sim_matrix)
    mask[2*N:,2*N:] = 1
    mask.fill_diagonal_(0)

    sim_matrix = sim_matrix[2*N:]
    mask = mask[2*N:]
    mask = mask / mask.sum(1, keepdim=True)

    lsm = F.log_softmax(sim_matrix, dim=1)
    lsm = lsm * mask
    d_loss = -lsm.sum(1).mean()
    return d_loss


def loss_D_fn(P, D, options, images, gen_images,labels):
    assert images.size(0) == gen_images.size(0)
    gen_images = gen_images.detach()
    N = images.size(0)

    cat_images = torch.cat([images, images, gen_images,gen_images], dim=0)

    d_all, aux = D(P.augment_fn(cat_images), sg_linear=True, projection=True, projection2=True)


    # views = aux['projection']
    # views = F.normalize(views)
    # view1, view2, others = views[:N], views[N:2*N], views[2*N:]
    # simclr_loss = nt_xent(view1, view2, temperature=P.temp, distributed=P.distributed, normalize=False)
    simclr_loss=0
    reals = aux['projection2']
    reals = F.normalize(reals)
    real1, real2, fakes,fakes2 = reals[:N], reals[N:2*N], reals[2*N:3*N],reals[3*N:]

    sup_loss=supcon_real(real1,real2,fakes,fakes2,labels)
    d_real, d_gen = d_all[:N], d_all[2*N:3*N]
    if options['loss'] == 'nonsat':
        d_loss = F.softplus(d_gen).mean() + F.softplus(-d_real).mean()
    elif options['loss'] == 'wgan':
        d_loss = d_gen.mean() - d_real.mean()
    elif options['loss'] == 'hinge':
        d_loss = F.relu(1. + d_gen, inplace=True).mean() + F.relu(1. - d_real, inplace=True).mean()
    elif options['loss'] == 'lsgan':
        d_loss_real = ((d_real - 1.0) ** 2).mean()
        d_loss_fake = (d_gen ** 2).mean()
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
    else:
        raise NotImplementedError()

    return simclr_loss + P.lbd_a * sup_loss, {
        "penalty": d_loss,
        "d_real": d_real.mean(),
        "d_gen": d_gen.mean(),
    }


def loss_G_fn(P, D, options, images, gen_images):
    N = gen_images.size(0)
    cat_images = torch.cat([gen_images, gen_images], dim=0)

    d_gen,aux = D(P.augment_fn2(cat_images), sg_linear=True, projection=True, projection2=True)

    reals = aux['projection2']
    reals = F.normalize(reals)
    fakes, fakes2 = reals[:N], reals[N:2 * N]
    sup_loss = supconfake( fakes, fakes2)

    if options['loss'] == 'nonsat':
        g_loss = F.softplus(-d_gen).mean()
    elif options['loss'] == 'lsgan':
        g_loss = 0.5 * ((d_gen - 1.0) ** 2).mean()
    else:
        g_loss = -d_gen.mean()

    return g_loss+sup_loss/2
