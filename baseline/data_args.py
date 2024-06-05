#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

import torch

def filt_aug(features, db_range=[-6, 6], n_band=[3, 6], min_bw=6, filter_type="linear"):
    # this is updated FilterAugment algorithm used for ICASSP 2022
    if not isinstance(filter_type, str):
        if torch.rand(1).item() < filter_type:
            filter_type = "step"
            n_band = [2, 5]
            min_bw = 4
        else:
            filter_type = "linear"
            n_band = [3, 6]
            min_bw = 6

    batch_size, _, time, n_freq_bin = features.shape  # 수정된 부분

    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()

    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1

        band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw

        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

        if filter_type == "step":
            band_factors = torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            band_factors = 10 ** (band_factors / 20)

            freq_filt = torch.ones((batch_size, 1, time, n_freq_bin)).to(features)  # 수정된 부분
            for i in range(n_freq_band):
                freq_filt[:, :, :, band_bndry_freqs[i]:band_bndry_freqs[i + 1]] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

        elif filter_type == "linear":
            band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            freq_filt = torch.ones((batch_size, 1, time, n_freq_bin)).to(features)  # 수정된 부분
            for i in range(n_freq_band):
                for j in range(batch_size):
                    # print("linspace_result size:",torch.linspace(band_factors[j, i], band_factors[j, i+1],
                    #                    band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1).unsqueeze(-1).size())
                    # print("freq_filt slice size:", freq_filt[j, :, :, band_bndry_freqs[i]:band_bndry_freqs[i+1]].size())
                    freq_filt[j, :, :, band_bndry_freqs[i]:band_bndry_freqs[i+1]] = \
                        torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                       band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(0).unsqueeze(0)

            freq_filt = 10 ** (freq_filt / 20)

        return features * freq_filt

    else:
        return features



# def filt_aug(features, db_range=[-6, 6], n_band=[3, 6], min_bw=6, filter_type="linear"):
#     # this is updated FilterAugment algorithm used for ICASSP 2022
#     if not isinstance(filter_type, str):
#         if torch.rand(1).item() < filter_type:
#             filter_type = "step"
#             n_band = [2, 5]
#             min_bw = 4
#         else:
#             filter_type = "linear"
#             n_band = [3, 6]
#             min_bw = 6

#     batch_size, n_freq_bin, _ = features.shape
#     n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()   # [low, high)
#     if n_freq_band > 1:
#         while n_freq_bin - n_freq_band * min_bw + 1 < 0:
#             min_bw -= 1
#         band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
#                                                     (n_freq_band - 1,)))[0] + \
#                            torch.arange(1, n_freq_band) * min_bw
#         band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

#         if filter_type == "step":
#             band_factors = torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
#             band_factors = 10 ** (band_factors / 20)

#             freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
#             for i in range(n_freq_band):
#                 freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

#         elif filter_type == "linear":
#             band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
#             freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
#             for i in range(n_freq_band):
#                 for j in range(batch_size):
#                     freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
#                         torch.linspace(band_factors[j, i], band_factors[j, i+1],
#                                        band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
#             freq_filt = 10 ** (freq_filt / 20)
#         return features * freq_filt

#     else:
#         return features


# def filt_aug_prototype(features, db_range=(-7.5, 6), n_bands=(2, 5)):
#     # this is FilterAugment algorithm used for DCASE 2021 Challeng Task 4
#     batch_size, n_freq_bin, _ = features.shape
#     n_freq_band = torch.randint(low=n_bands[0], high=n_bands[1], size=(1,)).item()   # [low, high)
#     if n_freq_band > 1:
#         band_bndry_freqs = torch.cat((torch.tensor([0]),
#                                       torch.sort(torch.randint(1, n_freq_bin-1, (n_freq_band - 1, )))[0],
#                                       torch.tensor([n_freq_bin])))
#         band_factors = torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
#         band_factors = 10 ** (band_factors / 20)

#         freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
#         for i in range(n_freq_band):
#             freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)
#         return features * freq_filt
#     else:
#         return features


# def frame_shift(features, label=None, net_pooling=None):
#     if label is not None:
#         batch_size, _, _ = features.shape
#         shifted_feature = []
#         shifted_label = []
#         for idx in range(batch_size):
#             shift = int(random.gauss(0, 90))
#             shifted_feature.append(torch.roll(features[idx], shift, dims=-1))
#             shift = -abs(shift) // net_pooling if shift < 0 else shift // net_pooling
#             shifted_label.append(torch.roll(label[idx], shift, dims=-1))
#         return torch.stack(shifted_feature), torch.stack(shifted_label)
#     else:
#         batch_size, _, _ = features.shape
#         shifted_feature = []
#         for idx in range(batch_size):
#             shift = int(random.gauss(0, 90))
#             shifted_feature.append(torch.roll(features[idx], shift, dims=-1))
#         return torch.stack(shifted_feature)


# def mixup(features, label=None, permutation=None, c=None, alpha=0.2, beta=0.2, mixup_label_type="soft", returnc=False):
#     with torch.no_grad():
#         batch_size = features.size(0)

#         if permutation is None:
#             permutation = torch.randperm(batch_size)

#         if c is None:
#             if mixup_label_type == "soft":
#                 c = np.random.beta(alpha, beta)
#             elif mixup_label_type == "hard":
#                 c = np.random.beta(alpha, beta) * 0.4 + 0.3   # c in [0.3, 0.7]

#         mixed_features = c * features + (1 - c) * features[permutation, :]
#         if label is not None:
#             if mixup_label_type == "soft":
#                 mixed_label = torch.clamp(c * label + (1 - c) * label[permutation, :], min=0, max=1)
#             elif mixup_label_type == "hard":
#                 mixed_label = torch.clamp(label + label[permutation, :], min=0, max=1)
#             else:
#                 raise NotImplementedError(f"mixup_label_type: {mixup_label_type} not implemented. choice in "
#                                           f"{'soft', 'hard'}")
#             if returnc:
#                 return mixed_features, mixed_label, c, permutation
#             else:
#                 return mixed_features, mixed_label
#         else:
#             return mixed_features
# 
# def feature_transformation(features, n_transform, choice, filtaug_choice, filter_db_range, filter_bands,
#                            filter_minimum_bandwidth, filter_type, freq_mask_ratio, noise_snrs):
#     if n_transform == 2:
#         feature_list = []
#         for _ in range(n_transform):
#             features_temp = features
#             if choice[0]:
#                 if filtaug_choice == "prototype":
#                     features_temp = filt_aug_prototype(features_temp, db_range=filter_db_range, n_bands=filter_bands)
#                 elif filtaug_choice == "updated":
#                     features_temp = filt_aug(features_temp, db_range=filter_db_range, n_band=filter_bands,
#                                              min_bw=filter_minimum_bandwidth, filter_type=filter_type)
#             if choice[1]:
#                 features_temp = freq_mask(features_temp, mask_ratio=freq_mask_ratio)
#             if choice[2]:
#                 features_temp = add_noise(features_temp, snrs=noise_snrs)
#             feature_list.append(features_temp)
#         return feature_list
#     elif n_transform == 1:
#         if choice[0]:
#             if filtaug_choice == "prototype":
#                 features = filt_aug_prototype(features, db_range=filter_db_range, n_bands=filter_bands)
#             elif filtaug_choice == "updated":
#                 features = filt_aug(features, db_range=filter_db_range, n_band=filter_bands,
#                                     min_bw=filter_minimum_bandwidth, filter_type=filter_type)
#         if choice[1]:
#             features = freq_mask(features, mask_ratio=freq_mask_ratio)
#         if choice[2]:
#             features = add_noise(features, snrs=noise_snrs)
#         return [features, features]
#     else:
#         return [features, features]


# def freq_mask(features, mask_ratio=16):
#     batch_size, n_freq_bin, _ = features.shape
#     max_mask = int(n_freq_bin/mask_ratio)
#     if max_mask == 1:
#         f_widths = torch.ones(batch_size)
#     else:
#         f_widths = torch.randint(low=1, high=max_mask, size=(batch_size,))   # [low, high)

#     for i in range(batch_size):
#         f_width = f_widths[i]
#         f_low = torch.randint(low=0, high=n_freq_bin-f_width, size=(1,))

#         features[i, f_low:f_low+f_width, :] = 0
#     return features


# def add_noise(features, snrs=(15, 30), dims=(1, 2)):
#     if isinstance(snrs, (list, tuple)):
#         snr = (snrs[0] - snrs[1]) * torch.rand((features.shape[0],), device=features.device).reshape(-1, 1, 1) + snrs[1]
#     else:
#         snr = snrs

#     snr = 10 ** (snr / 20)
#     sigma = torch.std(features, dim=dims, keepdim=True) / snr
#     return features + torch.randn(features.shape, device=features.device) * sigma
    
def plt_(batch_input,target,ema_batch_input,target_s):
    x= batch_input[18].cpu().squeeze().transpose(0,1)
    plt.subplot(2,2,1)
    plt.imshow(x,origin='lower')

    x= target[18].cpu().squeeze().transpose(0,1)
    plt.subplot(2,2,2)
    plt.imshow(x,origin='lower')

    x= ema_batch_input[18].cpu().squeeze().transpose(0,1)
    plt.subplot(2,2,3)
    plt.imshow(x,origin='lower')

    t= target_s[18].cpu().squeeze().transpose(0,1)
    plt.subplot(2,2,4)
    plt.imshow(t,origin='lower')

    plt.show()

def plt_mix_up(batch_input,target,ema_batch_input,target_s):
    x= batch_input[18].cpu().squeeze().transpose(0,1)
    plt.subplot(2,3,1)
    plt.imshow(x,origin='lower')

    x= target[18].cpu().squeeze().transpose(0,1)
    plt.subplot(2,3,2)
    plt.imshow(x,origin='lower')

    x= batch_input[19].cpu().squeeze().transpose(0,1)
    plt.subplot(2,3,3)
    plt.imshow(x,origin='lower')

    t= target[19].cpu().squeeze().transpose(0,1)
    plt.subplot(2,3,4)
    plt.imshow(t,origin='lower')

    x= ema_batch_input[18].cpu().squeeze().transpose(0,1)
    plt.subplot(2,3,5)
    plt.imshow(x,origin='lower')

    t= target_s[18].cpu().squeeze().transpose(0,1)
    plt.subplot(2,3,6)
    plt.imshow(t,origin='lower')

    plt.show()



def time_mask(features, labels=None, net_pooling=None, mask_ratios=(10, 50)):
    if labels is not None:
        _, n_frame,_ = labels.shape
        t_width = torch.randint(low=int(n_frame/mask_ratios[1]), high=int(n_frame/mask_ratios[0]), size=(1,))   # [low, high)
        t_low = torch.randint(low=0, high=n_frame-t_width[0], size=(1,))
        features[:, :, t_low * net_pooling:(t_low+t_width)*net_pooling,:] = 0
        labels[:, t_low:t_low+t_width,:] = 0
        return features, labels
    else:
        # _, _, n_frame = features.shape
        _, n_frame,_ = features.shape
        t_width = torch.randint(low=int(n_frame/mask_ratios[1]), high=int(n_frame/mask_ratios[0]), size=(1,))   # [low, high)
        t_low = torch.randint(low=0, high=n_frame-t_width[0], size=(1,))
        features[:, t_low:(t_low + t_width),:] = 0
        return features
    
def time_mask_ex_target(features, net_pooling=None, mask_ratios=(10, 50)):
    _, _, n_frame,_ = features.shape
    t_width = torch.randint(low=int(n_frame/mask_ratios[1]), high=int(n_frame/mask_ratios[0]), size=(1,))   # [low, high)
    t_low = torch.randint(low=0, high=n_frame-t_width[0], size=(1,))
    features[:, :, t_low * net_pooling:(t_low+t_width)*net_pooling,:] = 0
    return features


def ema_input_target(Nsample, target, batch_input = None):
    if batch_input is not None:
        ema_batch_input = torch.roll(batch_input, 4*Nsample, 2)
    else:
        ema_batch_input = None

    target_s = target.clone()
    temp = target_s[18:24:].clone()
    temp = torch.roll(temp, Nsample, 1)
    target_s[18:24:] = temp
    
    return ema_batch_input, target_s

def mix_up(batch_input, Nsample, target=None):
    ema_batch_input = batch_input.clone()
    for i in range(0,23+1):
        if i==5:
            ema_batch_input[i] = Nsample*batch_input[i] + (1-Nsample)*batch_input[0]
        elif i == 17:
            ema_batch_input[i] = Nsample*batch_input[i] + (1-Nsample)*batch_input[6]
        elif i== 23:
            ema_batch_input[i] = Nsample*batch_input[i] + (1-Nsample)*batch_input[18]
        else:
            ema_batch_input[i] = Nsample*batch_input[i] + (1-Nsample)*batch_input[i+1]
    if target is not None:
        target_s = target.clone()
        for i in range(0,23+1):
            # 사실 weak target 같은 경우에는 바꿀 필요 없지만 unlabled 바꿔야 하니까 그냥 바꿔주자
            if i==5:
                target_s[i] = Nsample*target[i] + (1-Nsample)*target[0]
            elif i == 17:
                target_s[i] = Nsample*target[i] + (1-Nsample)*target[6]
            elif i== 23:
                target_s[i] = Nsample*target[i] + (1-Nsample)*target[18]
            else:
                target_s[i] = Nsample*target[i] + (1-Nsample)*target[i+1]
    else:
        target_s = None


    return ema_batch_input, target_s

def mix_up_spec(batch_input, Nsample, target=None):
    ema_batch_input = batch_input.clone()
    for i in range(0,23+1):
        if i==5:
            ema_batch_input[i] = Nsample*batch_input[i] + (1-Nsample)*batch_input[0]
        elif i == 17:
            ema_batch_input[i] = Nsample*batch_input[i] + (1-Nsample)*batch_input[6]
        elif i== 23:
            ema_batch_input[i] = Nsample*batch_input[i] + (1-Nsample)*batch_input[18]
        else:
            ema_batch_input[i] = Nsample*batch_input[i] + (1-Nsample)*batch_input[i+1]
    if target is not None:
        target_s = target.clone()
        for i in range(0,23+1):
            # 사실 weak target 같은 경우에는 바꿀 필요 없지만 unlabled 바꿔야 하니까 그냥 바꿔주자
            if i==5:
                target_s[i] = Nsample*target[i] + (1-Nsample)*target[0]
            elif i == 17:
                target_s[i] = Nsample*target[i] + (1-Nsample)*target[6]
            elif i== 23:
                target_s[i] = Nsample*target[i] + (1-Nsample)*target[18]
            else:
                target_s[i] = Nsample*target[i] + (1-Nsample)*target[i+1]
        target_s_one = (target_s > 0.0).float().clone()
        target_one = (target > 0.0).float().clone()
        target_s_sum = torch.sum(target_s_one.clone(), dim=2)
        target_sum = torch.sum(target_one.clone(), dim=2)
        spec_list = []
        # target_s_copy = target_s.clone()
        # ema_batch_input_copy = ema_batch_input.clone()
        for i in range(18, 23+1):
            for j in range(0, 156+1):
                if target_s_sum[i][j] >= 3:
                    if target_sum[i][j] < target_s_sum[i][j]:
                        # target_s_copy = target_s.clone()
                        # ema_batch_input_copy = ema_batch_input.clone()

                        # target_s[i, j, :] = 0
                        ema_batch_input[i, :, j, :] = 0

                        # target_s = target_s_copy
                        # ema_batch_input = ema_batch_input_copy

                        spec_list.append((i, j))
    else:
        target_s = None
    
    return ema_batch_input, target_s, spec_list

def strong_spec(strong_pred_rolling, spec_list):
    strong_pred_rolling_copy = strong_pred_rolling.clone()
    for i in spec_list:
        strong_pred_rolling_copy[i[0], i[1], :] = 0
    return strong_pred_rolling_copy



def G_noise(features,snr=None):
    if snr is None:
        return features
    
    new_features = features.clone()
    new_features_cpu = new_features.cpu()
    new_features_numpy = new_features_cpu.numpy()
    feat = new_features_numpy[0][0]
    
    data = torch.zeros(24, 1, 628, 128)
    nbatch, _, nfrq, nfrm = new_features_cpu.shape
    for bter in range(nbatch):
        feat = new_features_numpy[bter]
        std = np.sqrt(np.mean((feat ** 2) * (10 ** (-snr / 10)), axis=-2))
        temp = feat.squeeze() + std*np.random.randn(nfrq, nfrm)
        data[bter,0] = torch.tensor(temp)
    return data
    # return data.cuda()

def shifting(batch_input, target):
    Nsample = 40
    # 순정 들어가서 나중에 G-noise 추가
    batch_input_roll, target_s= ema_input_target(Nsample,batch_input, target)
    zero_tensor_input = torch.zeros((24, 1, 628, 128), device='cuda')
    zero_tensor_input = zero_tensor_input[:,:, :4*Nsample , :]
    batch_input_sliced = batch_input_roll[:,:,4*Nsample : , :]
    #돌린 만큼 0tensor 합치기
    batch_input_shifting = torch.cat([zero_tensor_input,batch_input_sliced],dim=2)
    #G-noise 추가
    # ema_batch_input = G_noise(batch_input_shifting,snr)
    zero_tensor = torch.zeros((24, 157, 10)).cuda()
    zero_tensor = zero_tensor[:, :Nsample, :]
    target_s = target_s[:, Nsample : , :].cuda()
    # target_s에 0tensor 추가
    target_s = torch.cat([zero_tensor,target_s],dim=1)
    
    return batch_input_shifting, target_s


def cut_mix(Nsample, target, batch_input=None):
    if batch_input is not None:    
        ema_batch_input = batch_input.clone()
        for i in range(0,23+1):
            if i==5:
                batch_input_i = batch_input[i].clone()
                batch_input_0 = batch_input[0].clone()
                batch_input_i = batch_input_i[:, :4*Nsample , :]
                batch_input_0 = batch_input_0[:,4*Nsample : , :]
                ema_batch_input[i] = torch.cat([batch_input_i,batch_input_0],1)
            elif i == 17:
                batch_input_i = batch_input[i].clone()
                batch_input_0 = batch_input[6].clone()
                batch_input_i = batch_input_i[:, :4*Nsample , :]
                batch_input_0 = batch_input_0[:,4*Nsample : , :]
                ema_batch_input[i] = torch.cat([batch_input_i,batch_input_0],1)
            elif i== 23:
                batch_input_i = batch_input[i].clone()
                batch_input_0 = batch_input[18].clone()
                batch_input_i = batch_input_i[:, :4*Nsample , :]
                batch_input_0 = batch_input_0[:,4*Nsample : , :]
                ema_batch_input[i] = torch.cat([batch_input_i,batch_input_0],1)
            else:
                batch_input_i = batch_input[i].clone()
                batch_input_0 = batch_input[i+1].clone()
                batch_input_i = batch_input_i[:, :4*Nsample , :]
                batch_input_0 = batch_input_0[:,4*Nsample : , :]
                ema_batch_input[i] = torch.cat([batch_input_i,batch_input_0],1)

    else:
        ema_batch_input = None

    target_s = target.clone()
    for i in range(0,23+1):
        # 사실 weak target 같은 경우에는 바꿀 필요 없지만 unlabled 바꿔야 하니까 그냥 바꿔주자
        if i==5:
            target_i = target[i].clone()
            target_0 = target[0].clone()
            target_i = target_i[ :Nsample , :]
            target_0 = target_0[Nsample : , :]
            target_s[i] = torch.cat([target_i,target_0],0)
        elif i == 17:
            target_i = target[i].clone()
            target_0 = target[6].clone()
            target_i = target_i[ :Nsample , :]
            target_0 = target_0[Nsample : , :]
            target_s[i] = torch.cat([target_i,target_0],0)
        elif i== 23:
            target_i = target[i].clone()
            target_0 = target[18].clone()
            target_i = target_i[ :Nsample , :]
            target_0 = target_0[Nsample : , :]
            target_s[i] = torch.cat([target_i,target_0],0)
        else:
            target_i = target[i].clone()
            target_0 = target[i+1].clone()
            target_i = target_i[ :Nsample , :]
            target_0 = target_0[Nsample : , :]
            target_s[i] = torch.cat([target_i,target_0],0)


    return ema_batch_input, target_s


def frequency_mask(features, mask_ratios=(10, 50)):
    _, _,_,n_frame = features.shape
    t_width = torch.randint(low=int(n_frame/mask_ratios[1]), high=int(n_frame/mask_ratios[0]), size=(1,))   # [low, high)
    t_low = torch.randint(low=0, high=n_frame-t_width[0], size=(1,))
    features[:, :, :, t_low :(t_low+t_width)] = 0
    return features

def frequency_shifting(features, frequency_shift):
    original_tensor = features.clone()
    bs, channel, time, frequency = features.shape
    rolled_tensor = torch.roll(original_tensor, shifts=frequency_shift, dims=-1)
    zero_tensor = torch.zeros((24, 1, 628, 128), device='cuda')
    if frequency_shift> 0:
        zero_tensor = zero_tensor[:,:,:,:frequency_shift]
        rolled_tensor = rolled_tensor[:,:,:,frequency_shift:]
        padded_tensor = torch.cat((zero_tensor, rolled_tensor), dim=-1)
    else:
        zero_tensor = zero_tensor[:,:,:,:abs(frequency_shift)]
        rolled_tensor = rolled_tensor[:,:,:,:frequency + frequency_shift]
        padded_tensor = torch.cat((rolled_tensor, zero_tensor), dim=-1)

    return padded_tensor


def cut_mix_spec(target, batch_input):
    target_s = target.clone()
    ema_batch_input = batch_input.clone()
    target_one = (target > 0.0).float().clone()
    target_sum = torch.sum(target_one.clone(), dim=2)
    Start, Finish = 0, 0
    
    for i in range(18, 23+1):
        for j in range(0, 156+1):
            if target_sum[i][j] == 1:
                if Start == 0:
                    Start = j

            elif target_sum[i][j] > 1:
                # print("aaaaaaaaaaaaaaaaaaaaaa")
                Start = 0

            else:
                if Start != 0:
                    Finish = j
                    pad_length = Finish - Start
                    if target_sum[i][Start-1] > 1:
                        Start, Finish = 0, 0
                        continue
                    ema_batch_input[i, :, Start*4-int(pad_length/2):Start*4, :] = 0
                    ema_batch_input[i, :, Finish*4:Finish*4+int(pad_length/2), :] = 0
                    Start, Finish = 0, 0
    
    return ema_batch_input, target_s

def mask_magnitude(batch_input):
    tmp = batch_input.clone()
    tmp = torch.exp(tmp)
    threshold_value = 1.0
    magnitudes = torch.abs(tmp)
    mask = magnitudes < threshold_value
    tmp[mask] = 1.0e-16
    tmp = torch.log(tmp)
    return tmp

def mask_magnitude_plus(batch_input):
    tmp = batch_input.clone()
    tmp = torch.exp(tmp)
    tmp_ = tmp.clone()
    threshold_value = torch.mean(tmp)*0.2# 0.4, 0.6, 0.8
    # threshold_value = 1.0 
    magnitudes = torch.abs(tmp)
    mask = magnitudes < threshold_value
    tmp[mask] = 0
    # a = random.randrange(1,10)/10
    # tmp__ = (a*tmp + (1-a)*tmp_)/2
    tmp__ = tmp/2 + tmp_
    tmp = torch.log(tmp__)
    return tmp


# def mask_magnitude_plus(batch_input):
#     tmp = batch_input.clone()
#     tmp = torch.exp(tmp)
#     tmp_ = tmp.clone()
#     # threshold_value = torch.mean(tmp)*0.8 # 0.4, 0.6, 0.8
#     threshold_value = 1.0
#     magnitudes = torch.abs(tmp)
#     mask = magnitudes > threshold_value
#     tmp[mask] = 1.0e+16
#     tmp__ = tmp - tmp_
#     tmp = torch.log(tmp__)
#     return tmp