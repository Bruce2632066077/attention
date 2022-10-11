import torch
import numpy as np

def sequence_mask(size):
    """
    序列掩码，解码器输入数据时掩盖后续词的位置
    :param size: 生成词个数
    :return: 右上角为False，主对角线及左下角为True的bool矩阵
    """
    attn_shape = (1, size, size)
    """
    np.triu：返回函数的上三角矩阵A，k=1得到主对角线向上平移一个距离的对角线，
    即保留右上对角线及其以上的数据，其余置为0，即a_11=0
    """
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return (torch.from_numpy(mask) == 0).cuda()  # 通过==0返回的是bool矩阵，即矩阵元素为bool值

def src_trg_mask(src, r2l_trg, trg, pad_idx):
    """
    :param src: 编码器的输入
    :param r2l_trg: r2l方向解码器的输入
    :param trg: l2r方向解码器的输入
    :param pad_idx: pad的索引
    :return: trg为None，返回编码器输入的掩码；trg存在，返回编码器和解码器输入的掩码
    """
    # TODO: enc_src_mask是元组，是否可以改成list，然后修改这种冗余代码
    # 通过src的长短，即视频特征向量提取的模式，判断有多少种特征向量需要进行mask
    if isinstance(src, tuple) and len(src) == 4:
        # 不同模式的视频特征向量的mask
        src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)  # 二维特征向量
        src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)  # 三维特征向量
        src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)  # 目标检测特征向量
        src_rel_mask = (src[3][:, :, 0] != pad_idx).unsqueeze(1)  # 目标关系特征向量

        # 视频所有特征向量mask的拼接
        enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask, src_rel_mask)
        dec_src_mask = src_image_mask & src_motion_mask  # 视频二维和三维特征向量mask的拼接
        src_mask = (enc_src_mask, dec_src_mask)  # 视频最终的mask
    elif isinstance(src, tuple) and len(src) == 3:
        src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
        src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
        src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)

        enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask)
        dec_src_mask = src_image_mask & src_motion_mask
        src_mask = (enc_src_mask, dec_src_mask)
    elif isinstance(src, tuple) and len(src) == 2:
        src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
        src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)

        enc_src_mask = (src_image_mask, src_motion_mask)
        dec_src_mask = src_image_mask & src_motion_mask
        src_mask = (enc_src_mask, dec_src_mask)
    else:
        # 即只有src_image_mask，即二维特征的mask
        src_mask = src_image_mask = (src[:, :, 0] != pad_idx).unsqueeze(1)

    # 判断是否需要对trg，也就是解码器的输入进行掩码
    if trg and r2l_trg:
        """
        trg_mask是填充掩码和序列掩码，&前是填充掩码，&后是通过subsequent_mask函数得到的序列掩码
        其中type_as，是为了让序列掩码和填充掩码的维度一致
        """
        trg_mask = (trg != pad_idx).unsqueeze(1) & sequence_mask(trg.size(1)).type_as(src_image_mask.data)
        # r2l_trg的填充掩码
        r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_image_mask.data)
        # r2l_trg的填充掩码和序列掩码
        r2l_trg_mask = r2l_pad_mask & sequence_mask(r2l_trg.size(1)).type_as(src_image_mask.data)
        # src_mask[batch, 1, lens]  trg_mask[batch, 1, lens]
        return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask
    else:
        return src_mask
