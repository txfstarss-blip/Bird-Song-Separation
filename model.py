import json
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import fold, unfold

from asteroid import torch_utils
from asteroid.models import BaseModel
from asteroid_filterbanks import make_enc_dec
from asteroid.engine.optimizers import make_optimizer
from asteroid.masknn import activations, norms
from asteroid.masknn.recurrent import DPRNNBlock
from asteroid.models.base_models import _shape_reconstructed, _unsqueeze_to_3d
from asteroid.utils.generic_utils import has_arg
from asteroid.utils.torch_utils import pad_x_to_y, script_if_tracing, jitable_shape
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# ==========================================
# 新增增强模块：SENet & EnhancedSelector
# ==========================================

class SENet(nn.Module):
    """Squeeze-and-Excitation 模块，用于特征通道重校准"""
    def __init__(self, channels, reduction=4):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnhancedSelector(nn.Module):
    """
    改进的强化计数器模块：
    1. 保持 4D 结构 [B, C, K, S]，利用 2D 卷积捕捉时间上下文。
    2. 并行分支：3x3 标准卷积 + 3x3 空洞卷积（扩张率2，模拟 5x5 感受野）。
    3. 引入 SENet 注意力机制增强关键特征。
    """
    def __init__(self, bn_chan, n_srcs):
        super(EnhancedSelector, self).__init__()
        # 分支1：局部特征
        self.branch1 = nn.Conv2d(bn_chan, 32, kernel_size=3, padding=1)
        # 分支2：跨步特征（空洞卷积）
        self.branch2 = nn.Conv2d(bn_chan, 32, kernel_size=3, padding=2, dilation=2)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            SENet(64),
            nn.ReLU()
        )
        
        # 全局池化与分类
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, len(n_srcs))

    def forward(self, x):
        # x: [Batch, Channels, Chunk_Size, N_Chunks]
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        x = torch.cat([out1, out2], dim=1)
        x = self.fusion(x)
        x = self.avg_pool(x).view(x.size(0), -1)
        return self.fc(x)

# ==========================================
# 原有模型结构修改
# ==========================================

def make_model_and_optimizer(conf, sample_rate):
    model = MultiDecoderDPRNN(**conf["masknet"], **conf["filterbank"], sample_rate=sample_rate)
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    return model, optimizer


class MultiDecoderDPRNN(BaseModel):
    def __init__(
        self,
        n_srcs,
        bn_chan=128,
        hid_size=128,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        mask_act="sigmoid",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
        kernel_size=16,
        n_filters=64,
        stride=8,
        encoder_activation=None,
        use_mulcat=False,
        sample_rate=8000,
    ):
        super().__init__(sample_rate=sample_rate)
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or "linear")()
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.encoder, _ = make_enc_dec(
            "free",
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
        )
        self.masker = DPRNN_MultiStage(
            in_chan=n_filters,
            bn_chan=bn_chan,
            hid_size=hid_size,
            chunk_size=chunk_size,
            hop_size=hop_size,
            n_repeats=n_repeats,
            norm_type=norm_type,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            use_mulcat=use_mulcat,
            num_layers=num_layers,
            dropout=dropout,
        )
        # 调用修改后的选择器逻辑
        self.decoder_select = Decoder_Select(
            kernel_size=kernel_size,
            stride=stride,
            in_chan=n_filters,
            n_srcs=n_srcs,
            bn_chan=bn_chan,
            chunk_size=chunk_size,
            hop_size=hop_size,
            mask_act=mask_act,
        )

    def forward(self, wav, ground_truth=None):
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)
        tf_rep = self.enc_activation(self.encoder(wav))
        est_masks_list = self.masker(tf_rep)
        decoded, selector_output = self.decoder_select(
            est_masks_list, tf_rep, ground_truth=ground_truth
        )
        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape), _shape_reconstructed(
            selector_output, shape
        )

    def forward_wav(self, wav, slice_size=32000, *args, **kwargs):
        assert not self.training, "forward_wav is only used for test mode"
        T = wav.size(-1)
        if wav.ndim == 1:
            wav = wav.reshape(1, wav.size(0))
        assert wav.ndim == 2
        slice_stride = slice_size // 2
        T_padded = max(int(np.ceil(T / slice_stride)), 2) * slice_stride
        wav = F.pad(wav, (0, T_padded - T))
        slices = wav.unfold(dimension=-1, size=slice_size, step=slice_stride)
        slice_nb = slices.size(1)
        slices = slices.squeeze(0).unsqueeze(1)
        tf_rep = self.enc_activation(self.encoder(slices))
        est_masks_list = self.masker(tf_rep)
        
        # 修改点：使用 EnhancedSelector 进行推理
        selector_input = est_masks_list[-1]
        selector_output = self.decoder_select.selector(selector_input)
        est_idx, _ = selector_output.argmax(-1).mode()
        est_spks = self.decoder_select.n_srcs[est_idx]
        
        output_wavs, _ = self.decoder_select(
            est_masks_list, tf_rep, ground_truth=[est_spks] * slice_nb
        )
        output_wavs = output_wavs.squeeze(1)[:, :est_spks, :]
        
        output_cat = output_wavs.new_zeros(est_spks, slice_nb * slice_size)
        output_cat[:, :slice_size] = output_wavs[0]
        start = slice_stride
        for i in range(1, slice_nb):
            overlap_prev = output_cat[:, start : start + slice_stride].unsqueeze(0)
            overlap_next = output_wavs[i : i + 1, :, :slice_stride]
            pw_losses = pairwise_neg_sisdr(overlap_next, overlap_prev)
            _, best_indices = PITLossWrapper.find_best_perm(pw_losses)
            reordered = PITLossWrapper.reorder_source(output_wavs[i : i + 1, :, :], best_indices)
            output_cat[:, start : start + slice_size] += reordered.squeeze(0)
            output_cat[:, start : start + slice_stride] /= 2
            start += slice_stride
        return output_cat[:, :T]


class DPRNN_MultiStage(nn.Module):
    def __init__(self, in_chan, bn_chan, hid_size, chunk_size, hop_size, n_repeats, norm_type, bidirectional, rnn_type, use_mulcat, num_layers, dropout):
        super(DPRNN_MultiStage, self).__init__()
        self.bn_chan = bn_chan
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        self.net = nn.ModuleList([])
        for i in range(self.n_repeats):
            self.net.append(
                DPRNNBlock(bn_chan, hid_size, norm_type=norm_type, bidirectional=bidirectional, rnn_type=rnn_type, use_mulcat=use_mulcat, num_layers=num_layers, dropout=dropout)
            )

    def forward(self, mixture_w):
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        output = unfold(
            output.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        n_chunks = output.shape[-1]
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        output_list = []
        for i in range(self.n_repeats):
            output = self.net[i](output)
            output_list.append(output)
        return output_list


class SingleDecoder(nn.Module):
    def __init__(self, kernel_size, stride, in_chan, n_src, bn_chan, chunk_size, hop_size, mask_act):
        super(SingleDecoder, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_chan = in_chan
        self.bn_chan = bn_chan
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_src = n_src

        net_out_conv = nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, in_chan, 1, bias=False)

        mask_nl_class = activations.get(mask_act)
        self.output_act = mask_nl_class(dim=1) if has_arg(mask_nl_class, "dim") else mask_nl_class()

        _, self.trans_conv = make_enc_dec("free", kernel_size=kernel_size, stride=stride, n_filters=in_chan)

    def forward(self, output, mixture_w):
        batch, bn_chan, chunk_size, n_chunks = output.size()
        _, in_chan, n_frames = mixture_w.size()
        output = self.first_out(output)
        output = output.reshape(batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks)
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(
            output.reshape(batch * self.n_src, to_unfold, n_chunks),
            (n_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        output = output.reshape(batch * self.n_src, self.bn_chan, -1)
        output = self.net_out(output) * self.net_gate(output)
        score = self.mask_net(output)
        est_mask = self.output_act(score)
        est_mask = est_mask.reshape(batch, self.n_src, self.in_chan, n_frames)
        source_w = est_mask * mixture_w.unsqueeze(1)
        source_w = source_w.reshape(batch * self.n_src, self.in_chan, n_frames)
        est_wavs = self.trans_conv(source_w)
        return est_wavs.reshape(batch, self.n_src, -1)


class Decoder_Select(nn.Module):
    def __init__(self, kernel_size, stride, in_chan, n_srcs, bn_chan, chunk_size, hop_size, mask_act):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_srcs = n_srcs
        self.chunk_size = chunk_size

        self.n_src2idx = {n_src: i for i, n_src in enumerate(n_srcs)}
        self.decoders = torch.nn.ModuleList()
        for n_src in n_srcs:
            self.decoders.append(
                SingleDecoder(kernel_size, stride, in_chan, n_src, bn_chan, chunk_size, hop_size, mask_act)
            )
        
        # 这里应用了 EnhancedSelector 替换原有的 Sequential 结构
        self.selector = EnhancedSelector(bn_chan, n_srcs)

    def forward(self, output_list, mixture_w, ground_truth):
        batch, bn_chan, chunk_size, n_chunks = output_list[0].size()
        _, in_chan, n_frames = mixture_w.size()
        if not self.training:
            output_list = output_list[-1:]
        num_stages = len(output_list)
        
        # 保持 4D 形状输入到 selector
        output = torch.stack(output_list, 1).reshape(batch * num_stages, bn_chan, chunk_size, n_chunks)
        selector_output = self.selector(output).reshape(batch, num_stages, -1)
        
        output = output.reshape(batch, num_stages, bn_chan, chunk_size, n_chunks)
        mixture_w_rep = mixture_w.unsqueeze(1).repeat(1, num_stages, 1, 1)
        
        if ground_truth is not None:
            decoder_selected = torch.LongTensor([self.n_src2idx[truth] for truth in ground_truth])
        else:
            decoder_selected = selector_output.reshape(batch, -1).argmax(1)
            
        T = self.kernel_size + self.stride * (n_frames - 1)
        output_wavs = torch.zeros(batch, num_stages, max(self.n_srcs), T).to(output.device)
        for i in range(batch):
            idx = decoder_selected[i]
            output_wavs[i, :, :self.n_srcs[idx], :] = self.decoders[idx](output[i], mixture_w_rep[i])
        return output_wavs, selector_output


def load_best_model(train_conf, exp_dir, sample_rate):
    model, _ = make_model_and_optimizer(train_conf, sample_rate=sample_rate)
    try:
        with open(os.path.join(exp_dir, "best_k_models.json"), "r") as f:
            best_k = json.load(f)
        best_model_path = min(best_k, key=best_k.get)
    except FileNotFoundError:
        all_ckpt = os.listdir(os.path.join(exp_dir, "checkpoints/"))
        all_ckpt = [(ckpt, int("".join(filter(str.isdigit, os.path.basename(ckpt))))) for ckpt in all_ckpt if "ckpt" in ckpt]
        all_ckpt.sort(key=lambda x: x[1])
        best_model_path = os.path.join(exp_dir, "checkpoints", all_ckpt[-1][0])
    checkpoint = torch.load(best_model_path, map_location="cpu")
    model = torch_utils.load_state_dict_in(checkpoint["state_dict"], model)
    model.eval()
    return model

if __name__ == "__main__":
    network = MultiDecoderDPRNN(n_srcs=[2, 3], bn_chan=32, hid_size=32, n_filters=16)
    input_data = torch.rand(2, 3200)
    wavs, selector_output = network(input_data, [3, 2])
    print(f"Training shape: {wavs.shape}")
    network.eval()
    wavs, selector_output = network(input_data)
    print(f"Eval shape: {wavs.shape}")