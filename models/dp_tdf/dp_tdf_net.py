import torch.nn as nn
import torch

from models.dp_tdf.modules import TFC_TDF, TFC_TDF_Res1, TFC_TDF_Res2
from models.dp_tdf.bandsequence import BandSequenceModelModule
from models.dp_tdf.batch_norm import get_norm


class STFT:
    def __init__(self, n_fft, hop_length, dim_f):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = dim_f

    def __call__(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape([*batch_dims, c * 2, -1, x.shape[-1]])
        return x[..., :self.dim_f, :]

    def inverse(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x[..., 0] + x[..., 1] * 1.j
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True)
        x = x.reshape([*batch_dims, 2, -1])
        return x


class DPTDFNet(nn.Module):
    def __init__(self, num_blocks, l, g, k, bn, bias, bn_norm, bandsequence, block_type, 
                 dim_f=2048, dim_t=256, n_fft=6144, hop_length=1024, audio_ch=2, **kwargs):

        super(DPTDFNet, self).__init__()

        self.num_blocks = num_blocks
        self.l = l
        self.g = g
        self.k = k
        self.bn = bn
        self.bias = bias
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.audio_ch = audio_ch
        self.dim_c_in = audio_ch * 2
        self.dim_c_out = audio_ch * 2

        self.n = num_blocks // 2
        scale = (2, 2)

        if block_type == "TFC_TDF":
            T_BLOCK = TFC_TDF
        elif block_type == "TFC_TDF_Res1":
            T_BLOCK = TFC_TDF_Res1
        elif block_type == "TFC_TDF_Res2":
            T_BLOCK = TFC_TDF_Res2
        else:
            raise ValueError(f"Unknown block type {block_type}")

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_c_in, out_channels=g, kernel_size=(1, 1)),
            get_norm(bn_norm, g),
            nn.ReLU(),
        )

        f = self.dim_f
        c = g
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()

        for i in range(self.n):
            c_in = c

            self.encoding_blocks.append(T_BLOCK(c_in, c, l, f, k, bn, bn_norm, bias=bias))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c + g),
                    nn.ReLU()
                )
            )
            f = f // 2
            c += g

        self.bottleneck_block1 = T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias)
        self.bottleneck_block2 = BandSequenceModelModule(
            **bandsequence,
            input_dim_size=c,
            hidden_dim_size=2*c
        )

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c - g),
                    nn.ReLU()
                )
            )

            f = f * 2
            c -= g

            self.decoding_blocks.append(T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.dim_c_out, kernel_size=(1, 1)),
        )

        # Add STFT processing
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, dim_f=dim_f)

    def forward(self, x):
        '''
        Args:
            x: (batch, audio_ch, time) - raw audio input
        '''
        # Convert raw audio to spectrogram
        x = self.stft(x)
        
        x = self.first_conv(x)

        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        x = self.bottleneck_block1(x)
        x = self.bottleneck_block2(x)

        for i in range(self.n):
            x = self.us[i](x)
            x = x * ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        # Convert spectrogram back to audio
        x = self.stft.inverse(x)

        return x
