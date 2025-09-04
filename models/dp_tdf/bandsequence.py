import torch
import torch.nn as nn


class RNNModule(nn.Module):
    """
    RNN submodule of BandSequence module
    """

    def __init__(
            self,
            group_num: int,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'lstm',
            bidirectional: bool = True
    ):
        super(RNNModule, self).__init__()
        self.groupnorm = nn.GroupNorm(group_num, input_dim_size)
        self.rnn = getattr(nn, rnn_type)(
            input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(
            hidden_dim_size * 2 if bidirectional else hidden_dim_size,
            input_dim_size
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        Input shape:
            across T - [batch_size, k_subbands, time, n_features]
            OR
            across K - [batch_size, time, k_subbands, n_features]
        """
        B, K, T, N = x.shape

        out = x.view(B * K, T, N)

        out = self.groupnorm(
            out.transpose(-1, -2)
        ).transpose(-1, -2)
        out = self.rnn(out)[0]
        out = self.fc(out)

        x = out.view(B, K, T, N) + x

        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class BandSequenceModelModule(nn.Module):
    """
    BandSequence (2nd) Module of BandSplitRNN.
    Runs input through n BiLSTMs in two dimensions - time and subbands.
    """

    def __init__(
            self,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'lstm',
            bidirectional: bool = True,
            num_layers: int = 12,
            n_heads: int = 4,
    ):
        super(BandSequenceModelModule, self).__init__()

        self.bsrnn = nn.ModuleList([])
        self.n_heads = n_heads

        input_dim_size = input_dim_size // n_heads
        hidden_dim_size = hidden_dim_size // n_heads
        group_num = input_dim_size // 16

        for _ in range(num_layers):
            rnn_across_t = RNNModule(
                group_num, input_dim_size, hidden_dim_size, rnn_type, bidirectional
            )
            rnn_across_k = RNNModule(
                group_num, input_dim_size, hidden_dim_size, rnn_type, bidirectional
            )
            self.bsrnn.append(
                nn.Sequential(rnn_across_t, rnn_across_k)
            )

    def forward(self, x: torch.Tensor):
        """
        Input shape: [batch_size, k_subbands, time, n_features]
        Output shape: [batch_size, k_subbands, time, n_features]
        """
        # x (b,c,t,f)
        b, c, t, f = x.shape
        x = x.view(b * self.n_heads, c // self.n_heads, t, f)

        x = x.permute(0, 3, 2, 1).contiguous()
        for i in range(len(self.bsrnn)):
            x = self.bsrnn[i](x)

        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(b, c, t, f)
        return x
