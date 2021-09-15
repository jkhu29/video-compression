import torch
import numpy as np
import torch.nn as nn
import torch


class Memory_Attention_Layer(nn.Module):
    def __init__(self, input_channels: int = 64, hidden_channels: int = 64, num_heads: int = 8):
        super(Memory_Attention_Layer, self).__init__()
        assert input_channels % num_heads == 0
        self.num_heads = num_heads
        self.single_head_channel = input_channels // num_heads
        self.single_hidden_channel = hidden_channels // num_heads
        self.Keys = []
        self.Queries = []
        self.Values = []
        for i in range(num_heads):
            self.Keys.append(nn.Linear(self.single_head_channel, self.single_hidden_channel))
            self.Queries.append(nn.Linear(self.single_head_channel, self.single_hidden_channel))
            self.Values.append(nn.Linear(self.single_head_channel, self.single_head_channel))
        self.Keys = nn.ModuleList(self.Keys)
        self.Queries = nn.ModuleList(self.Queries)
        self.Values = nn.ModuleList(self.Values)
        self.softmax = nn.Softmax(dim=-1)
        self.layernorm0x = nn.LayerNorm([input_channels])
        self.layernorm1x = nn.LayerNorm([input_channels])
        self.layernorm0k = nn.LayerNorm([hidden_channels])
        self.layernorm1k = nn.LayerNorm([hidden_channels])
        self.layernorm0q = nn.LayerNorm([hidden_channels])
        self.layernorm1q = nn.LayerNorm([hidden_channels])
        self.FFNx = nn.Sequential(nn.Linear(input_channels, input_channels), nn.ReLU(), nn.Linear(input_channels, input_channels))
        self.FFNk = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.FFNq = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.learnable_position_emb = nn.Embedding(256, input_channels)

    def get_kqv(self, x):
        # x.shape (N,S,input_channels)
        K = []
        Q = []
        V = []
        for i in range(self.num_heads):
            x_single_head = x[:, :, self.single_head_channel*i:self.single_head_channel*(i+1)]
            K.append(self.Keys[i](x_single_head))
            # (N, S, H/num)
            Q.append(self.Queries[i](x_single_head))
            # (N, S, H/num)
            V.append(self.Values[i](x_single_head))
            # (N, S, C/num)

        return torch.cat(K, dim=-1), torch.cat(Q, dim=-1), torch.cat(V, dim=-1)

    def cross_attention(self, k, q, v, cls="x"):
        # k (N, S, H)
        # q (N, S, H)
        # v (N, S, C)
        _, _, h = k.shape
        k_T = k.permute((0, 2, 1)) #(N,H,S)
        weights = self.softmax(torch.bmm(q, k_T)) / np.sqrt(h) #(N,S,S)
        output = torch.bmm(weights,v)#(N,S,C)
        norm0 = getattr(self, 'layernorm0{}'.format(cls))
        norm1 = getattr(self, 'layernorm1{}'.format(cls))
        FFN = getattr(self, 'FFN{}'.format(cls))
        z = norm0(output + v)
        return norm1(FFN(z) + z)

    def forward(self, x):
        # x shape (N, F, S, C)
        _, num, length, _ = x.shape
        loc_emb = self.learnable_position_emb(
            torch.arange(length).to(x.device)
        ).view(1, 1, length, -1)
        x += loc_emb
        k_memory, q_memory, v = self.get_kqv(x[:,0, ...])
        V = [v.unsqueeze(1)]
        K = [k_memory.unsqueeze(1)]
        Q = [q_memory.unsqueeze(1)]
        for i in range(1, num):
            k, q, v = self.get_kqv(x[:, i, ...])
            k = self.cross_attention(k, q_memory, k, cls='k')
            q = self.cross_attention(k_memory, q, q, cls='q')
            v = self.cross_attention(k, q, v, cls='x')
            k_memory = k
            q_memory = q
            V.append(v.unsqueeze(1))
            K.append(k.unsqueeze(1))
            Q.append(q.unsqueeze(1))
        return torch.cat(K, dim=1), torch.cat(Q, dim=1), torch.cat(V, dim=1)


class EntropyBlock(nn.Module):
    def __init__(self, input_channels: int = 64, hidden_channels: int = 64, num_layers: int = 1, num_heads:int = 8):
        super(EntropyBlock, self).__init__()
        self.MAL_encoder = []
        self.MAL_decoder = []
        for i in range(num_layers):
            self.MAL_encoder.append(Memory_Attention_Layer(input_channels, hidden_channels, num_heads))
            self.MAL_decoder.append(Memory_Attention_Layer(input_channels, hidden_channels, num_heads))
        self.MAL_encoder = nn.ModuleList(self.MAL_encoder)
        self.MAL_decoder = nn.ModuleList(self.MAL_decoder)
        self.postlayer_k = nn.Sequential(nn.Linear(hidden_channels, input_channels))
        self.postlayer_q = nn.Sequential(nn.Linear(hidden_channels, input_channels), nn.ReLU())
    
    def forward(self, x):
        # x shape (N,F,S,C)
        for MAL in self.MAL_encoder:
            k, q, x = MAL(x)
        sigma = torch.clamp(self.postlayer_q(q), 1e-8, 1.0)
        mu = self.postlayer_k(k)
        normal = torch.distributions.Normal(mu, sigma)

        if self.training:
            noise = torch.empty_like(x).uniform_(-0.5, 0.5)
            compress_x = x + noise
        else:
            compress_x = torch.round(x)

        probs = normal.cdf(compress_x + 0.5) - normal.cdf(compress_x - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-8) / np.log(2.0), 0, 50))
        for MAL in self.MAL_decoder:
            k, q, compress_x = MAL(compress_x)
        return probs, total_bits, compress_x


if __name__ == "__main__":
    model = EntropyBlock(64, 128)
    model.train()
    x = torch.ones((2 ,3, 128, 64))
    probs, total_bits, rebuild_x = model(x)
    # (N,S,C)
    print(rebuild_x.shape)
    # (batch_size, frame_sequence, patches, input_channels)
