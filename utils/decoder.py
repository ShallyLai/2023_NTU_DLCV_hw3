import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# TODO Add cross attention and BEFT

class Config:
    def __init__(self, checkpoint=None, cross_attention = None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd =  768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint # pre-trained weight from TA
        self.cross_pth = cross_attention

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn_CA = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj_CA = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x, x_encoder):
        #print("x.shape:", x.shape) # torch.Size([16, 128, 768]) 
        #print("x_encoder.shape:", x_encoder.shape) # torch.Size([16, 197, 768])

        B, T, C = x.size()  # batch, context, embedding
        T_encoder = x_encoder.size(1)

        q, k, v = self.c_attn_CA(x).split(self.n_embd, dim=2) # queries, keys, and values
        _, k_encoder, v_encoder = self.c_attn_CA(x_encoder).split(self.n_embd, dim=2)

        k = k_encoder.view(B, T_encoder, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v_encoder.view(B, T_encoder, self.n_head, C // self.n_head).transpose(1, 2)
        #print(v.shape)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:, :, :T, :T_encoder] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        return self.c_proj_CA((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd) # for cross attention
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        #self.cross_attn = CrossAttention(cfg)  # Cross-Attention module
        
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))
        
        # multi-head attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=cfg.n_embd,
            num_heads=cfg.n_head
        )

    def forward(self, x, encoder_x):
        x = x + self.attn(self.ln_1(x))

        x = x.permute(1, 0, 2)
        encoder_x = encoder_x.permute(1, 0, 2)
        att_output, _ = self.multihead_attention(self.ln_2(x), encoder_x, encoder_x)
        x_cross = att_output.permute(1, 0, 2)

        #x_cross = x + self.cross_attn(self.ln_2(x), encoder_x) 
        x_mlp = x_cross + self.mlp(self.ln_3(x_cross))
        return x_mlp

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)
            # Freeze these weights
            for name, param in self.transformer.named_parameters():
                if any(name.endswith(w) for w in transposed):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        # Load trained weight
        #if self.cfg.cross_pth is not None:
        #    saved_weights = torch.load(self.cfg.cross_pth)
        #    model_state_dict = self.transformer.state_dict()
        #    for key in model_state_dict.keys():
        #        if key in saved_weights:
        #            model_state_dict[key] = saved_weights[key]
        #    self.transformer.load_state_dict(model_state_dict, strict = False)

    def forward(self, x: Tensor, image_input: Tensor):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        for i, block in enumerate(self.transformer.h):
            #print(i) # 0~11
            x = x + block(x, image_input)
        x = self.lm_head(self.transformer.ln_f(x)) 

        return x

