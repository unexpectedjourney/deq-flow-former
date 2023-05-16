import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange


class Refiner(nn.Module):
    def __init__(self, dim, max_pos_side=100, heads=4, dim_head=128):
        super(Refiner, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

        self.reduce_size = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
        )

        self.to_v_flow = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_v_net = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_v_inp = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma_flow = nn.Parameter(torch.zeros(1))
        self.gamma_net = nn.Parameter(torch.zeros(1))
        self.gamma_inp = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project_flow = nn.Conv2d(inner_dim, dim, 1, bias=False)
            self.project_net = nn.Conv2d(inner_dim, dim, 1, bias=False)
            self.project_inp = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project_flow = None
            self.project_net = None
            self.project_inp = None

    def prepare_flow(self, flow):
        flow = self.reduce_size(flow)
        return flow

    def forward(self, prev_frame, curr_frame, prev_flow, prev_net, prev_inp):
        heads, _, _, h, w = self.heads, *prev_frame.shape

        q = self.to_q(curr_frame)
        k = self.to_k(prev_frame)

        q, k = map(
            lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k)
        )
        q = self.scale * q
        sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)
        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)

        prev_flow = self.prepare_flow(prev_flow)

        v_flow = self.to_v_flow(prev_flow)
        v_net = self.to_v_net(prev_net)
        v_inp = self.to_v_inp(prev_inp)

        v_flow = rearrange(v_flow, 'b (h d) x y -> b h (x y) d', h=heads)
        v_net = rearrange(v_net, 'b (h d) x y -> b h (x y) d', h=heads)
        v_inp = rearrange(v_inp, 'b (h d) x y -> b h (x y) d', h=heads)

        out_flow = einsum('b h i j, b h j d -> b h i d', attn, v_flow)
        out_flow = rearrange(out_flow, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        out_net = einsum('b h i j, b h j d -> b h i d', attn, v_net)
        out_net = rearrange(out_net, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        out_inp = einsum('b h i j, b h j d -> b h i d', attn, v_inp)
        out_inp = rearrange(out_inp, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        if self.project is not None:
            out_flow = self.project_flow(out_flow)
            out_net = self.project_net(out_net)
            out_inp = self.project_inp(out_inp)

        out_flow = prev_flow + self.gamma_flow * out_flow
        out_net = prev_net + self.gamma_net * out_net
        out_inp = prev_inp + self.gamma_inp * out_inp

        return out_flow, out_net, out_inp
