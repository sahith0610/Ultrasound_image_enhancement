class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -freq)
        args = t[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, up=False):
        super().__init__()
        self.conv1  = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point1 = nn.Conv2d(in_ch, out_ch, 1)
        self.norm1  = nn.GroupNorm(4, out_ch)
        self.act1   = nn.ReLU(inplace=True)
        self.time_proj = nn.Linear(time_dim, out_ch)
        Conv2 = nn.ConvTranspose2d if up else nn.Conv2d
        self.conv2 = Conv2(out_ch, out_ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(4, out_ch)
        self.act2  = nn.ReLU(inplace=True)

    def forward(self, x, t_emb):
        h = self.act1(self.norm1(self.point1(self.conv1(x))))
        t = self.time_proj(t_emb).view(-1, h.size(1), 1, 1)
        h = h + t
        h = self.act2(self.norm2(self.conv2(h)))
        return h


class TimeConditionedUNet(nn.Module):
    def __init__(self, in_ch=1, base=16, time_dim=32, depth=4):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        skip_chs = [base * (2**i) for i in range(depth)]
        self.downs = nn.ModuleList()
        ch = in_ch
        for out_ch in skip_chs:
            self.downs.append(ConvBlock(ch, out_ch, time_dim, up=False))
            ch = out_ch
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch*2, ch, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.ups = nn.ModuleList()
        for skip in reversed(skip_chs):
            self.ups.append(ConvBlock(ch + skip, skip, time_dim, up=True))
            ch = skip
        self.final = nn.Conv2d(ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        skips = []; h = x
        for down in self.downs:
            h = down(h, t_emb); skips.append(h)
            h = F.avg_pool2d(h, 2)
        h = self.bottleneck(h)
        for up in self.ups:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            h = torch.cat([h, skips.pop()], dim=1)
            h = up(h, t_emb)
        return self.final(h)

class Discriminator(nn.Module):
    def __init__(self, in_ch=1, base=16, num_layers=4):
        super().__init__()
        layers = []; ch = in_ch
        for i in range(num_layers):
            out = base * (2**i)
            layers += [nn.Conv2d(ch, out, 4, 2, 1),
                       nn.LeakyReLU(0.2, inplace=True)]
            ch = out
        layers.append(nn.Conv2d(ch, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)