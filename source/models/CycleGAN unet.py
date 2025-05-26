class UNetGen(nn.Module):
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        def down(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(c_out, affine=True),
                nn.ReLU(True)
            )
        def up(c_in, c_out):
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(c_out, affine=True),
                nn.ReLU(True)
            )

        self.enc1 = down(in_ch,  base)
        self.enc2 = down(base,   base*2)
        self.enc3 = down(base*2, base*4)
        self.enc4 = down(base*4, base*8)

        self.bot  = nn.Sequential(
            nn.Conv2d(base*8, base*8, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(base*8, base*8, 3, 1, 1), nn.ReLU(True),
        )

        self.dec4 = up(base*8, base*4)
        self.dec3 = up(base*8, base*2)
        self.dec2 = up(base*4, base)
        self.dec1 = up(base*2, base)

        self.final = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bot(e4)

        d4 = self.dec4(b)
        d3 = self.dec3(torch.cat([d4,e3],1))
        d2 = self.dec2(torch.cat([d3,e2],1))
        d1 = self.dec1(torch.cat([d2,e1],1))

        return torch.tanh(self.final(d1))


class Disc(nn.Module):
    def __init__(self, in_ch=1, base=64, layers=3):
        super().__init__()
        seq = []
        ch = in_ch
        for i in range(layers):
            out = base*(2**i)
            seq += [
                nn.Conv2d(ch, out, 4, 2, 1),
                nn.InstanceNorm2d(out, affine=True) if i>0 else nn.Identity(),
                nn.LeakyReLU(0.2, True)
            ]
            ch = out
        seq += [nn.Conv2d(ch, 1, 4, 1, 1)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)