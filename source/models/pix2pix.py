class pixgen(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.downs = nn.ModuleList([
            down_block(in_ch, 64, norm=False), down_block(64,128),
            down_block(128,256), down_block(256,512), down_block(512,512),
            down_block(512,512), down_block(512,512), down_block(512,512)
        ])
        self.ups = nn.ModuleList([
            up_block(512,512,0.5), up_block(1024,512,0.5), up_block(1024,512,0.5),
            up_block(1024,512), up_block(1024,256), up_block(512,128), up_block(256,64)
        ])
        self.final = nn.Sequential(nn.ConvTranspose2d(128,out_ch,4,2,1), nn.Tanh())

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
        skips = skips[::-1]
        for i, u in enumerate(self.ups):
            x = u(x)
            x = torch.cat([x, skips[i+1]], dim=1)
        return self.final(x)

class pixdisc(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        def block(ic, oc, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, 2, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        self.model = nn.Sequential(
            block(in_ch,   32, norm=False),
            block(32,      64),
            block(64,     128),
            block(128,    256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, a, b):
        x = torch.cat([a, b], dim=1)
        return self.model(x)