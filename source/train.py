def train(args):

    betas = torch.linspace(1e-4, 0.02, args.timesteps)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    def q_sample(x0, t, noise):
        a = alphas_bar[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise


    Gf = Generator(1, args.base, args.time_dim, args.unet_depth)
    Gb = Generator(1, args.base, args.time_dim, args.unet_depth)
    Df = Discriminator(1, args.base, args.disc_layers)
    Db = Discriminator(1, args.base, args.disc_layers)

    opt_G = optim.Adam(list(Gf.parameters()) + list(Gb.parameters()), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(list(Df.parameters()) + list(Db.parameters()), lr=args.lr_d, betas=(0.5, 0.999))

    mse, l1 = nn.MSELoss(), nn.L1Loss()

    layer_ids = [3, 8, 15, 22, 29]
    w_i = torch.tensor([1, .75, .5, .25, .1])
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    def phi_loss(x, y):
        def prep(t):
            return F.interpolate(t.repeat(1, 3, 1, 1), size=224, mode='bilinear', align_corners=False)
        fx, fy = [], []
        h = prep(x)
        for i, l in enumerate(vgg):
            h = l(h)
            if i in layer_ids:
                fx.append(h)
        h = prep(y)
        for i, l in enumerate(vgg):
            h = l(h)
            if i in layer_ids:
                fy.append(h)
        dists = [F.mse_loss(a, b, reduction='none').mean([1, 2, 3]).sqrt() for a, b in zip(fx, fy)]
        return (torch.stack(dists, 1) @ w_i).mean()

    T = args.timesteps
    
    ds = Ultrasounddataset(args.dataA, args.dataB, args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    for ep in range(1, args.epochs + 1):
        for realA, realB in dl:

            with torch.no_grad():
                fakeB_det = Gf(realA, torch.rand(realA.size(0)))
                fakeA_det = Gb(realB, torch.rand(realB.size(0)))
            loss_D = (
                mse(Df(realB), torch.ones_like(Df(realB))) +
                mse(Df(fakeB_det), torch.zeros_like(Df(fakeB_det))) +
                mse(Db(realA), torch.ones_like(Db(realA))) +
                mse(Db(fakeA_det), torch.zeros_like(Db(fakeA_det)))
            )
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            eps = torch.randn_like(realA)
            t_ddpm = torch.randint(0, T, (realA.size(0),))
            x_noisy = q_sample(realA, t_ddpm, eps)
            pred_eps = Gf(x_noisy, t_ddpm.float() / (T - 1))
            ddpm_loss = mse(pred_eps, eps)

            t_norm = torch.randint(0, T, (realA.size(0),)).float() / (T - 1)
            fakeB = Gf(realA, t_norm)
            fakeA = Gb(realB, t_norm)
            adv_loss = (
                mse(Df(fakeB), torch.ones_like(Df(fakeB))) +
                mse(Db(fakeA), torch.ones_like(Db(fakeA)))
            )
            cyc_loss = (
                l1(Gb(fakeB, t_norm), realA) +
                l1(Gf(fakeA, t_norm), realB)
            )
            id_loss = (
                l1(Gb(realA, torch.zeros_like(t_norm)), realA) +
                l1(Gf(realB, torch.zeros_like(t_norm)), realB)
            )
            per_loss = phi_loss(fakeB, realB) + phi_loss(fakeA, realA)

            loss_G = (
                args.lambda_adv * adv_loss +
                args.lambda_cycle * cyc_loss +
                args.lambda_id * id_loss +
                args.lambda_per * per_loss +
                args.lambda_ddpm * ddpm_loss
            )
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
