
## MS-SSIM
def compute_ms_ssim(x, y, max_val=1.0, levels=5, weights=None):
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    
    assert x.shape == y.shape
    assert len(weights) == levels

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    C3 = C2 / 2

    def compute_stats(a, b):
        mean_a = a.mean(dim=[2, 3], keepdim=True)
        mean_b = b.mean(dim=[2, 3], keepdim=True)
        var_a = ((a - mean_a) ** 2).mean(dim=[2, 3], keepdim=True)
        var_b = ((b - mean_b) ** 2).mean(dim=[2, 3], keepdim=True)
        cov_ab = ((a - mean_a) * (b - mean_b)).mean(dim=[2, 3], keepdim=True)
        return mean_a, mean_b, var_a, var_b, cov_ab

    mcs = []
    for i in range(levels):
        mu_x, mu_y, var_x, var_y, cov_xy = compute_stats(x, y)

        l = (2 * mu_x * mu_y + C1) / (mu_x**2 + mu_y**2 + C1)

        c = (2 * var_x.sqrt() * var_y.sqrt() + C2) / (var_x + var_y + C2)

        s = (cov_xy + C3) / (var_x.sqrt() * var_y.sqrt() + C3)

        if i < levels - 1:
            mcs.append((c * s).squeeze())
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            y = torch.nn.functional.avg_pool2d(y, kernel_size=2, stride=2)
        else:
            lM = l.squeeze()

    mcs = torch.stack(mcs, dim=0)  # [levels-1, B, C]
    weights_tensor = torch.tensor(weights[:-1], device=x.device).view(-1, 1, 1)
    mcs_prod = (mcs ** weights_tensor).prod(dim=0)

    ms_ssim = (lM ** weights[-1]) * mcs_prod
    return ms_ssim.mean()

## PSNR
def psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(1 / torch.sqrt(mse))