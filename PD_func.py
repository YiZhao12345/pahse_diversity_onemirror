import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from fresnel_utils import (
    two_step_prop_torch,
    plot_wavefront_from_zernike
)

# ============================================================
# 前向模型：波前系数 → PSF（通过 two_step_prop_torch）
# ============================================================

def forward_psf(c, Z_mat, pupil_mask, wavefront_delta,
                wvl, d1, d2, Dz, device):
    """
    给定 Zernike 系数 c，计算归一化 PSF。

    参数:
        c               : (nZ,) float tensor，requires_grad=True
        Z_mat           : (H,W,nZ) float tensor，Zernike 基底
        pupil_mask      : (H,W) float tensor，瞳函数掩模
        wavefront_delta : (H,W) float tensor，已知多样性像差（焦面传0）
        wvl, d1, d2, Dz : 光学参数

    返回:
        PSF_norm : (H,W) real tensor，归一化 PSF
    """
    # 波前 φ = Σ c_j·Z_j + delta
    wavefront = torch.einsum('hwj,j->hw', Z_mat, c) + wavefront_delta  # (H,W)

    # 复振幅瞳函数
    Uin = pupil_mask.to(torch.complex64) * torch.exp(
        1j * 2 * torch.pi * wavefront.to(torch.complex64)
    )

    # 菲涅尔传播
    Uout     = two_step_prop_torch(Uin, wvl, d1, d2, Dz, device)
    PSF      = torch.abs(Uout) ** 2
    PSF_norm = PSF / (PSF.max() + 1e-30)

    return PSF_norm


# ============================================================
# 损失函数：基于 two_step_prop 的相位差代价函数
# ============================================================

def cost_pd(c, Z_mat, pupil_mask,
            PSF_focus_t, PSF_de_t,
            delta_focus, delta_de,
            wvl, d1, d2, Dz, device,
            gamma=1e-6, alpha=0.0):
    """
    J(c) = ||PSF_pred_focus - PSF_focus||² + ||PSF_pred_de - PSF_de||²
           + γ·||c||²

    注：与 zernretrieve_loop 保持同样的正则化结构。
    """
    PSF_pred_focus = forward_psf(c, Z_mat, pupil_mask, delta_focus,
                                  wvl, d1, d2, Dz, device)
    PSF_pred_de    = forward_psf(c, Z_mat, pupil_mask, delta_de,
                                  wvl, d1, d2, Dz, device)

    loss = (torch.sum((PSF_pred_focus - PSF_focus_t) ** 2) +
            torch.sum((PSF_pred_de    - PSF_de_t   ) ** 2))

    if alpha != 0.0:
        loss = loss + alpha * torch.dot(c, c)

    return loss


# ============================================================
# 主优化函数
# ============================================================

def phase_diversity_retrieve(PSF_focus, PSF_de, Z_UDA,
                              defocus_coeff=None,
                              wvl=2e-6, d1=None, d2=2e-6/1e-4,
                              Dz=132.812,
                              gamma=1e-6, alpha=0.0,
                              lr=0.01, max_iter=200,
                              tol=1e-6,
                              optimizer_type='Adam',
                              verbose=True):
    """
    相位差波前复原——two_step_prop 前向模型 + PyTorch 自动微分。

    参数:
        PSF_focus      : (H,W) numpy，焦面归一化 PSF
        PSF_de         : (H,W) numpy，离焦归一化 PSF
        Z_UDA          : list of (H,W) numpy，正交 Zernike 基底
        defocus_coeff  : float，已知离焦 Zernike 系数（第4项系数）
        wvl            : 波长 (m)
        d1             : 源平面采样间距 (m)，默认 6.605a/N
        d2             : 像面采样间距 (m)
        Dz             : 传播距离 (m)
        gamma          : L2 正则化强度
        alpha          : Zernike 系数正则化强度
        lr             : 学习率
        max_iter       : 最大迭代次数
        tol            : 收敛阈值
        optimizer_type : 'Adam' 或 'LBFGS'
        verbose        : 是否打印迭代信息

    返回:
        c_est          : (nZ,) numpy，估计 Zernike 系数
        J_hist         : list，损失历史
        wavefront_est  : (H,W) numpy，重建波前
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    H, W = PSF_focus.shape
    nZ   = len(Z_UDA)

    # 默认采样间距
    if d1 is None:
        a  = 1e-4
        N0 = H
        d1 = 6.605 * a / N0

    def to_t(x, cplx=False):
        dt = torch.complex64 if cplx else torch.float32
        return torch.tensor(x, dtype=dt, device=device)

    # ── 数据张量 ──────────────────────────────────────────────────────────
    PSF_focus_t = to_t(PSF_focus.astype(np.float32))
    PSF_de_t    = to_t(PSF_de.astype(np.float32))
    Z_mat       = to_t(np.stack(Z_UDA, axis=-1))          # (H,W,nZ)
    pupil_mask  = to_t((np.abs(Z_UDA[0]) > 0).astype(np.float32))

    # ── 已知多样性像差 ────────────────────────────────────────────────────
    delta_focus = torch.zeros(H, W, dtype=torch.float32, device=device)

    if defocus_coeff is None:
        defocus_coeff = 1.0
        print("  defocus_coeff 未指定，使用默认值 1.0")
    delta_de = to_t((plot_wavefront_from_zernike(Z_UDA,defocus_coeff)
                     if nZ >= 4 else np.zeros((H, W))).astype(np.float32))

    # ── 初始化参数 ────────────────────────────────────────────────────────
    c = torch.zeros(nZ, dtype=torch.float32,
                    device=device, requires_grad=True)

    # ── 优化器 ────────────────────────────────────────────────────────────
    if optimizer_type == 'LBFGS':
        optimizer = torch.optim.LBFGS([c], lr=lr, max_iter=20,
                                       history_size=10,
                                       line_search_fn='strong_wolfe')
    else:
        optimizer = torch.optim.Adam([c], lr=lr)

    # ── 迭代 ──────────────────────────────────────────────────────────────
    J_hist = []

    def closure():
        optimizer.zero_grad()
        J = cost_pd(c, Z_mat, pupil_mask,
                    PSF_focus_t, PSF_de_t,
                    delta_focus, delta_de,
                    wvl, d1, d2, Dz, device,
                    gamma=gamma, alpha=alpha)
        J.backward()          # ← 自动微分
        return J

    for i in range(max_iter):
        if optimizer_type == 'LBFGS':
            J_val = optimizer.step(closure)
        else:
            J_val = closure()
            optimizer.step()

        J_scalar = J_val.item()
        J_hist.append(J_scalar)

        if verbose and i % 10 == 0:
            grad_norm = c.grad.norm().item() if c.grad is not None else float('nan')
            print(f"  iter {i:4d} | J = {J_scalar:.6e} | ||∇|| = {grad_norm:.4e}")

        # 收敛
        if len(J_hist) >= 3:
            dJ = abs(J_hist[-2] - J_hist[-1]) / (abs(J_hist[0] - J_hist[-1]) + 1e-30)
            if dJ < tol:
                print(f"  收敛于第 {i} 次迭代（ΔJ/J={dJ:.2e}）")
                break

        # 发散
        if len(J_hist) >= 3:
            if J_hist[-3] < J_hist[-1] and J_hist[-2] < J_hist[-1]:
                print(f"  发散于第 {i} 次迭代，停止")
                break

    # ── 结果 ──────────────────────────────────────────────────────────────
    c_est         = c.detach().cpu().numpy()
    Z_mat_np      = np.stack(Z_UDA, axis=-1)
    pupil_mask_np = (np.abs(Z_UDA[0]) > 0).astype(float)
    wavefront_est = np.einsum('hwj,j->hw', Z_mat_np, c_est) * pupil_mask_np

    return c_est, J_hist, wavefront_est


# ============================================================
# 画图函数（前向模型也用 two_step_prop_torch）
# ============================================================

def plot_phase_diversity_result(PSF_focus, PSF_de,
                                 wavefront_est, c_est,
                                 J_hist, Z_UDA,
                                 wvl=2e-6, d1=None, d2=2e-6/1e-4, Dz=132.812,
                                 defocus_coeff=1.0,
                                 wavefront_true=None,
                                 c_true=None,
                                 save_path="image/phase_diversity_result.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W   = PSF_focus.shape
    nZ     = len(c_est)

    if d1 is None:
        a  = 1e-4
        d1 = 6.605 * a / H

    # ── 用估计系数重建两张 PSF ────────────────────────────────────────────
    Z_mat      = torch.tensor(np.stack(Z_UDA, axis=-1),
                               dtype=torch.float32, device=device)
    pupil_mask = torch.tensor((np.abs(Z_UDA[0]) > 0).astype(np.float32),
                               device=device)
    c_t        = torch.tensor(c_est, dtype=torch.float32, device=device)

    delta_focus = torch.zeros(H, W, dtype=torch.float32, device=device)
    delta_de    = torch.tensor(
        (plot_wavefront_from_zernike(Z_UDA,defocus_coeff) if nZ >= 4 else np.zeros((H, W))).astype(np.float32),
        device=device)

    with torch.no_grad():
        PSF_pred_focus = forward_psf(c_t, Z_mat, pupil_mask, delta_focus,
                                      wvl, d1, d2, Dz, device).cpu().numpy()
        PSF_pred_de    = forward_psf(c_t, Z_mat, pupil_mask, delta_de,
                                      wvl, d1, d2, Dz, device).cpu().numpy()

    # ── 差值 ──────────────────────────────────────────────────────────────
    has_true_wf = wavefront_true is not None
    has_true_c  = c_true is not None
    if not has_true_wf: wavefront_true = np.zeros_like(wavefront_est)
    if not has_true_c:  c_true = np.zeros(nZ)

    pupil_np = (np.abs(Z_UDA[0]) > 0).astype(float)
    wf_diff  = wavefront_true - wavefront_est
    c_diff   = np.array(c_true)[:nZ] - c_est
    psf_focus_diff = PSF_focus  - PSF_pred_focus
    psf_de_diff    = PSF_de     - PSF_pred_de

    # ── 色条统一范围 ──────────────────────────────────────────────────────
    wf_abs  = max(np.abs(wavefront_true).max(), np.abs(wavefront_est).max()) + 1e-10
    psf_max = max(PSF_focus.max(), PSF_pred_focus.max()) + 1e-10

    # ============================================================
    # 图1：波前（3列）+ Zernike 系数（3列）
    # ============================================================
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 9))

    # 波前行
    titles_wf = ["原始波前", "预测波前", f"差值（RMS={np.std(wf_diff[pupil_np>0]):.4f}）"]
    datas_wf  = [wavefront_true, wavefront_est, wf_diff]
    vmaxs_wf  = [wf_abs, wf_abs, np.abs(wf_diff).max()+1e-10]
    for col, (title, data, vm) in enumerate(zip(titles_wf, datas_wf, vmaxs_wf)):
        im = axes1[0, col].imshow(data, origin='lower', cmap='RdBu_r',
                                   vmin=-vm, vmax=vm, aspect='auto')
        plt.colorbar(im, ax=axes1[0, col])
        axes1[0, col].set_title(title)

    # Zernike 系数行
    x_idx = np.arange(1, nZ + 1)
    colors = ['steelblue', 'darkorange', 'firebrick']
    labels = ['原始 Zernike 系数', '预测 Zernike 系数',
              f'差值（RMS={np.std(c_diff):.4f}）']
    datas_c = [c_true[:nZ], c_est, c_diff]
    for col, (label, data, color) in enumerate(zip(labels, datas_c, colors)):
        axes1[1, col].bar(x_idx, data, alpha=0.8, color=color)
        axes1[1, col].axhline(0, color='k', linewidth=0.5)
        axes1[1, col].set_title(label)
        axes1[1, col].set_xlabel("Zernike 索引")
        axes1[1, col].set_ylabel("系数值")

    fig1.suptitle("波前与 Zernike 系数对比", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path1 = save_path.replace(".png", "_wavefront.png")
    plt.savefig(path1, dpi=150)
    plt.show()

    # ============================================================
    # 图2：焦面PSF（3列）+ 离焦PSF（3列）
    # ============================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))

    rows = [
        ("焦面 PSF",
         PSF_focus, PSF_pred_focus, psf_focus_diff),
        ("离焦 PSF",
         PSF_de,    PSF_pred_de,    psf_de_diff),
    ]
    for row_idx, (row_title, orig, pred, diff) in enumerate(rows):
        diff_vm = np.abs(diff).max() + 1e-10
        for col_idx, (data, title, vm, cmap) in enumerate([
            (orig, f"{row_title}（原始）",  psf_max,  'hot'),
            (pred, f"{row_title}（预测）",  psf_max,  'hot'),
            (diff, f"{row_title}（差值）",  diff_vm,  'RdBu_r'),
        ]):
            vmin = 0 if cmap == 'hot' else -vm
            im = axes2[row_idx, col_idx].imshow(data, origin='lower',
                                                 cmap=cmap, vmin=vmin, vmax=vm,
                                                 aspect='auto')
            plt.colorbar(im, ax=axes2[row_idx, col_idx])
            axes2[row_idx, col_idx].set_title(title)

    fig2.suptitle("PSF 对比（two_step_prop 前向模型）",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    path2 = save_path.replace(".png", "_PSF.png")
    plt.savefig(path2, dpi=150)
    plt.show()

    # ============================================================
    # 图3：损失曲线 + 统计
    # ============================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))

    axes3[0].semilogy(J_hist, 'b-', linewidth=1.5)
    axes3[0].set_xlabel("迭代次数"); axes3[0].set_ylabel("J (log)")
    axes3[0].set_title("损失函数收敛曲线"); axes3[0].grid(True, alpha=0.3)

    axes3[1].axis('off')
    info = (f"波前 RMS（原始）  : {np.std(wavefront_true[pupil_np>0]):.4f}\n"
            f"波前 RMS（预测）  : {np.std(wavefront_est[pupil_np>0]):.4f}\n"
            f"波前 RMS（差值）  : {np.std(wf_diff[pupil_np>0]):.4f}\n\n"
            f"Zernike RMS（差值）: {np.std(c_diff):.4f}\n"
            f"迭代次数          : {len(J_hist)}\n"
            f"最终损失 J        : {J_hist[-1]:.4e}")
    axes3[1].text(0.05, 0.5, info, fontsize=12,
                  transform=axes3[1].transAxes, verticalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes3[1].set_title("结果统计")

    plt.tight_layout()
    path3 = save_path.replace(".png", "_loss.png")
    plt.savefig(path3, dpi=150)
    plt.show()

    print(f"结果已保存:\n  {path1}\n  {path2}\n  {path3}")