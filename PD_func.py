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

def forward_otf(c, Z_mat, pupil_mask, wavefront_delta,
                wvl, d1, d2, Dz, device):
    """
    给定 Zernike 系数 c，计算归一化 PSF。

    参数:
        c               : (nZ,) float tensor
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
    PSF_norm = PSF / (PSF.sum() + 1e-30)   #归一化
    OTF = torch.fft.fft2(PSF)  # 新增：PSF → OTF
    return OTF


# ============================================================
# 损失函数：支持多张离焦图片,不在仅限于PSF图像
# ============================================================

def cost_pd_image(c, Z_mat, pupil_mask,
                  imgDs,                      # (H,W,K) 观测图像的FFT
                  wavefront_deltas,           # (H,W,K) 已知多样性像差
                  wvl, d1, d2, Dz, device,
                  gamma=1e-6, alpha=0.0, Rc=None):
    """
    基于 zernretrieve_loop 的损失函数，适用于任意场景图像输入。

    J(c) = Σ_u [ Σ_k|Dk|² - |Σ_k Dk*·Sk|² / Q ] + α·cᵀRc·c
    Q(u) = Σ_k|Sk|² + γ

    imgDs  : (H,W,K) complex，K张观测图像的傅里叶变换
             输入为实际拍摄图像时：imgDs[:,:,k] = fft2(image_k)
    """
    K = imgDs.shape[2]

    # 计算每张图对应的 OTF
    Sks = torch.stack([
        forward_otf(c, Z_mat, pupil_mask,
                    wavefront_deltas[:, :, k],
                    wvl, d1, d2, Dz, device)
        for k in range(K)
    ], dim=2)                                 # (H,W,K)

    # Q = Σ_k|Sk|² + γ
    Q = torch.sum(torch.abs(Sks) ** 2, dim=2) + gamma   # (H,W)

    # 分子：Σ_k Dk* · Sk
    numer = torch.sum(torch.conj(imgDs) * Sks, dim=2)   # (H,W)

    # 图像项
    imgDs_sq = torch.sum(torch.abs(imgDs) ** 2, dim=2)  # (H,W)
    J = torch.sum(imgDs_sq - torch.abs(numer) ** 2 / Q).real

    # 相位正则项
    if alpha != 0.0 and Rc is not None:
        J = J + alpha * (c @ Rc @ c)

    return J

# ============================================================
# 主优化函数：支持多张离焦图片输入
# ============================================================

def phase_diversity_retrieve(image_focus, image_de_list, Z_UDA,
                              coff_div_list=None,
                              c0=None,
                              wvl=2e-6, d1=None, d2=2e-6/1e-4,
                              Dz=132.812,
                              gamma=1e-6, alpha=0.0,
                              lr=0.01, max_iter=200,
                              tol=1e-6,
                              optimizer_type='Adam',
                              verbose=True):
    """
    相位差波前复原——支持多张离焦图片。

    参数:
        PSF_focus      : (H,W) numpy，焦面归一化 PSF
        PSF_de_list    : list of (H,W) numpy，多张离焦归一化 PSF
                         （单张时也可传 list，如 [PSF_de]）
        Z_UDA          : list of (H,W) numpy，正交 Zernike 基底（共 nZ 项）
        coff_div_list  : list of (nZ,) numpy，每张离焦图对应的已知 Zernike 系数
                         长度需与 PSF_de_list 一致
        c0             : (nZ,) numpy，初始 Zernike 系数（默认全零）
        wvl            : 波长 (m)
        d1             : 源平面采样间距 (m)
        d2             : 像面采样间距 (m)
        Dz             : 传播距离 (m)
        gamma          : Zernike 系数 L2 正则化强度
        alpha          : 额外相位正则化强度（通常=0）
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
    # 单张输入兼容处理
    if isinstance(image_de_list, np.ndarray):
        image_de_list = [image_de_list]
    if coff_div_list is not None and isinstance(coff_div_list, np.ndarray):
        coff_div_list = [coff_div_list]

    n_de = len(image_de_list)
    assert coff_div_list is None or len(coff_div_list) == n_de, \
        f"coff_div_list 长度({len(coff_div_list)}) 需与 PSF_de_list 长度({n_de}) 一致"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}，离焦图片数量: {n_de}")

    H, W = image_focus.shape
    nZ   = len(Z_UDA)

    if d1 is None:
        a  = 1e-4
        d1 = 6.605 * a / H

    def to_t(x):
        return torch.tensor(np.array(x).astype(np.float32),
                            dtype=torch.float32, device=device)

    # ── 图像 → 频域（关键：用实际图像替代PSF）────────────────────────────
    def img_to_fft(img):
        t = to_t(img)
        return torch.fft.fft2(t).to(torch.complex64)

    I_focus = img_to_fft(image_focus)
    I_de_list = [img_to_fft(im) for im in image_de_list]

    # imgDs : (H,W,K)，K = 1(焦面) + n_de(离焦)
    imgDs = torch.stack([I_focus] + I_de_list, dim=2)  # (H,W,1+n_de)

    # ── Zernike 基底 / 瞳函数 ────────────────────────────────────────────
    Z_mat = to_t(np.stack(Z_UDA, axis=-1))
    pupil_mask = to_t((np.abs(Z_UDA[0]) > 0).astype(np.float32))

    # ── 焦面多样性像差（零）──────────────────────────────────────────────
    delta_focus = torch.zeros(H, W, dtype=torch.float32, device=device)

    # ── 多样性像差 wavefront_deltas (H,W,K) ──────────────────────────────
    delta_focus = torch.zeros(H, W, dtype=torch.float32, device=device)
    delta_list = [delta_focus]  # 第0张：焦面，delta=0

    for k in range(n_de):
        if coff_div_list is not None:
            wf_de = plot_wavefront_from_zernike(Z_UDA, coff_div_list[k])
            delta_list.append(to_t(wf_de.astype(np.float32)))
        else:
            delta_list.append(torch.zeros(H, W, dtype=torch.float32, device=device))

    wavefront_deltas = torch.stack(delta_list, dim=2)  # (H,W,K)

    # ── 初始化参数 ────────────────────────────────────────────────────────
    if c0 is not None:
        c_init = torch.tensor(c0[:nZ].astype(np.float32), device=device)
    else:
        c_init = torch.zeros(nZ, dtype=torch.float32, device=device)
    c = c_init.clone().detach().requires_grad_(True)

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
        J = cost_pd_image(c, Z_mat, pupil_mask,
                          imgDs, wavefront_deltas,
                          wvl, d1, d2, Dz, device,
                          gamma=gamma, alpha=alpha)
        J.backward()
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
# 画图函数：支持多张离焦 PSF 对比
# ============================================================

def plot_phase_diversity_result(PSF_focus, PSF_de_list,
                                 wavefront_est, c_est,
                                 J_hist, Z_UDA,
                                 coff_div_list=None,
                                 wvl=2e-6, d1=None, d2=2e-6/1e-4, Dz=132.812,
                                 wavefront_true=None,
                                 c_true=None,
                                 save_path="image/phase_diversity_result.png"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 单张兼容
    if isinstance(PSF_de_list, np.ndarray):
        PSF_de_list = [PSF_de_list]
    if coff_div_list is not None and isinstance(coff_div_list, np.ndarray):
        coff_div_list = [coff_div_list]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W   = PSF_focus.shape
    nZ     = len(c_est)
    n_de   = len(PSF_de_list)

    if d1 is None:
        a  = 1e-4
        d1 = 6.605 * a / H

    def to_t(x):
        return torch.tensor(np.array(x).astype(np.float32),
                            dtype=torch.float32, device=device)

    Z_mat      = to_t(np.stack(Z_UDA, axis=-1))
    pupil_mask = to_t((np.abs(Z_UDA[0]) > 0).astype(np.float32))
    c_t        = to_t(c_est)

    # ── 重建焦面 PSF ──────────────────────────────────────────────────────
    delta_focus = torch.zeros(H, W, dtype=torch.float32, device=device)
    with torch.no_grad():
        OTF_pred_focus = forward_otf(c_t, Z_mat, pupil_mask, delta_focus,
                                      wvl, d1, d2, Dz, device).cpu().numpy()
        PSF_pred_focus = torch.abs(torch.fft.ifft2(OTF_pred_focus)).cpu().numpy()
        PSF_pred_focus=PSF_pred_focus/PSF_pred_focus.max()

    # ── 重建各张离焦 PSF ──────────────────────────────────────────────────
    PSF_pred_de_list = []
    for k in range(n_de):
        if coff_div_list is not None:
            wf_de    = plot_wavefront_from_zernike(Z_UDA, coff_div_list[k])
            delta_de = to_t(wf_de.astype(np.float32))
        else:
            delta_de = torch.zeros(H, W, dtype=torch.float32, device=device)
        with torch.no_grad():
            OTF_pred_de = forward_otf(c_t, Z_mat, pupil_mask, delta_de,
                                    wvl, d1, d2, Dz, device).cpu().numpy()
            PSF_pred_de = torch.abs(torch.fft.ifft2(OTF_pred_de)).cpu().numpy()
            psf_pred_de=psf_pred_de/psf_pred_de.max()
        PSF_pred_de_list.append(psf_pred_de)

    # ── 波前 / 系数差值 ───────────────────────────────────────────────────
    pupil_np = (np.abs(Z_UDA[0]) > 0).astype(float)
    if wavefront_true is None: wavefront_true = np.zeros_like(wavefront_est)
    if c_true is None:         c_true = np.zeros(nZ)

    wf_diff = wavefront_true - wavefront_est
    c_diff  = np.array(c_true)[:nZ] - c_est
    wf_abs  = max(np.abs(wavefront_true).max(), np.abs(wavefront_est).max()) + 1e-10

    # ============================================================
    # 图1：波前 + Zernike 系数
    # ============================================================
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 9))

    titles_wf = ["原始波前", "预测波前",
                 f"差值（RMS={np.std(wf_diff[pupil_np>0]):.4f}）"]
    datas_wf  = [wavefront_true, wavefront_est, wf_diff]
    vmaxs_wf  = [wf_abs, wf_abs, np.abs(wf_diff).max()+1e-10]
    for col, (title, data, vm) in enumerate(zip(titles_wf, datas_wf, vmaxs_wf)):
        im = axes1[0, col].imshow(data, origin='lower', cmap='RdBu_r',
                                   vmin=-vm, vmax=vm, aspect='auto')
        plt.colorbar(im, ax=axes1[0, col])
        axes1[0, col].set_title(title)

    x_idx  = np.arange(1, nZ + 1)
    colors = ['steelblue', 'darkorange', 'firebrick']
    labels = ['原始 Zernike 系数', '预测 Zernike 系数',
              f'差值（RMS={np.std(c_diff):.4f}）']
    datas_c = [np.array(c_true)[:nZ], c_est, c_diff]
    for col, (label, data, color) in enumerate(zip(labels, datas_c, colors)):
        axes1[1, col].bar(x_idx, data, alpha=0.8, color=color)
        axes1[1, col].axhline(0, color='k', linewidth=0.5)
        axes1[1, col].set_title(label)
        axes1[1, col].set_xlabel("Zernike 索引")
        axes1[1, col].set_ylabel("系数值")

    fig1.suptitle("波前与 Zernike 系数对比", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path1 = save_path.replace(".png", "_wavefront.png")
    plt.savefig(path1, dpi=150); plt.show()

    # ============================================================
    # 图2：焦面PSF（1行）+ 各离焦PSF（每张1行），各3列（原始/预测/差值）
    # ============================================================
    n_rows  = 1 + n_de
    fig2, axes2 = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes2 = axes2[np.newaxis, :]   # 保证二维索引

    psf_max = max(PSF_focus.max(), PSF_pred_focus.max()) + 1e-10

    def _plot_psf_row(ax_row, orig, pred, row_title, psf_vmax):
        diff     = orig - pred
        diff_vm  = np.abs(diff).max() + 1e-10
        for col_idx, (data, title, vm, cmap) in enumerate([
            (orig, f"{row_title}（原始）", psf_vmax, 'hot'),
            (pred, f"{row_title}（预测）", psf_vmax, 'hot'),
            (diff, f"{row_title}（差值）", diff_vm,  'RdBu_r'),
        ]):
            vmin = 0 if cmap == 'hot' else -vm
            im = ax_row[col_idx].imshow(data, origin='lower', cmap=cmap,
                                         vmin=vmin, vmax=vm, aspect='auto')
            plt.colorbar(im, ax=ax_row[col_idx])
            ax_row[col_idx].set_title(title)

    # 焦面行
    _plot_psf_row(axes2[0], PSF_focus, PSF_pred_focus, "焦面 PSF", psf_max)

    # 各离焦行
    for k in range(n_de):
        psf_vmax_k = max(PSF_de_list[k].max(), PSF_pred_de_list[k].max()) + 1e-10
        _plot_psf_row(axes2[k+1], PSF_de_list[k], PSF_pred_de_list[k],
                      f"离焦 PSF #{k+1}", psf_vmax_k)

    fig2.suptitle("PSF 对比（two_step_prop 前向模型）",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    path2 = save_path.replace(".png", "_PSF.png")
    plt.savefig(path2, dpi=150); plt.show()

    # ============================================================
    # 图3：损失曲线 + 统计
    # ============================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))

    axes3[0].semilogy(J_hist, 'b-', linewidth=1.5)
    axes3[0].set_xlabel("迭代次数"); axes3[0].set_ylabel("J (log)")
    axes3[0].set_title("损失函数收敛曲线"); axes3[0].grid(True, alpha=0.3)

    axes3[1].axis('off')
    info = (f"离焦图片数量      : {n_de}\n"
            f"Zernike 项数      : {nZ}\n\n"
            f"波前 RMS（原始）  : {np.std(wavefront_true[pupil_np>0]):.4f}\n"
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
    plt.savefig(path3, dpi=150); plt.show()

    print(f"结果已保存:\n  {path1}\n  {path2}\n  {path3}")


def compute_hessian(cost_func, c_base, eps=1e-3):
    """
    数值计算 Hessian 矩阵，非对角元素代表变量间的耦合强度。

    H[i,j] ≈ ∂²J / ∂ci∂cj

    |H[i,j]| 大 → i,j 之间耦合强
    |H[i,j]| ≈ 0 → i,j 之间独立
    """
    nZ = len(c_base)
    H = np.zeros((nZ, nZ))
    J0 = cost_func(c_base)

    for i in range(nZ):
        for j in range(i, nZ):
            c_pp = c_base.copy();c_pp[i] += eps; c_pp[j] += eps
            c_pm = c_base.copy();c_pm[i] += eps;c_pm[j] -= eps
            c_mp = c_base.copy();c_mp[i] -= eps;c_mp[j] += eps
            c_mm = c_base.copy();c_mm[i] -= eps;c_mm[j] -= eps
            H[i, j] = (cost_func(c_pp) - cost_func(c_pm)
                       - cost_func(c_mp) + cost_func(c_mm)) / (4 * eps ** 2)
            H[j, i] = H[i, j]  # 对称

    # 归一化：相关矩阵
    diag = np.sqrt(np.diag(H))
    denom = np.outer(diag, diag) + 1e-30
    corr = H / denom  # 类似相关系数，[-1,1]

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(H, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title("Hessian 矩阵（∂²J/∂ci∂cj）")
    axes[0].set_xlabel("Zernike 索引");axes[0].set_ylabel("Zernike 索引")

    im1 = axes[1].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title("归一化耦合矩阵\n|值|>0.3 表示存在显著耦合")
    axes[1].set_xlabel("Zernike 索引");axes[1].set_ylabel("Zernike 索引")

    plt.tight_layout()
    plt.savefig("image_test3/hessian_coupling_OTFloss.png", dpi=150)
    plt.show()

    # 找出耦合强的变量对
    threshold = 0.3
    print("\n显著耦合的变量对（|corr| > 0.3）：")
    found = False
    for i in range(nZ):
        for j in range(i + 1, nZ):
            if abs(corr[i, j]) > threshold:
                print(f"  Z{i + 1} ↔ Z{j + 1}: corr = {corr[i, j]:+.3f}")
                found = True
    if not found:
        print("  ✅ 未发现显著耦合")

    return H, corr

def sensitivity_scan(image_focus, image_de_list, Z_UDA,
                     coff_div_list, F0,
                     wvl=2e-6, d1=None, d2=2e-6 / 1e-4, Dz=132.812,
                     gamma=1e-6, alpha=0.0,
                     scan_range=0.5, scan_steps=21):
    """
    对每个 Zernike 系数单独扫描损失函数，判断 F0 是否在最小值附近。

    参数:
        image_focus  : (H,W) numpy，焦面图像（任意场景）
        image_de_list: list of (H,W) numpy，多张离焦图像
        F0           : (nZ,) 基准 Zernike 系数（待验证的点）
        scan_range   : 每个系数的扫描范围 ± scan_range
        scan_steps   : 每个系数的扫描点数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    H, W = image_focus.shape
    nZ = len(F0)

    if d1 is None:
        a = 1e-4
        d1 = 6.605 * a / H

    def to_t(x):
        return torch.tensor(np.array(x).astype(np.float32),
                            dtype=torch.float32, device=device)

    # ── 数据预处理：图像 → FFT，合并为 imgDs (H,W,K) ─────────────────────
    I_focus   = torch.fft.fft2(to_t(image_focus)).to(torch.complex64)
    I_de_list = [torch.fft.fft2(to_t(im)).to(torch.complex64)
                 for im in image_de_list]
    imgDs     = torch.stack([I_focus] + I_de_list, dim=2)   # (H,W,K)

    # ── Zernike 基底 / 瞳函数 ────────────────────────────────────────────
    Z_mat      = to_t(np.stack(Z_UDA, axis=-1))
    pupil_mask = to_t((np.abs(Z_UDA[0]) > 0).astype(np.float32))

    # ── 多样性像差：合并为 wavefront_deltas (H,W,K) ───────────────────────
    delta_focus = torch.zeros(H, W, dtype=torch.float32, device=device)
    delta_de_list = []
    for coff in coff_div_list:
        wf_de = plot_wavefront_from_zernike(Z_UDA, coff)
        delta_de_list.append(to_t(wf_de.astype(np.float32)))
    wavefront_deltas = torch.stack([delta_focus] + delta_de_list, dim=2)  # (H,W,K)

    # ── 扫描轴 ────────────────────────────────────────────────────────────
    scan_vals = np.linspace(-scan_range, scan_range, scan_steps)

    # ── 计算 F0 处的基准损失 ──────────────────────────────────────────────
    with torch.no_grad():
        c_base = to_t(F0)
        J_base = cost_pd_image(c_base, Z_mat, pupil_mask,
                               imgDs, wavefront_deltas,
                               wvl, d1, d2, Dz, device,
                               gamma=gamma, alpha=alpha).item()
    print(f"F0 处基准损失 J = {J_base:.6e}")

    # ── 逐项扫描 ──────────────────────────────────────────────────────────
    J_curves = np.zeros((nZ, scan_steps))

    for j in range(nZ):
        for k, dv in enumerate(scan_vals):
            c_scan = F0.copy()
            c_scan[j] += dv                  # 只扰动第 j 项
            with torch.no_grad():
                c_t = to_t(c_scan)
                J_curves[j, k] = cost_pd_image(
                    c_t, Z_mat, pupil_mask,
                    imgDs, wavefront_deltas,
                    wvl, d1, d2, Dz, device,
                    gamma=gamma, alpha=alpha
                ).item()

        min_idx = np.argmin(J_curves[j])
        print(f"  Z{j + 1:02d}: 最小值在偏移 {scan_vals[min_idx]:+.3f}，"
              f"J_min={J_curves[j, min_idx]:.4e}，"
              f"J_base={J_curves[j, scan_steps // 2]:.4e}")

    # ── 画图 ──────────────────────────────────────────────────────────────
    cols = 5
    rows = int(np.ceil(nZ / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    axes = axes.flatten()

    for j in range(nZ):
        ax = axes[j]
        ax.plot(scan_vals, J_curves[j], 'b-', linewidth=1.5)
        ax.axvline(0, color='r', linestyle='--', linewidth=1, label='F0')
        ax.axvline(scan_vals[np.argmin(J_curves[j])],
                   color='g', linestyle=':', linewidth=1.5, label='J_min')
        ax.set_title(f"Z{j + 1}")
        ax.set_xlabel("Δc")
        ax.set_ylabel("J")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(nZ, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("灵敏度扫描：各 Zernike 系数对损失函数的影响\n"
                 "红线=F0位置，绿线=最小值位置", fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs("image_test3", exist_ok=True)
    plt.savefig("image_test3/sensitivity_scan_OTFloss.png", dpi=150)
    plt.show()
    print("灵敏度扫描图已保存: image_test3/sensitivity_scan_OTFloss.png")

    return J_curves, scan_vals