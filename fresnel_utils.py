"""
fresnel_utils.py
菲涅尔衍射传播工具函数库
对应原 MATLAB 自定义函数的 Python 版本
"""

import numpy as np
from scipy.ndimage import zoom
import torch
from scipy.io import loadmat
# ============================================================
# 基础傅里叶变换工具
# ============================================================
def ft2(g: np.ndarray, delta: float) -> np.ndarray:
    """二维离散傅里叶变换（含移位和物理缩放）"""
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g))) * delta ** 2
def ift2(G: np.ndarray, delta_f: float) -> np.ndarray:
    """二维离散逆傅里叶变换（含移位和物理缩放）"""
    N = G.shape[0]
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(G))) * (N * delta_f) ** 2
def myconv2(A: np.ndarray, B: np.ndarray, delta: float) -> np.ndarray:
    """利用 FFT 实现的二维卷积"""
    N = A.shape[0]
    return ift2(ft2(A, delta) * ft2(B, delta), 1.0 / (N * delta))
# ============================================================
# 零填充
# ============================================================
def zero_padding_complex(matrix: np.ndarray, padding_factor: float) -> np.ndarray:
    """
    对复数（或实数）矩阵进行零填充，居中放置。
    padding_factor=1 时返回与输入等大的矩阵（无填充）。
    """
    M, N = matrix.shape

    M_new = int(round(M * padding_factor))
    N_new = int(round(N * padding_factor))
    # 保证偶数尺寸
    M_new += M_new % 2
    N_new += N_new % 2

    padded = np.zeros((M_new, N_new), dtype=complex)

    M_start = (M_new - M) // 2
    N_start = (N_new - N) // 2

    padded[M_start:M_start + M, N_start:N_start + N] = matrix
    return padded
# ============================================================
# 混叠检查
# ============================================================
def check_aliasing_condition(Uin: np.ndarray, wvl: float,
                              d1: float, d2: float, Dz: float) -> bool:
    """
    检查两步菲涅尔传播是否满足无混叠条件。
    返回 True 表示不会发生混叠。
    """
    N = Uin.shape[0]
    m = d2 / d1
    Dz1 = Dz / (1.0 - m)
    d1a = wvl * abs(Dz1) / (N * d1)
    Dz2 = Dz - Dz1

    # 条件 1：第一步无混叠
    f_max1 = (N * d1 / 2.0) / (wvl * abs(Dz1))
    cond1 = (1.0 / d1) >= 2.0 * f_max1

    # 条件 2：第二步无混叠
    f_max2 = (N * d1a / 2.0) / (wvl * abs(Dz2))
    cond2 = (1.0 / d1a) >= 2.0 * f_max2

    if not cond1:
        print(f"[警告] 第一步可能混叠！")
        print(f"  当前 d1²={d1**2:.2e}，允许最大值={(wvl*abs(Dz1))/N:.2e}")
        print(f"  建议：N ≥ {int(np.ceil(wvl*abs(Dz1)/d1**2))} 或减小 d1")
    if not cond2:
        print(f"[警告] 第二步可能混叠！")
        print(f"  当前 d1a²={d1a**2:.2e}，允许最大值={(wvl*abs(Dz2))/N:.2e}")
        print(f"  建议：调整参数使 Dz2 增大或使用更大的 N")
    if cond1 and cond2:
        print("两步传播不会发生混叠 ✅")

    return cond1 and cond2
# ============================================================
# 菲涅尔传播（一步法）
# ============================================================
def fresnel_prop(Uin: np.ndarray, wvl: float, d1: float, Dz: float):
    """
    一步菲涅尔衍射传播（固定像距）。

    返回
    ----
    x2, y2 : 观测面坐标网格 (m)
    Uout   : 观测面复振幅
    """
    N = Uin.shape[0]
    k = 2 * np.pi / wvl

    coords1 = (np.arange(-N // 2, N // 2)) * d1
    x1, y1 = np.meshgrid(coords1, coords1)

    fx = np.arange(-N // 2, N // 2) / (N * d1)
    x2, y2 = np.meshgrid(fx * wvl * Dz, fx * wvl * Dz)

    Uout = (1.0 / (1j * wvl * Dz)
            * np.exp(1j * k / (2 * Dz) * (x2 ** 2 + y2 ** 2))
            * ft2(Uin * np.exp(1j * k / (2 * Dz) * (x1 ** 2 + y1 ** 2)), d1))
    return x2, y2, Uout
# ============================================================
# 菲涅尔传播（两步法）
# ============================================================
def two_step_prop(Uin: np.ndarray, wvl: float,
                  d1: float, d2: float, Dz: float):
    """
    两步菲涅尔衍射传播（任意像面间距）。

    参数
    ----
    Uin : 源平面复振幅
    wvl : 波长 (m)
    d1  : 源平面采样间距 (m)
    d2  : 观测面采样间距 (m)
    Dz  : 传播距离 (m)，可为负（反向传播）

    返回
    ----
    x2_1d, y2_1d : 观测面 1-D 坐标向量 (m)
    Uout         : 观测面复振幅
    """
    N = Uin.shape[0]
    k = 2 * np.pi / wvl

    coords1 = (np.arange(-N // 2, N // 2)) * d1
    x1, y1 = np.meshgrid(coords1, coords1)

    m = d2 / d1
    Dz1 = Dz / (1.0 - m)                        # 第一步传播距离
    d1a = wvl * abs(Dz1) / (N * d1)              # 中间面采样间距

    coords1a = (np.arange(-N // 2, N // 2)) * d1a
    x1a, y1a = np.meshgrid(coords1a, coords1a)

    # 第一步
    Uitm = (1.0 / (1j * wvl * Dz1)
            * np.exp(1j * k / (2 * Dz1) * (x1a ** 2 + y1a ** 2))
            * ft2(Uin * np.exp(1j * k / (2 * Dz1) * (x1 ** 2 + y1 ** 2)), d1))

    Dz2 = Dz - Dz1                               # 第二步传播距离

    coords2 = (np.arange(-N // 2, N // 2)) * d2
    x2, y2 = np.meshgrid(coords2, coords2)

    # 第二步
    Uout = (1.0 / (1j * wvl * Dz2)
            * np.exp(1j * k / (2 * Dz2) * (x2 ** 2 + y2 ** 2))
            * ft2(Uitm * np.exp(1j * k / (2 * Dz2) * (x1a ** 2 + y1a ** 2)), d1a))

    return x2[0, :], y2[:, 0], Uout
_zernike_fringe = None  # 确保这行在文件顶层（函数外）
# ============================================================
# 导入zernike系数顺序
# ============================================================
def _load_zernike_fringe(mat_path):
    global _zernike_fringe
    if _zernike_fringe is None:
        data = loadmat(mat_path)
        _zernike_fringe = data["Zernikefringe"].astype(int)
    return _zernike_fringe
# ============================================================
# 生成zernike基底
# ============================================================
def zernike(J, bound, res, R, x0, y0, mat_path):
    """
    在 [-bound, bound]^2 网格中生成第 J 个 Zernike 模式（Fringe 编号）。

    参数:
        J        : Zernike 项序号（从1开始，对应 Zernikefringe.mat 的行）
        bound    : 网格范围 [-bound, bound]
        res      : 采样点数
        R        : 圆半径（物理单位，用于归一化 rho）
        x0, y0   : 圆心偏移
        mat_path : Zernikefringe.mat 文件路径

    返回:
        Znm : (res, res) ndarray，第 J 个 Zernike 多项式
    """
    # ── 网格构建 ──────────────────────────────────────────────────────────
    x = np.linspace(-bound, bound, res)
    y = np.linspace(-bound, bound, res)
    x_shift = x - x0
    y_shift = y - y0
    x_grid, y_grid = np.meshgrid(x_shift, y_shift)

    # 笛卡尔 → 极坐标（对应 MATLAB cart2pol）
    theta = np.arctan2(y_grid, x_grid)
    rho   = np.sqrt(x_grid**2 + y_grid**2)

    # rho 归一化到单位圆
    rho_unit = rho / R

    # ── 读取 (n, m) ───────────────────────────────────────────────────────
    fringe = _load_zernike_fringe(mat_path)
    n = fringe[J - 1, 0]   # MATLAB 从1开始，Python 补偿 -1
    m = fringe[J - 1, 1]

    # ── 径向多项式 R_n^|m|(rho) ───────────────────────────────────────────
    from math import factorial
    Rnm = np.zeros_like(rho_unit)
    for s in range((n - abs(m)) // 2 + 1):
        num =  (-1)**s * factorial(n - s)
        den = (factorial(s) *
               factorial((n + abs(m)) // 2 - s) *
               factorial((n - abs(m)) // 2 - s))
        Rnm += (num / den) * rho_unit**(n - 2 * s)

    # ── 角向部分 ──────────────────────────────────────────────────────────
    if m < 0:
        Znm = Rnm * np.sin(abs(m) * theta)
    else:
        Znm = Rnm * np.cos(m * theta)

    # ── 掩膜：圆外置零 ────────────────────────────────────────────────────
    Znm[rho > R] = 0.0

    return Znm
# ============================================================
# schmidt正交化
# ============================================================
def gram_schmidt_mask(N, bound, res, R, x0, y0, mask,mat_path):
    """
    在掩膜区域上执行 Gram-Schmidt 正交化。

    参数:
        N           : 需要正交化的项数
        bound       : 区域范围 [-bound, bound]
        res         : 采样点数
        R           : 圆半径
        x0, y0      : 圆心位置
        mask        : 掩膜 (res, res)

    返回:
        ortho_bases : list of ndarray，正交化后的基函数列表
    """
    # 计算面积微元
    x = np.linspace(-bound, bound, res)
    dx = x[1] - x[0]
    dA = dx * dx  # dx == dy（均匀网格）

    # 计算标准 Zernike 多项式并应用掩膜
    Z_standard = []
    for j in range(1, N + 1):
        z = zernike(j, bound, res, R, x0, y0,mat_path)
        Z_standard.append(z * mask)

    # Gram-Schmidt 正交化
    ortho_bases = []
    for i in range(N):
        current_z = Z_standard[i].copy()

        # 减去在所有已有正交基上的投影
        for j in range(i):
            num = np.sum(current_z * ortho_bases[j]) * dA
            den = np.sum(ortho_bases[j] * ortho_bases[j]) * dA
            inner_prod = num / den
            current_z = current_z - inner_prod * ortho_bases[j]

        ortho_bases.append(current_z)

    return ortho_bases
# ============================================================
# Zernike 拟合：根据波前图拟合系数
# ============================================================
def zernike_fit(wavefront: np.ndarray, zernike_basis: list):
    """
    用最小二乘法将波前拟合到 Zernike 基底。

    参数
    ----
    wavefront     : M×M 波前矩阵
    zernike_basis : 长度为 num_terms 的列表，每项为 M×M 数组

    返回
    ----
    coefficients : 拟合系数向量
    mse          : 拟合均方误差
    """
    M = wavefront.shape[0]
    num_terms = len(zernike_basis)

    b = wavefront.ravel()
    A = np.column_stack([z.ravel() for z in zernike_basis])   # (M², num_terms)

    coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    reconstructed = (A @ coefficients).reshape(M, M)
    mse = np.mean((wavefront - reconstructed) ** 2)
    print(f"Zernike 拟合均方误差: {mse:.6f}")
    return coefficients, mse
# ============================================================
# Zernike 拟合：根据系数生成波前图
# ============================================================
def plot_wavefront_from_zernike(Z_UDA, coeffs):
    """
    根据 Zernike 基底和系数生成波前

    参数:
        Z_UDA     : list of (H,W)  正交 Zernike 基底
        coeffs    : array-like     对应的 Zernike 系数，长度 = len(Z_UDA)

    返回:
        wavefront : (H,W) ndarray  重建波前
    """
    coeffs = np.array(coeffs)
    Z_mat = np.stack(Z_UDA, axis=-1)  # (H, W, nZ)
    wavefront = np.einsum('hwj,j->hw', Z_mat, coeffs)  # (H, W)
    return wavefront
# ============================================================
# 方形区域提取（从全孔径波前中抠出单个子镜区域）
# ============================================================
def extract_square_region(original_matrix: np.ndarray, logic_mask: np.ndarray):
    """
    从原始矩阵中提取逻辑掩码所圈定区域，居中放置于方形零矩阵中。

    返回
    ----
    centered_region  : 方形区域矩阵
    placement_info   : dict，含边界与放置信息
    """
    if original_matrix.shape != logic_mask.shape:
        raise ValueError("原始矩阵与逻辑掩码尺寸不一致")

    rows, cols = np.where(logic_mask)
    if len(rows) == 0:
        raise ValueError("逻辑掩码中没有值为 True 的元素")

    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    region_h = max_row - min_row + 1
    region_w = max_col - min_col + 1
    square_size = max(region_h, region_w)

    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    zero_center = square_size // 2  # 0-indexed

    s_row = zero_center - (center_row - min_row)
    s_col = zero_center - (center_col - min_col)
    e_row = s_row + region_h
    e_col = s_col + region_w

    # 边界裁剪
    s_row = max(0, s_row); e_row = min(square_size, e_row)
    s_col = max(0, s_col); e_col = min(square_size, e_col)
    act_h = e_row - s_row
    act_w = e_col - s_col

    centered_region = np.zeros((square_size, square_size),
                                dtype=original_matrix.dtype)
    centered_region[s_row:e_row, s_col:e_col] = \
        original_matrix[min_row:min_row + act_h, min_col:min_col + act_w]

    placement_info = {
        "square_size": square_size,
        "zero_center": zero_center,
        "original_center": (center_row, center_col),
        "original_region": (min_row, max_row, min_col, max_col),
        "placement_range": (s_row, e_row, s_col, e_col),
        "region_size": (region_h, region_w),
    }
    return centered_region, placement_info
# ============================================================
# 在全孔径坐标中生成 Zernike 基底
# ============================================================
def Z_UDA_in_mask(wavefront_matrix: np.ndarray,
                  mask: np.ndarray,
                  mask1: np.ndarray,
                  bound: dict,
                  zerniken: int,
                  Z_UDA: list) -> list:
    """
    将方形区域内的 Zernike 基底映射回全孔径坐标。

    参数
    ----
    wavefront_matrix : 全孔径波前矩阵（用于确定尺寸）
    mask             : 方形区域内的逻辑掩码
    mask1            : 全孔径逻辑掩码
    bound            : extract_square_region 返回的 placement_info
    zerniken         : Zernike 项数
    Z_UDA            : 列表，每项为方形区域内的 Zernike 基底矩阵

    返回
    ----
    Z_UDA_in_mask_list : 长度为 zerniken 的列表，每项为全孔径尺寸的 Zernike 基底
    """
    s_row, e_row, s_col, e_col = bound["placement_range"]
    orig_min_row, _, orig_min_col, _ = bound["original_region"]

    sq_rows, sq_cols = np.where(mask)  # 方形区域内有效像素

    # 反推全孔径坐标偏移
    row_offset = orig_min_row - s_row
    col_offset = orig_min_col - s_col

    result = []
    for i in range(zerniken):
        buf = np.zeros_like(wavefront_matrix, dtype=float)
        z_basis = Z_UDA[i]
        for r_sq, c_sq in zip(sq_rows, sq_cols):
            r_orig = r_sq + row_offset
            c_orig = c_sq + col_offset
            if (0 <= r_orig < wavefront_matrix.shape[0] and
                    0 <= c_orig < wavefront_matrix.shape[1]):
                buf[r_orig, c_orig] = z_basis[r_sq, c_sq]
        buf *= mask1.astype(float)
        result.append(buf)
    return result
# ============================================================
# 相似度评估（用 NCC 替代 MATLAB SURF，更轻量且无需额外依赖）
# ============================================================
def img_similar(I1: np.ndarray, I2: np.ndarray) -> float:
    """
    计算两幅图像的归一化互相关（NCC）相似度，范围 [0, 1]。
    替代 MATLAB 中基于 SURF 特征匹配的方案（无需 opencv）。
    """
    def _normalize(I):
        lo, hi = I.min(), I.max()
        return (I - lo) / (hi - lo + 1e-12)

    a = _normalize(I1).ravel()
    b = _normalize(I2).ravel()

    # 若尺寸不同，将 I2 插值到 I1 尺寸
    if I1.shape != I2.shape:
        scale = (I1.shape[0] / I2.shape[0], I1.shape[1] / I2.shape[1])
        b = _normalize(zoom(I2, scale)).ravel()
        a = _normalize(I1).ravel()

    ncc = np.dot(a - a.mean(), b - b.mean()) / (
        np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()) + 1e-12)
    # 映射到 [0, 1]
    return float((ncc + 1.0) / 2.0)
# ============================================================
# 评价函数
# ============================================================
def evaluation_function(wvl, d1, d2, Dz,
                        wavefront_attitude: np.ndarray,
                        wavefront_phase_coff: np.ndarray,
                        Z_UDA_all: list,
                        I1: np.ndarray,
                        I2: np.ndarray,
                        deta: np.ndarray) -> float:
    """
    计算当前相位系数下的评价函数值（越接近 0 越好）。

    参数
    ----
    wavefront_phase_coff : shape (seg_number, z_number) 的系数矩阵
    Z_UDA_all            : 二维列表 Z_UDA_all[seg][z_idx]，每项为全孔径 Zernike 基底
    I1, I2               : 参考 PSF（正焦 / 离焦）
    deta                 : 离焦相位矩阵

    返回
    ----
    E : 评价函数值，E = 1 - (sim1 + sim2) / 2
    """
    zeros_padding = 1
    seg_number, z_number = wavefront_phase_coff.shape

    wavefront_phase = np.zeros(wavefront_attitude.shape, dtype=float)
    for j in range(seg_number):
        for i in range(z_number):
            wavefront_phase += wavefront_phase_coff[j, i] * Z_UDA_all[j][i]

    P = zero_padding_complex(
        wavefront_attitude * np.exp(1j * 2 * np.pi * wavefront_phase),
        zeros_padding)
    Pd = zero_padding_complex(P * np.exp(1j * 2 * np.pi * deta), zeros_padding)

    check_aliasing_condition(P, wvl, d1, d2, Dz)
    _, _, PSF_P = two_step_prop(P, wvl, d1, d2, Dz)

    check_aliasing_condition(Pd, wvl, d1, d2, Dz)
    _, _, PSF_Pd = two_step_prop(Pd, wvl, d1, d2, Dz)

    sim1 = img_similar(I1, np.abs(PSF_P) ** 2)
    sim2 = img_similar(I2, np.abs(PSF_Pd) ** 2)
    E = 1.0 - (sim1 + sim2) / 2.0
    return E

def two_step_prop_torch(Uin, wvl, d1, d2, Dz, device):
    """
    PyTorch 版两步菲涅尔传播，支持自动微分
    Uin : (N, N) complex tensor
    """
    N = Uin.shape[0]
    k = 2 * np.pi / wvl

    coords1 = (torch.arange(-N // 2, N // 2, dtype=torch.float32, device=device)) * d1
    x1, y1 = torch.meshgrid(coords1, coords1, indexing='xy')

    m = d2 / d1
    Dz1 = Dz / (1.0 - m)
    d1a = wvl * abs(Dz1) / (N * d1)

    coords1a = (torch.arange(-N // 2, N // 2, dtype=torch.float32, device=device)) * d1a
    x1a, y1a = torch.meshgrid(coords1a, coords1a, indexing='xy')

    # 第一步
    phase1_in  = torch.exp(1j * torch.tensor(k / (2 * Dz1), device=device)
                           * (x1 ** 2 + y1 ** 2).to(torch.complex64))
    phase1_out = torch.exp(1j * torch.tensor(k / (2 * Dz1), device=device)
                           * (x1a ** 2 + y1a ** 2).to(torch.complex64))

    scale1 = torch.tensor(d1 ** 2, device=device, dtype=torch.complex64)
    Uitm = (1.0 / (1j * wvl * Dz1)
            * phase1_out
            * torch.fft.fftshift(
                torch.fft.fft2(
                    torch.fft.ifftshift(Uin * phase1_in)
                )
              ) * scale1)

    # 第二步
    Dz2 = Dz - Dz1
    coords2 = (torch.arange(-N // 2, N // 2, dtype=torch.float32, device=device)) * d2
    x2, y2 = torch.meshgrid(coords2, coords2, indexing='xy')

    phase2_in  = torch.exp(1j * torch.tensor(k / (2 * Dz2), device=device)
                           * (x1a ** 2 + y1a ** 2).to(torch.complex64))
    phase2_out = torch.exp(1j * torch.tensor(k / (2 * Dz2), device=device)
                           * (x2 ** 2 + y2 ** 2).to(torch.complex64))

    scale2 = torch.tensor(d1a ** 2, device=device, dtype=torch.complex64)
    Uout = (1.0 / (1j * wvl * Dz2)
            * phase2_out
            * torch.fft.fftshift(
                torch.fft.fft2(
                    torch.fft.ifftshift(Uitm * phase2_in)
                )
              ) * scale2)

    return Uout

