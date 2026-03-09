import clr, os, winreg
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 在 import pyplot 之前设置
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
from itertools import islice
from fresnel_utils import (
    zernike_fit,
    Z_UDA_in_mask,
    two_step_prop,
    extract_square_region,
    gram_schmidt_mask,
    plot_wavefront_from_zernike
    )
from PD_func import(
    phase_diversity_retrieve,
    estimate_object,
    plot_phase_diversity_result,
    sensitivity_scan,
    compute_hessian,
    cost_pd_image,
    simulate_imaging
)
# This boilerplate requires the 'pythonnet' module.
# The following instructions are for installing the 'pythonnet' module via pip:
#    1. Ensure you are running a Python version compatible with PythonNET. Check the article "ZOS-API using Python.NET" or
#    "Getting started with Python" in our knowledge base for more details.
#    2. Install 'pythonnet' from pip via a command prompt (type 'cmd' from the start menu or press Windows + R and type 'cmd' then enter)
#
#        python -m pip install pythonnet

# determine the Zemax working directory
aKey = winreg.OpenKey(winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER), r"Software\Zemax", 0, winreg.KEY_READ)
zemaxData = winreg.QueryValueEx(aKey, 'ZemaxRoot')
NetHelper = os.path.join(os.sep, zemaxData[0], r'ZOS-API\Libraries\ZOSAPI_NetHelper.dll')
winreg.CloseKey(aKey)

# add the NetHelper DLL for locating the OpticStudio install folder
clr.AddReference(NetHelper)
import ZOSAPI_NetHelper

pathToInstall = ''
# uncomment the following line to use a specific instance of the ZOS-API assemblies
#pathToInstall = r'C:\C:\Program Files\Zemax OpticStudio'

# connect to OpticStudio
success = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(pathToInstall);

zemaxDir = ''
if success:
    zemaxDir = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory();
    print('Found OpticStudio at:   %s' + zemaxDir);
else:
    raise Exception('Cannot find OpticStudio')

# load the ZOS-API assemblies
clr.AddReference(os.path.join(os.sep, zemaxDir, r'ZOSAPI.dll'))
clr.AddReference(os.path.join(os.sep, zemaxDir, r'ZOSAPI_Interfaces.dll'))
import ZOSAPI

TheConnection = ZOSAPI.ZOSAPI_Connection()
if TheConnection is None:
    raise Exception("Unable to intialize NET connection to ZOSAPI")

TheApplication = TheConnection.ConnectAsExtension(0)
if TheApplication is None:
    raise Exception("Unable to acquire ZOSAPI application")

if TheApplication.IsValidLicenseForAPI == False:
    raise Exception("License is not valid for ZOSAPI use.  Make sure you have enabled 'Programming > Interactive Extension' from the OpticStudio GUI.")

TheSystem = TheApplication.PrimarySystem
if TheSystem is None:
    raise Exception("Unable to acquire Primary system")

def reshape(data, x, y, transpose = False):
    """Converts a System.Double[,] to a 2D list for plotting or post processing
    
    Parameters
    ----------
    data      : System.Double[,] data directly from ZOS-API 
    x         : x width of new 2D list [use var.GetLength(0) for dimension]
    y         : y width of new 2D list [use var.GetLength(1) for dimension]
    transpose : transposes data; needed for some multi-dimensional line series data
    
    Returns
    -------
    res       : 2D list; can be directly used with Matplotlib or converted to
                a numpy array using numpy.asarray(res)
    """
    if type(data) is not list:
        data = list(data)
    var_lst = [y] * x;
    it = iter(data)
    res = [list(islice(it, i)) for i in var_lst]
    if transpose:
        return self.transpose(res);
    return res
    
def transpose(data):
    """Transposes a 2D list (Python3.x or greater).  
    
    Useful for converting mutli-dimensional line series (i.e. FFT PSF)
    
    Parameters
    ----------
    data      : Python native list (if using System.Data[,] object reshape first)    
    
    Returns
    -------
    res       : transposed 2D list
    """
    if type(data) is not list:
        data = list(data)
    return list(map(list, zip(*data)))

print('Connected to OpticStudio')

# The connection should now be ready to use.  For example:
print('Serial #: ', TheApplication.SerialCode)
# 自动寻找系统中可用的中文字体
def set_chinese_font():
    # Windows 常见中文字体候选列表
    chinese_fonts = [
        'Microsoft YaHei',  # 微软雅黑
        'SimHei',  # 黑体
        'SimSun',  # 宋体
        'KaiTi',  # 楷体
        'FangSong',  # 仿宋
    ]

    available = {f.name for f in fm.fontManager.ttflist}

    for font in chinese_fonts:
        if font in available:
            matplotlib.rcParams['font.family'] = font
            print(f"已使用字体: {font}")
            return font

    # 若以上都没有，直接指定字体文件路径
    font_paths = [
        r'C:\Windows\Fonts\msyh.ttc',  # 微软雅黑
        r'C:\Windows\Fonts\simhei.ttf',  # 黑体
        r'C:\Windows\Fonts\simsun.ttc',  # 宋体
    ]
    for path in font_paths:
        import os
        if os.path.exists(path):
            font_prop = fm.FontProperties(fname=path)
            matplotlib.rcParams['font.family'] = font_prop.get_name()
            fm.fontManager.addfont(path)
            print(f"已加载字体文件: {path}")
            return path

    print("⚠️ 未找到中文字体，汉字可能显示为方块")
    return None
set_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False  # 修复负号显示
# ============================================================
# 1. 参数设置
# ============================================================
a       = 1e-4              # 归一化单位 (m)
N0      = 512               # 采样点数
d1      = 6.605 * a / N0    # 源平面采样间距 (m)
d2      = 10e-6 / a          # 像面采样间距 (m)
D1      = 6.605 * a         # 系统孔径 (m)
D2      = 512 * d2          # 接收面尺寸 (m)
wvl     = 2e-6              # 波长 (m)
Dz      = 132.812           # 传播距离/焦距 (m)
gamma   = 0              # 损失函数分母的正则化系数
tol     = 1e-6              # 损失函数收敛阈值
# ============================================================
# 2. 加载文件
# ============================================================
TheSystem = TheApplication.PrimarySystem
sampleDir = TheApplication.SamplesDir
testFile = os.path.join(sampleDir, "JWSTwithapeatur.zmx")
TheSystem.LoadFile(testFile, False)
print(f"已加载文件: {testFile}")
# ============================================================
# 3. 创建分析对象
# ============================================================
wavefront_analysis = TheSystem.Analyses.New_Analysis(
    ZOSAPI.Analysis.AnalysisIDM.WavefrontMap
)
PSF_analysis = TheSystem.Analyses.New_Analysis(
    ZOSAPI.Analysis.AnalysisIDM.HuygensPsf
)
# ============================================================
# 4. 波前图设置
# ============================================================
wavefront_settings = wavefront_analysis.GetSettings()
wavefront_settings.Sampling = ZOSAPI.Analysis.SampleSizes.S_512x512
wavefront_settings.ShowAs = ZOSAPI.Analysis.ShowAs.FalseColor
wavefront_settings.UseExitPupil = False
# ============================================================
# 5. PSF 设置
# ============================================================
PSF_settings = PSF_analysis.GetSettings()
PSF_settings.PupilSampleSize  = ZOSAPI.Analysis.SampleSizes.S_128x128
PSF_settings.ImageSampleSize  = ZOSAPI.Analysis.SampleSizes.S_512x512
PSF_settings.ImageDelta       = 10
PSF_settings.ShowAsType       = ZOSAPI.Analysis.HuygensShowAsTypes.InverseGreyScale
PSF_settings.Normalize        = True
PSF_settings.UseCentroid      = False
# ============================================================
# 7. 获取初始波前图
# ============================================================
wavefront_analysis.ApplyAndWaitForCompletion()
wavefront_results = wavefront_analysis.GetResults()
wavefront_data = wavefront_results.GetDataGrid(0)
# 获取网格尺寸
nx = wavefront_data.Nx  # 列数
ny = wavefront_data.Ny  # 行数
wavefront_matrix_focus = np.array(list(wavefront_data.Values), dtype=float)
# 将 NaN 替换为 0（对应 MATLAB 的 isnan + 赋零）
wavefront_matrix_focus = np.nan_to_num(wavefront_matrix_focus, nan=0.0)
# ✅ 变成二维
wavefront_matrix_focus = wavefront_matrix_focus.reshape(ny, nx)
# ============================================================
# 8. 获取初始PSF
# ============================================================
PSF_analysis.ApplyAndWaitForCompletion()
PSF_results = PSF_analysis.GetResults()
PSF_data = PSF_results.GetDataGrid(0)
psf_nx, psf_ny = PSF_data.Nx, PSF_data.Ny  # PSF 尺寸可能不同（512x512）
PSF_matrix_zemax = np.array(list(PSF_data.Values), dtype=float)
PSF_matrix_zemax = PSF_matrix_zemax.reshape(psf_ny, psf_nx)  # ✅ reshape
# 对应 MATLAB: flip(flip(PSF_matrix_zemax, 1), 2)
# MATLAB axis 1 = 行(上下翻转), axis 2 = 列(左右翻转)
PSF_matrix_focus = np.flip(np.flip(PSF_matrix_zemax, axis=0), axis=1)
PSF_matrix_focus_norm=PSF_matrix_focus/PSF_matrix_focus.max()
# ============================================================
# 构建zernike基底
# ============================================================
wavefront_attitude = (wavefront_matrix_focus != 0).astype(float)
[UDA1,bound]=extract_square_region(wavefront_matrix_focus,wavefront_attitude)
mask=(UDA1!=0)
# 计算正交Zernike多项式
mat_path=r"C:\Users\25313\Documents\Zemax\ZOS-API Projects\MATLABStandaloneApplication7\Zernikefringe.mat"
Z_UDA_temp= gram_schmidt_mask(37,1,bound['square_size'],1,0,0,mask,mat_path)
Z_UDA=Z_UDA_in_mask(wavefront_matrix_focus,mask,wavefront_attitude,bound,37,Z_UDA_temp)
[F0,mse0] = zernike_fit(wavefront_matrix_focus, Z_UDA)
# ============================================================
# 保存原始厚度
# ============================================================
LDESurface = TheSystem.LDE      # 镜头数据编辑器
scGroup = LDESurface.GetRowAt(6)
temp_focus = scGroup.Thickness   # 保存原始厚度
# ============================================================
# 引入离焦量
# ============================================================
defocus_steps =[-2,-1,1,2]
# 存储多组结果
coff_div_list      = []   # 每个离焦量对应的 Zernike 系数
PSF_de_norm_list   = []   # 每个离焦量对应的归一化 PSF
for step in defocus_steps:
    scGroup.Thickness = step*0.04
    # ============================================================
    # 10. 获取离焦后波前图
    # ============================================================
    wavefront_analysis.ApplyAndWaitForCompletion()
    wavefront_results = wavefront_analysis.GetResults()
    wavefront_data = wavefront_results.GetDataGrid(0)
    wavefront_matrix_de = np.array(list(wavefront_data.Values), dtype=float)
    wavefront_matrix_de = np.nan_to_num(wavefront_matrix_de, nan=0.0)
    wavefront_matrix_de = wavefront_matrix_de.reshape(ny, nx)  # ✅ reshape
    # 计算波前差
    wavefront_matrix_deta = wavefront_matrix_de - wavefront_matrix_focus
    coff_div, mse1 = zernike_fit(wavefront_matrix_deta, Z_UDA)
    coff_div_list.append(coff_div)
    # ============================================================
    # 11. 获取离焦后 PSF
    # ============================================================
    PSF_analysis.ApplyAndWaitForCompletion()
    PSF_results = PSF_analysis.GetResults()
    PSF_data = PSF_results.GetDataGrid(0)
    psf_nx, psf_ny = PSF_data.Nx, PSF_data.Ny  # PSF 尺寸可能不同（512x512）
    PSF_matrix_zemax = np.array(list(PSF_data.Values), dtype=float)
    PSF_matrix_zemax = PSF_matrix_zemax.reshape(psf_ny, psf_nx)  # ✅ reshape
    # 对应 MATLAB: flip(flip(PSF_matrix_zemax, 1), 2)
    # MATLAB axis 1 = 行(上下翻转), axis 2 = 列(左右翻转)
    PSF_matrix_de = np.flip(np.flip(PSF_matrix_zemax, axis=0), axis=1)
    PSF_matrix_de_norm=PSF_matrix_de /PSF_matrix_de.max()
    PSF_de_norm_list.append(PSF_matrix_de_norm)
# ============================================================
# 13. 恢复原始厚度
# ============================================================
scGroup.Thickness = temp_focus
print(f"已恢复原始厚度: {temp_focus} mm")
# ============================================================
# 12. 绘制离焦前后波前图（对应 MATLAB subplot(1,3,x)）
# ============================================================
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# axes = axes.flatten()  # 将 2×2 打平为 [ax0, ax1, ax2, ax3]
#
# im0 = axes[0].imshow(wavefront_matrix_focus, origin='lower', aspect='auto')
# plt.colorbar(im0, ax=axes[0])
# axes[0].set_title("初始波前图")
#
# im1 = axes[1].imshow(PSF_matrix_focus_norm, origin='lower', aspect='auto')
# plt.colorbar(im1, ax=axes[1])
# axes[1].set_title("初始PSF")
#
# im2 = axes[2].imshow(wavefront_matrix_de, origin='lower', aspect='auto')
# plt.colorbar(im2, ax=axes[2])
# axes[2].set_title("离焦波前图")
#
# im2 = axes[3].imshow(PSF_matrix_de_norm, origin='lower', aspect='auto')
# plt.colorbar(im2, ax=axes[3])
# axes[3].set_title("离焦PSF")
# plt.tight_layout()
# plt.savefig("image/wavefront_PSF.png", dpi=150)
# plt.show()
# print("波前图已保存为 wavefront_PSF.png")

# ============================================================
# 构建复振幅波前
# ============================================================
# wavefront_phase    = np.exp(1j * 2 * np.pi * wavefront_matrix_focus)
# wavefront_attitude = (wavefront_matrix_focus != 0).astype(float)
# wavefront_complex  = wavefront_attitude * wavefront_phase
# x_psf, y_psf, hex_7comband_psf = two_step_prop(
#         wavefront_complex, wvl, d1, d2, Dz)
# # x_psf 为归一化坐标，乘以 a 还原为物理单位 (m)
# x_figure = x_psf * a
# y_figure = y_psf * a
# PSF_two = np.abs(hex_7comband_psf) ** 2
# PSF_two_norm = PSF_two /PSF_two.max()
# ============================================================
# 绘制公式PSF和zemax的PSF
# ============================================================
# fig, ax = plt.subplots(1,3,figsize=(18, 5))
#
# im1 = ax[0].imshow(PSF_matrix_focus_norm,origin='lower', aspect='equal')
# plt.colorbar(im1, ax=ax[0])
# ax[0].set(title="Zemax 的 PSF 图像", xlabel="x / mm", ylabel="y / mm")
#
# im2 = ax[1].imshow(PSF_two_norm,origin='lower', aspect='equal')
# plt.colorbar(im2, ax=ax[1])
# ax[1].set(title="Python 根据衍射公式得到的 PSF 图像", xlabel="x / mm", ylabel="y / mm")
#
# im3 = ax[2].imshow(PSF_matrix_focus_norm-PSF_two_norm,origin='lower', aspect='equal')
# plt.colorbar(im3, ax=ax[2])
# ax[2].set(title="PSF图像差值", xlabel="x / mm", ylabel="y / mm")
# plt.tight_layout()
# plt.savefig("image/psf_zemax_and_python.png", dpi=150)
# plt.show()
Z_UDA_10=Z_UDA[:10]
coff_div_list_10 = [c[:10] for c in coff_div_list]
# # ============================================================
# # 统一准备所有中间变量（在调用优化和诊断函数之前运行一次）
# # ============================================================
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# image_focus=PSF_matrix_focus_norm
# image_de_norm_list=PSF_de_norm_list
# H, W = image_focus.shape
# nZ   = len(Z_UDA_10)
# gamma = 1e-6
#
# def to_t(x):
#     return torch.tensor(np.array(x).astype(np.float32),
#                         dtype=torch.float32, device=device)
#
# # Zernike 基底 / 瞳函数
# Z_mat      = to_t(np.stack(Z_UDA_10, axis=-1))
# pupil_mask = to_t((np.abs(Z_UDA_10[0]) > 0).astype(np.float32))
#
# # 图像 → FFT → imgDs (H,W,K)
# I_focus   = torch.fft.fft2(to_t(image_focus)).to(torch.complex64)
# I_de_list = [torch.fft.fft2(to_t(im)).to(torch.complex64)
#              for im in image_de_norm_list]
# imgDs     = torch.stack([I_focus] + I_de_list, dim=2)
#
# # 多样性像差 → wavefront_deltas (H,W,K)
# delta_focus = torch.zeros(H, W, dtype=torch.float32, device=device)
# delta_de_list = [to_t(plot_wavefront_from_zernike(Z_UDA_10, coff).astype(np.float32))
#                  for coff in coff_div_list_10]
# wavefront_deltas = torch.stack([delta_focus] + delta_de_list, dim=2)
#
# # ============================================================
# # 包装损失函数为纯 numpy 接口
# # ============================================================
# def cost_np(c_np):
#     with torch.no_grad():
#         c_t = torch.tensor(c_np.astype(np.float32), device=device)
#         return cost_pd_image(c_t, Z_mat, pupil_mask,
#                              imgDs, wavefront_deltas,
#                              wvl, d1, d2, Dz, device,
#                              gamma=gamma).item()
#
# # ============================================================
# # 调用诊断函数
# # ============================================================
# # Hessian 耦合分析
# H_mat, corr = compute_hessian(cost_np, F0[:10], eps=1e-3)
# # 灵敏度扫描，确保F0的系数（真值）对于损失函数而言是最小值
# J_curves, scan_vals = sensitivity_scan(
#     PSF_matrix_focus_norm,
#     PSF_de_norm_list,
#     Z_UDA_10,
#     coff_div_list=coff_div_list_10,
#     F0=F0[:10],
#     wvl=wvl, d1=d1,
#     d2=d2, Dz=Dz,
#     scan_range=0.5,    # 每个系数扰动范围 ±0.5
#     scan_steps=41      # 扫描点数
# )
# ============================================================
# 导入图片
# ============================================================
def load_image(path):
    img = mpimg.imread(path)          # 读取图片，值域自动处理
    if img.ndim == 3:
        img = np.mean(img, axis=2)    # RGB → 灰度
    img = img.astype(np.float32)
    img = img / (img.max() + 1e-30)
    return img

image_folder = r"C:\Users\25313\Documents\Zemax\ZOS-API Projects\PythonZOSConnection1\orginal_image"            # 修改为你的图片文件夹路径
image_files  = [f for f in os.listdir(image_folder)
                if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif'))]

print(f"找到 {len(image_files)} 张图片：{image_files}")
image_path = os.path.join(image_folder, image_files[0])
image_raw  = load_image(image_path)
print(f"原始图像尺寸: {image_raw.shape}")
# ============================================================
# 生成卷积图片
# ============================================================
img_focus,image_upsampled_temp = simulate_imaging(image_raw, PSF_matrix_focus_norm,
                               d_image=10e-6,
                               d_psf=10e-6)
img_de      = []   # 每个离焦量对应的图片
for i in range(len(PSF_de_norm_list)):
    img_out_temp,image_upsampled_temp = simulate_imaging(image_raw, PSF_de_norm_list[i],
                               d_image=10e-6,
                               d_psf=10e-6)
    img_de.append(img_out_temp)
# ============================================================
# 相位差算法调用
# ============================================================
c_est, J_hist, wavefront_est = phase_diversity_retrieve(
    img_focus,
    img_de,
    Z_UDA_10,
    coff_div_list = coff_div_list_10,        # 已知离焦量
    wvl=wvl, d1=d1,
    d2=d2, Dz=Dz,
    gamma=gamma,
    lr=0.001, max_iter=10000,
    tol=tol,
    optimizer_type='Adam'
)
#   用最优系数恢复物体
obj_est, obj_fft = estimate_object(
    c_est, Z_UDA_10,
    PSF_matrix_focus_norm,
    PSF_de_norm_list,
    coff_div_list=coff_div_list_10,
    gamma=gamma
)
plot_phase_diversity_result(
    PSF_matrix_focus_norm, PSF_de_norm_list,
    wavefront_est, c_est, J_hist, Z_UDA_10,
    coff_div_list = coff_div_list_10,
    wvl=wvl, d1=d1,
    d2=d2, Dz=Dz,
    wavefront_true = wavefront_matrix_focus,
    c_true = F0[:10],
    save_path="image_test4/phase_diversity_result_OTFloss.png"
)


