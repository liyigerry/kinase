# -*- coding: utf-8 -*-
# -------------------------------- Functional Module ---------------------------------
__all__ = [
    # Attribute
    '',

    #  Method
    '',

    # type
    '',

    # object
    '',
]
# -------------------------------- Functional Module ---------------------------------


# -------------------------------- Magic Attribute ---------------------------------
__author__ = "陆家立"  
__copyright__ = "版权所有 2024, 陆家立"  
__credits__ = ["个人"]  
__license__ = "GNU GPL 3.0"  
__version__ = "1.0.0"  
__maintainer__ = "陆家立"  
__email__ = "996153075@qq.com"  
__status__ = "开发"  

import math
import os
import datetime

__ctime__ = datetime.datetime.fromtimestamp(os.path.getctime(__file__)).strftime('%Y-%m-%d  %H:%M:%S')  
__mtime__ = datetime.datetime.fromtimestamp(os.path.getmtime(__file__)).strftime('%Y-%m-%d  %H:%M:%S')  
__atime__ = datetime.datetime.fromtimestamp(os.path.getatime(__file__)).strftime('%Y-%m-%d  %H:%M:%S')  
__date__ = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')  

# -------------------------------- Magic Attribute ---------------------------------


# -------------------------------- import library ---------------------------------
import sys
import os

tool_module_path = r'e:/Kinase_mapping/makemap'
if tool_module_path not in sys.path:
    sys.path.append(tool_module_path)

import mdtraj as md
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt, cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import AutoLocator, FormatStrFormatter, AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.ma import mean
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter
from scipy.stats import alpha
from tqdm import tqdm


# -------------------------------- import library ---------------------------------


# -------------------------------- Attribute---------------------------------
COLOR = "#000000"
title_mapping = {
    "J_DT": "BTK",
    "J_DZ": "BTK/C12",
    "J_SJ-137": "BTK/C137",
    "J_SJ-1216": "BTK/C1216",
    "J_SJ-2847": "BTK/C2847",
    "J_SJ-2909": "BTK/C2909",
    "J_SJ-5598": "BTK/C5598"
}
# -------------------------------- Attribute---------------------------------


# --------------------------------  Method---------------------------------
def h_concat(images, max_row = 2, cmap_img=None):
    widths, heights = zip(*(i.size for i in images))
    ncol = math.ceil(len(images) / max_row)
    total_width = widths[0] * ncol
    max_height = max(heights) * max_row
    max_height = max_height + cmap_img.size[1]
    mm_width = total_width - cmap_img.size[0]
    mm_width = int(mm_width/2)
    print("image =", images[0].size)
    print("cmap_img =", cmap_img.size)
    print("mm_width =", mm_width)

    new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    x_offset = 0
    y_offset = 0
    I_ = 0
    for row in range(max_row):
        for col in range(ncol):
            if I_ < len(images) and I_ < 2:
                new_im.paste(images[I_], (images[I_].width * col, 0))
                I_ += 1
            elif len(images) > I_ >= 2 and row == 0:
                break
            elif len(images) > I_ >= 2:
                new_im.paste(images[I_], (images[I_].width * col, images[I_].height))
                print((images[I_].width * col, images[I_].height))
                I_ += 1
            else:
                break
    print((0, max_height))
    new_im.paste(cmap_img, (mm_width, max(heights) * max_row))

    return new_im


def calculate_free_energy(
        rmsdValues: np.ndarray,
        rgValues: np.ndarray,
        range_=None,
        temperature=300,
        bins=100,
        density=None,
        weights=None,
):

    z0, x0, y0 = np.histogram2d(
        x=rmsdValues,
        y=rgValues ,
        bins=bins,
        range=([min(rmsdValues), max(rmsdValues)], [min(rgValues), max(rgValues)]) if range_ is None else range_,
        density=density,
        weights=weights,
    )
    X0 = 0.5 * (x0[:-1] + x0[1:])
    Y0 = 0.5 * (y0[:-1] + y0[1:])
    z0min_nonzero = np.min(z0[np.where(z0 > 0)])
    z0 = np.maximum(z0, z0min_nonzero)

    eq = 0.008593333
    en = -eq * temperature
    print(f"eq:{eq}, en:{en}")
    Free = en * np.log(z0)
    Free -= np.max(Free)
    Free = np.minimum(Free, 0)
    return [X0, Y0, Free]


def merge_images(images, output_path, cols=3):
    img_list = [Image.open(img_path) for img_path in images]
    widths, heights = zip(*(i.size for i in img_list))

    max_width = max(widths)
    total_height = sum(heights[i] for i in range(len(heights)) if i % cols == 0)

    new_image = Image.new('RGB', (max_width * cols, total_height), color=(255, 255, 255))

    x_offset = 0
    y_offset = 0
    row_count = 0

    for index, img in enumerate(img_list):
        new_image.paste(img, (x_offset, y_offset))
        x_offset += img.width

        if (index + 1) % cols == 0:
            x_offset = 0
            y_offset += img.height
            row_count += 1

    new_image.save(output_path, dpi=(600, 300))


def get_color_bar(
        name: str = "custom_cmap",
        colors: list = None,  
        N: int = 256, 
        vmin: int = 0,
        vmax: int = 1,
        direction: int = 0,
        label = "Free Energy (KJ/mol)",
        font_size: int = 14,
        label_pad: int = 20,
        save_path: str = 'colorbar.png',
        is_show: bool = False,
):
    cmap = LinearSegmentedColormap.from_list(name, colors, N=N)
    add_axes = (0.05, 0.5, 0.9, 0.4)
    orientation = 'horizontal'
    figsize = (8, 2)
    if direction:
        add_axes = (0.5, 0.05, 0.4, 0.9)
        orientation = 'vertical'
        figsize = (2, 8)
    color_bar_fig = plt.figure(figsize=figsize)  
    cax = color_bar_fig.add_axes(add_axes)  
    cbar = color_bar_fig.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
            cmap=cmap
        ),
        cax=cax,
        orientation=orientation
    )
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label(label, fontsize=font_size, labelpad=label_pad)  
    color_bar_fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=600)
    if is_show: plt.show()
    plt.close(color_bar_fig)


def compute_hist(
        data,
        bins: int = 100,
        range: Any = None,
        density=None,
        weights=None,
):

    z0, x0, y0 = np.histogram2d(
        x=data[0],
        y=data[1],
        bins=bins,
        range=([min(data[0]), max(data[0])], [min(data[1]), max(data[1])]) if range is None else range,
        density=density,
        weights=weights,
    )
    X0 = 0.5 * (x0[:-1] + x0[1:])
    Y0 = 0.5 * (y0[:-1] + y0[1:])
    z0min_nonzero = np.min(z0[np.where(z0 > 0)])
    z0 = np.maximum(z0, z0min_nonzero)

    Free = -2.578 * np.log(z0)
    Free -= np.max(Free)
    Free = np.minimum(Free, 0)
    # fig, ax = plt.subplots(2, 2)
    return X0, Y0, Free


def load_style(
        font_path: str = r"E:\Kinase_mapping\jupyter_draw\Times New Roman.ttf",
        font_size: int = 24,
):
    try:
        plt.style.use("chartlab.mplstyle")
    except:
        pass
    font = FontProperties(fname=font_path, size=font_size-4)
    plt.rcParams['font.sans-serif'] = [font.get_name()]  
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams['text.color'] = COLOR  
    plt.rcParams['axes.labelcolor'] = COLOR  
    plt.rcParams['xtick.color'] = COLOR 
    plt.rcParams['ytick.color'] = COLOR
    return font


def rg_rmsd(
        rmsd_data,
        rg_data,
        data,
        label,
        col_num: int = 12,
        sigma: float = 1.5,
        font_size: int = 32,
        font=None,
        text_location=(-0.10, 1.1),
        x_lim: tuple = (-0.5, 5.5),
        y_lim: tuple = (-0.5, 5.5),
        save_dir: str = None,
        is_show: bool = True,
):
    font = load_style(font_size=font_size) if font is None else font
    fig, axs = plt.subplots(
        nrows=2, ncols=2,
        width_ratios=[5, 1], height_ratios=[1, 5],
        figsize=(10, 10), dpi=600)
    fig.subplots_adjust(wspace=0, hspace=0)

    fig.delaxes(axs[0, 1])

    if "." in label:
        prefix, system_name = label.split(".", 1)
        prefix += "."  
    else:

        prefix = chr(index__ + 65) + "."
        system_name = title_mapping.get(label, label)


    axs[0, 0].text(
        x=0, y=1.45,  
        s=prefix,
        transform=axs[0, 0].transAxes,
        fontproperties=font,
        fontsize=32,
        ha='left',
        va='center'
    )

    renderer = fig.canvas.get_renderer()
    text_obj = axs[0, 0].text(
        x=0, y=0,
        s=system_name,
        transform=axs[0, 0].transAxes,
        fontproperties=font,
        fontsize=32,
        ha='left',
        va='center'
    )
    text_width = text_obj.get_window_extent(renderer=renderer).width
    fig_width = fig.get_window_extent(renderer=renderer).width
    ax_width = axs[0, 0].get_window_extent(renderer=renderer).width
    ax_x0 = axs[0, 0].get_window_extent(renderer=renderer).x0
    fig_x0 = fig.get_window_extent(renderer=renderer).x0
    relative_x = (ax_x0 - fig_x0 + ax_width / 2 - text_width / 2) / fig_width

    axs[0, 0].text(
        x=relative_x, y=1.45,
        s=system_name,
        transform=axs[0, 0].transAxes,
        fontproperties=font,
        fontsize=32,
        ha='left',
        va='center'
    )
    text_obj.remove()

    axs[0, 0].spines['top'].set_visible(False)
    axs[0, 0].spines['right'].set_visible(False)
    axs[1, 1].spines['top'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)

    cmap = cm.colors.LinearSegmentedColormap.from_list(
         'new_map6', [cm.nipy_spectral(i) for i in range(0, 256, 1)] + ['white'], col_num)
    cmap_r = LinearSegmentedColormap.from_list('one_cmap', ['#BB0000', '#FFFFFF'], col_num)


    if sigma: sig_data = gaussian_filter(data[2], sigma=sigma).T
    else: sig_data = data[2].T
    levels = [i for i in range(-col_num, 0, 1)] + [0]
    extent = [min(data[1]), max(data[1]), min(data[0]), max(data[0])]

    im = axs[1, 0].contourf(
        data[0], data[1], sig_data,
        cmap=cmap,
        levels=levels,
        extent=extent,
    )
    cmap = im.cmap
    norm = im.norm

    color_dict = {}
    for level in levels:
        normalized_level = norm(level)
        rgba_color = cmap(normalized_level)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba_color[0] * 255), int(rgba_color[1] * 255),
                                                 int(rgba_color[2] * 255))
        color_dict[level] = hex_color

    cs = axs[(1, 0)].contour(
        data[0], data[1], sig_data,
        cmap=cmap,
        linewidths=0.01,
        levels=levels,
        extent=extent,
                             )
    rmsdKde_result = sns.kdeplot(
        rmsd_data.T,
        label=label+'_RMSD',
        color='r',
        fill=True,
        bw_adjust=0.5,
        ax=axs[0, 0]
    )
    rg_result = sns.kdeplot(
        y=rg_data,
        label=label+'_Rg',
        color='b',
        fill=True,
        bw_adjust=0.5,
        ax=axs[1, 1]
    )
    axs[0, 0].set_xlim(x_lim[0], x_lim[1])
    axs[1, 0].set_xlim(x_lim[0], x_lim[1])
    axs[1, 0].set_ylim(y_lim[0], y_lim[1])
    axs[1, 1].set_ylim(y_lim[0], y_lim[1])

    axs[0, 0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1, 0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1, 0].yaxis.set_minor_locator(AutoMinorLocator())
    axs[1, 1].yaxis.set_minor_locator(AutoMinorLocator())

    axs[0, 0].tick_params(axis='x', which='minor', length=8, color='r')
    axs[1, 0].tick_params(axis='x', which='minor', length=8, color='r')
    axs[1, 0].tick_params(axis='y', which='minor', length=8, color='r')
    axs[1, 1].tick_params(axis='y', which='minor', length=8, color='r')

    axs[0, 0].tick_params(axis='both', which='major', labelsize=30)
    axs[0, 0].tick_params(axis='both', which='minor', labelsize=30)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=30)
    axs[1, 0].tick_params(axis='both', which='minor', labelsize=30)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=30)
    axs[1, 1].tick_params(axis='both', which='minor', labelsize=30)

    axs[0, 0].set_ylabel('Frequency', fontsize=font_size - 2, fontproperties=font)
    axs[0, 0].yaxis.set_label_coords(
        x=-0.10, 
        y=0.8    
    )
    axs[0, 0].set_xlabel(' ')
    axs[1, 0].set_xlabel('RMSD(Å)', fontsize=32, fontproperties=font)
    axs[1, 0].set_ylabel('Rg(Å)', fontsize=32,fontproperties=font)
    axs[1, 1].set_xlabel('Frequency', fontsize=32, fontproperties=font)
    axs[1, 1].set_ylabel(' ')

    colors_tag = ['#000000', '#820093', '#0000bb', '#0077dd', '#00a4b9', '#00a356', '#00bc00', '#00f500', '#cef800', '#ffc900', '#ff2800', '#d60000', '#ffffff', '#ffffff']
    get_color_bar(colors=colors_tag, vmin=-col_num, vmax=0, font_size=font_size - 2, direction=0, save_path=os.path.join(save_dir, '颜色条1.png'))
    get_color_bar(colors=colors_tag, vmin=-col_num, vmax=0, font_size=font_size - 2, direction=1, save_path=os.path.join(save_dir, '颜色条2.png'))

    plt.subplots_adjust(
    left=0.15,    
    right=0.95,  
    bottom=0.1,  
    top=0.9,    
    wspace=0.3,   
    hspace=0.3   
)
    axs[0, 0].autoscale(enable=True, axis='x', tight=False)
    axs[1, 1].autoscale(enable=True, axis='y', tight=False)

    axs[0, 0].xaxis.set_visible(False)
    axs[1, 1].yaxis.set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)

    if save_dir: 
        safe_label = label.replace('/', '_')  
        save_path = os.path.join(save_dir, f'{safe_label}_rg_rmsd.png')
    else: 
        safe_label = label.replace('/', '_')
    save_path = f'{safe_label}_rg_rmsd.png'
    plt.savefig(save_path)
    if is_show: plt.show()
    return save_path
def rmsf(
        ax,
        data,
        color,
        label,
        linewidth:float = 0.8,
        start_location: int = 0,
        Alpha: float = 1,
        font = None,
):
    ax.plot(
        range(len(data)-start_location), data[start_location:],
        label=label, color=color,
        linewidth=linewidth, alpha=Alpha)
    ax.legend(fontsize=8, loc=9, ncol=3)
    ax.set_xlim([0, len(data)])
    ax.set_ylim([0, max(data[start_location:])+0.01])


def fel_2d(
        ax,
        data: list=None,
        figsize=(12, 10),
        title: str = "FEL",
        xlabel: str = "RMSD(ns)",
        ylabel: str = "Rg(ns)",
        IP_value: int = 1,
        font_size: int = 36,
        font=None,
        text_location=(-0.10, 1.1),
        labelpad=15,
        pad=0.15,
):
    """"""
    font = load_style(font_size=font_size) if font is None else font
    ax.text(text_location[0], text_location[1], title, transform=ax.transAxes,
            fontproperties=font, fontsize=font_size + 2, ha='left', va='center')

    ip_func: interp2d = interp2d(data[0], data[1], data[2], kind="linear")
    x_new = np.linspace(np.min(data[0]), np.max(data[0]), IP_value * len(data[0]))
    y_new = np.linspace(np.min(data[1]), np.max(data[1]), IP_value * len(data[1]))

    img_new = ip_func(x_new, y_new)
    x_new, y_new = np.meshgrid(x_new, y_new)
    img_new = img_new.reshape(len(x_new), len(y_new))
    z_min = np.min(img_new)
    im = ax.contourf(
        x_new,
        y_new,
        img_new,
        zdir="z",
        offset=z_min,
        cmap="coolwarm",
    )

    ax.set_xlim(min(data[0]), max(data[0]))
    ax.set_ylim(min(data[1]), max(data[1]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis='x', labelsize=font_size-2)
    ax.tick_params(axis='y', labelsize=font_size-2)

    ax.set_xlabel(xlabel, fontsize=font_size, fontproperties=font, labelpad=labelpad)
    ax.set_ylabel(ylabel, fontsize=font_size, fontproperties=font, labelpad=labelpad)

    return im


def fel_3d(
        ax,
        data: list=None,
        title: str = "FEL",
        xlabel: str = "RMSD(ns)",
        ylabel: str = "Rg(ns)",
        zlabel: str = "K/mol",
        IP_value: int = 1,
        font_size: int = 36,
        font=None,
        text_location=(-0.10, 1.1),
        labelpad=15,
        pad=0.15,
):
    """"""
    font = load_style(font_size=font_size) if font is None else font
    ax.text2D(
        x=text_location[0], y=text_location[1], s=title,
        transform=ax.transAxes,
        fontproperties=font, fontsize=font_size + 2, ha='left', va='center')

    if data is None:
        raise ValueError("data 为 None")
    ip_func: interp2d = interp2d(data[0], data[1], data[2], kind="linear")
    x_new = np.linspace(np.min(data[0]), np.max(data[0]), IP_value * len(data[0]))
    y_new = np.linspace(np.min(data[1]), np.max(data[1]), IP_value * len(data[1]))

    img_new = ip_func(x_new, y_new)
    x_new, y_new = np.meshgrid(x_new, y_new)
    img_new = img_new.reshape(len(x_new), len(y_new))

    surf = ax.plot_surface(
        x_new,
        y_new,
        img_new,
        alpha=0.9,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
    )
    z_min = np.min(img_new)

    ax.contourf(
        x_new,
        y_new,
        img_new,
        zdir="z",
        offset=z_min,
        cmap="coolwarm",
    )

    ax.set_xlim(min(data[0]), max(data[0]))
    ax.set_ylim(min(data[1]), max(data[1]))

    ax.set_zlim(z_min, max(img_new[0]))
    ax.zaxis.set_major_locator(AutoLocator())
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.tick_params(axis='x', labelsize=font_size-2)
    ax.tick_params(axis='y', labelsize=font_size-2)
    ax.tick_params(axis='z', labelsize=font_size-2)

    ax.set_xlabel(xlabel, fontsize=font_size, fontproperties=font, labelpad=labelpad)
    ax.set_ylabel(ylabel, fontsize=font_size, fontproperties=font, labelpad=labelpad)
    ax.set_zlabel(zlabel, fontsize=font_size, fontproperties=font, labelpad=labelpad)

    return surf


# -------------------------------- 测试/执行 ---------------------------------
if __name__ == '__main__':
    t_dir_ = r"E:\Kinase_mapping\jupyter_draw\new_traj"
    traj_dir_list_ = os.listdir(t_dir_)
    traj_dir_path_list_ = [os.path.join(t_dir_, i_) for i_ in traj_dir_list_]
    traj_pdb_path_dict_ = {}
    for index_, traj_dir_path_ in enumerate(traj_dir_path_list_):
        dir_list_ = os.listdir(traj_dir_path_)
        dir_path_list_ = [os.path.join(traj_dir_path_, j_) for j_ in dir_list_]
        pdb_file_ = None
        xtc_file_list = []
        for path_ in dir_path_list_:
            if '.pdb' in os.path.basename(path_):
                pdb_file_ = path_
            elif '.xtc' in os.path.basename(path_):
                xtc_file_list.append(path_)
        traj_pdb_path_dict_[traj_dir_list_[index_]] = {
            'pdb': pdb_file_,
            'xtc': xtc_file_list
        }


    # print(traj_pdb_path_dict_)
    # rg_list = []
    # rmsd_list = []
    # rmsf_list = []
    # rg_dict_ = {}
    # rmsd_dict_ = {}
    # rmsf_dict_ = {}
    # dict_ = {}
    # start_index0_ = 1
    # for key, items in tqdm(traj_pdb_path_dict_.items(), total=len(traj_pdb_path_dict_), desc='获取rg和rmsd数据'):
    #     print(key, items['xtc'])
    #     rg_list_ = []
    #     rmsd_list_ = []
    #     rmsf_list_ = []
    #     for index1_, xtc_file_ in enumerate(items['xtc']):
    #         traj_object_ = md.load(xtc_file_, top=items['pdb'])
    #         reference = traj_object_[0]
    #         # atom_indices = md.compute_contacts(traj_object_)[0].flatten().tolist()
    #         atom_indices = traj_object_.topology.select('protein and name CA')
    #         rg_ = md.compute_rg(traj_object_)[start_index0_:]
    #         rmsd_ = md.rmsd(traj_object_, traj_object_[0])[start_index0_:]
    #         rmsf_ = md.rmsf(traj_object_, reference=reference, atom_indices=atom_indices)[start_index0_:]
    #         rg_list.append(rg_)
    #         rmsd_list.append(rmsd_)
    #         rmsf_list.append(rmsf_)
    #         rmsf_list_.append(rmsf_.tolist())
    #         rg_list_.extend(rg_.tolist())
    #         rmsd_list_.extend(rmsd_.tolist())
    #     rmsf_array_ = np.array(rmsf_list_)
    #     rmsf_averages_ = mean(rmsf_array_, axis=0)
    #
    #     # print("rg计算结果:\n", rg_list_)
    #     # print("rmsd计算结果:\n", rmsd_list_)
    #     rg_dict_[key] = rg_list_
    #     rmsd_dict_[key] = rmsd_list_
    #     rmsf_dict_[key] = rmsf_averages_
    #     dict_[key + '_rg'] = rg_list_
    #     dict_[key + '_rmsd'] = rmsd_list_
    # for key, items in rg_dict_.items():
    #     print(f'rg_dict_:[{key}]长度[{len(items)}]')
    # rg_dict_ = pd.DataFrame(rg_dict_)
    # rmsd_dict = pd.DataFrame(rmsd_dict_)
    # rmsf_dict_ = pd.DataFrame(rmsf_dict_)
    # dict_ = pd.DataFrame(rg_dict_)
    # rg_dict_.to_csv('new_rg.csv')
    # rmsd_dict.to_csv('new_rmsd.csv')
    # rmsf_dict_.to_csv('new_rmsf.csv')
    # dict_.to_csv('new_results.csv')


    # rg_data_ = pd.read_csv('new_rg.csv', index_col='Unnamed: 0')
    # rmsd_data_ = pd.read_csv('new_rmsd.csv', index_col='Unnamed: 0')
    # rmsf_data_ = pd.read_csv('new_rmsf.csv', index_col='Unnamed: 0')
    rg_data_ = pd.read_excel(r"rg data path")
    rmsd_data_ = pd.read_excel(r"rmsd data path")
    rmsf_data_ = pd.read_csv(r"rmsf data path", index_col='Unnamed: 0')

    rg_data_ = rg_data_ * 10  
    rmsd_data_ = rmsd_data_ * 10

    rg_data_dict = dict(rg_data_)
    rmsd_data_dict = dict(rmsd_data_)
    rmsf_data_dict = dict(rmsf_data_)

    print("rmsd['J_DT'] data:", rmsd_data_dict['J_DT'])
    print("rmsd['J_DZ'] data:", rmsd_data_dict['J_DZ'])
    print("data list:", rmsf_data_dict.keys())
    print("data list:", rmsf_data_dict['J_DZ'])
    # print("rg_data_dict =", rg_data_dict['J_DZ'])
    # print("rmsd_data_dict =", rmsd_data_dict['J_DZ'])
    # print("rmsf_data_dict =", rmsf_data_dict['J_DZ'])
    font_ = load_style()
    VIM_MIN = [999, 999]
    VIM_MAX = [-999, -999]
    while 1:
        print("#", ">" * 30, "Energy Mapping", "<" * 30)
        print('\t\t\t\t', '1 Generate Rg-RMSD Plot')
        print('\t\t\t\t', '2 Generate RMSF Plot')
        print('\t\t\t\t', '3 Plot 2D Free Energy Landscape (FEL) Diagram ')
        print('\t\t\t\t', '4 Plot 3D Free Energy Landscape (FEL) Diagram ')
        print("#", ">" * 30, "Energy Mapping", "<" * 30)
        input_value = input("Please select a function:")
        key_list_ = ['J_DT', 'J_DZ', 'J_SJ-137', 'J_SJ-1216', 'J_SJ-2847', 'J_SJ-2909', 'J_SJ-5598']
        print("rg_data_dict.keys() =", rg_data_dict.keys())
        if not input_value.isdigit():
            print(" Please enter a number")
            continue
        if int(input_value) == 1:
            data_dict = {}
            start_index = 0
            img_list_ = []
            print("rmsd_data_dict =", rmsd_data_dict)
            for index__, key in tqdm(enumerate(key_list_), total=len(key_list_), desc='Generate Rg-RMSD Plot'):
                data_dict[key] = []
                for hist_list in compute_hist(
                    np.array([rmsd_data_dict[key][start_index:], rg_data_dict[key][start_index:]]),
                    bins=100,
                ): data_dict[key].append(hist_list)
                vim_min_x = np.min(rmsd_data_dict[key][start_index:])
                vim_min_y = np.min(rg_data_dict[key][start_index:])
                vim_max_x = np.max(rmsd_data_dict[key][start_index:])
                vim_max_y = np.max(rg_data_dict[key][start_index:])
                if VIM_MIN[0] > vim_min_x: VIM_MIN[0] = vim_min_x
                if VIM_MIN[1] > vim_min_y: VIM_MIN[1] = vim_min_y
                if VIM_MAX[0] < vim_max_x: VIM_MAX[0] = vim_max_x
                if VIM_MAX[1] < vim_max_y: VIM_MAX[1] = vim_max_y
                img_file = rg_rmsd(
                    rmsd_data=rmsd_data_dict[key][start_index:],
                    rg_data=rg_data_dict[key][start_index:],
                    data=data_dict[key],
                    font_size=32,
                    text_location=(0.05, 1.45),
                    #label=chr(index__+65)+".  "+key,
                    #label=key,
                    #label=f"{chr(index__ + 65)}.{title_mapping.get(key, key)}",
                    label=title_mapping.get(key, key), 
                    col_num=13,
                    sigma=1.5,
                    x_lim=(1.50, 3.45),
                    y_lim=(18.30, 19.45),
                    save_dir=r"save path",
                    is_show=True
                )
                img_list_.append(img_file)
            print("VIM_MIN =", VIM_MIN)
            print("VIM_MAX =", VIM_MAX)
            label_list = [chr(i_+65)+"."+"_".join(f.split('_')[-2]) for i_, f in enumerate(img_list_)]

            print("img_list_ =", img_list_)
            images = [Image.open(str(f_)) for f_ in img_list_]
            cmap_img_ = Image.open(r"color bar path")
            merge_images(
                images=img_list_,
                output_path=r"output path",
                cols=4,
            )
            horizontal_image = h_concat(images=images, max_row=2, cmap_img=cmap_img_)
            horizontal_image.save('new_rg_rmsd.png')
        elif int(input_value) == 2:
            nrow_ = len(rmsf_data_dict)-2
            fig_, axs_ = plt.subplots(
                nrows=nrow_, ncols=1,
                # width_ratios=[5, 1, 1], height_ratios=[1, 5],
                figsize=(10, 2 * nrow_), dpi=600)
            key_list_ = list(rmsf_data_dict.keys())
            data_dict = {
                key_list_[6]: rmsf_data_dict[key_list_[6]],
                key_list_[0]: rmsf_data_dict[key_list_[0]],
                key_list_[5]: rmsf_data_dict[key_list_[5]],
                key_list_[2]: rmsf_data_dict[key_list_[2]],
                key_list_[1]: rmsf_data_dict[key_list_[1]],
                key_list_[3]: rmsf_data_dict[key_list_[3]],
                key_list_[4]: rmsf_data_dict[key_list_[4]],
            }
            new_key_list_ = list(data_dict.keys())
            colors = ['r', 'b', 'g', '#5db4b4', '#724e7c']
            linewidth_ = 1.3
            tag_dict = {
                "α": [
                    (48, 60),
                    (90, 96),
                    (103, 123),
                    (184, 200),
                    (211, 220),
                    (232, 242),
                    (252, 264),
                ],
                "β":[
                    (11, 20),
                    (23, 30),
                    (34, 40),
                    (70, 73),
                    (80, 83),
                    (135, 138),
                    (141, 145),
                ]
            }
            for index3_ in range(nrow_):
                rmsf(
                    ax=axs_[index3_],
                    data=data_dict[new_key_list_[0]],
                    label=new_key_list_[0],
                    color='#c1c1c0',
                    font=font_,
                    linewidth=linewidth_,
                    Alpha=1,
                )
                rmsf(
                    ax=axs_[index3_],
                    data=data_dict[new_key_list_[1]],
                    label=new_key_list_[1],
                    color='#cc9966',
                    font = font_,
                    linewidth=linewidth_,
                    Alpha=1,
                )
                rmsf(
                    ax=axs_[index3_],
                    data=data_dict[new_key_list_[2+index3_]],
                    label=new_key_list_[2+index3_],
                    color=colors[index3_],
                    linewidth=linewidth_,
                    font=font_,
                )
                colors_ = ['r', 'b']
                for number_, items in enumerate(tag_dict.values()):
                    for item in items:
                        axs_[index3_].axvspan(item[0], item[1], color=colors_[number_], alpha=0.1)

            fig_.subplots_adjust(wspace=0.1, hspace=0.5)
            plt.savefig('new_rmsf.png')
            plt.show()
            pass
        elif int(input_value) == 3:
            key_list_ = list(rmsd_data_dict.keys())
            data_dict = {
                0: [key_list_[4], rmsd_data_dict[key_list_[4]], rg_data_dict[key_list_[4]]],
                1: [key_list_[0], rmsd_data_dict[key_list_[0]], rg_data_dict[key_list_[0]]],
                2: [key_list_[3], rmsd_data_dict[key_list_[3]], rg_data_dict[key_list_[3]]],
                3: [key_list_[1], rmsd_data_dict[key_list_[1]], rg_data_dict[key_list_[1]]],
                4: [key_list_[2], rmsd_data_dict[key_list_[2]], rg_data_dict[key_list_[2]]],
            }
            new_key_list_ = list(data_dict.keys())
            nrow_ = 2
            ncols_ = math.ceil(len(new_key_list_)/nrow_)
            fig_, axs_ = plt.subplots(
                nrows=nrow_, ncols=ncols_,
                figsize=(10 * ncols_, 10 * nrow_), dpi=600)
            index__ = 0
            im_ = None
            for row_ in range(nrow_):
                for col_ in range(ncols_):
                    if index__ < len(new_key_list_):
                        data_ = calculate_free_energy(
                            rmsdValues=data_dict[index__][1],
                            rgValues=data_dict[index__][2],
                            temperature=300)
                        im_ = fel_2d(
                            ax=axs_[row_, col_],data=data_,
                            title=chr(index__ + 65) + ".  " + data_dict[index__][0], IP_value=12,
                            font_size=36, labelpad=10,
                        )
                        index__ += 1
                    else:
                        axs_[row_, col_].axis('off')

            plt.subplots_adjust(
                wspace=0.5, hspace=0.5
            )

            cbar = fig_.colorbar(im_, ax=axs_.ravel().tolist(), orientation='vertical', shrink=0.6, aspect=12)
            cbar.set_label('K/mol', fontsize=36)
            cbar.ax.tick_params(labelsize=36-2)  

            plt.savefig('FEL_2D.png')
        elif int(input_value) == 4:
            key_list_ = list(rmsd_data_dict.keys())
            data_dict = {
                0: [key_list_[4], rmsd_data_dict[key_list_[4]], rg_data_dict[key_list_[4]]],
                1: [key_list_[0], rmsd_data_dict[key_list_[0]], rg_data_dict[key_list_[0]]],
                2: [key_list_[3], rmsd_data_dict[key_list_[3]], rg_data_dict[key_list_[3]]],
                3: [key_list_[1], rmsd_data_dict[key_list_[1]], rg_data_dict[key_list_[1]]],
                4: [key_list_[2], rmsd_data_dict[key_list_[2]], rg_data_dict[key_list_[2]]],
            }
            new_key_list_ = list(data_dict.keys())
            nrow_ = 2
            ncols_ = math.ceil(len(new_key_list_)/nrow_)
            fig_ = plt.figure(figsize=(10 * ncols_, 10 * nrow_), dpi=600)

            index__ = 0
            im_ = None
            axs_ = []
            for row_ in range(nrow_):
                for col_ in range(ncols_):
                    if index__ < len(new_key_list_):
                        data_ = calculate_free_energy(
                            rmsdValues=data_dict[index__][1],
                            rgValues=data_dict[index__][2],
                            temperature=300)
                        ax_ = fig_.add_subplot(nrow_, ncols_, index__ + 1, projection='3d')
                        im_ = fel_3d(
                            ax=ax_,data=data_,
                            title=chr(index__ + 65) + ".  " + data_dict[index__][0], IP_value=12,
                            font_size=36, labelpad=25,
                        )
                        axs_.append(ax_)
                        index__ += 1
                    else:
                        pass

            plt.subplots_adjust(

                wspace=0.5, hspace=0.5
            )
            cbar = fig_.colorbar(im_, ax=axs_, orientation='vertical', shrink=0.6, aspect=12)
            cbar.set_label('K/mol', fontsize=36)
            cbar.ax.tick_params(labelsize=36-2)  

            plt.savefig('FEL_3D.png')
        elif int(input_value) == 0:
            break

