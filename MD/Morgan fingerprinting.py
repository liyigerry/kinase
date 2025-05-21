import argparse
import os
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import SubplotParams
from matplotlib.gridspec import GridSpec
import seaborn as sns
from matplotlib.font_manager import FontProperties
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from typing import (
    Any,
    Tuple,
    Optional,
    Dict,
    List
)

# ============================== Hyperparameter Module ==================================
class superArges:
    def __init__(
        self,
        figsize: Tuple = (20, 16),
        dpi: int = 600,
        nrows: int = 1,
        ncols: int = 1,
        title: str = None,
        xlabels: List = None,
        ylabels: List = None,
        labels: List = None,
        fontsize: int = 12,
        colors: List = None,
        font: FontProperties = None,
        save_path: str = "output.png"
    ):
        fig_number = nrows * ncols
        if not xlabels:
            xlabels = [None] * fig_number
        elif isinstance(xlabels, str):
            xlabels = [xlabels] * fig_number
        elif isinstance(xlabels, list) and len(xlabels) == 1:
            xlabels = xlabels * fig_number
        if not ylabels:
            ylabels = [None] * fig_number
        elif isinstance(ylabels, str):
            ylabels = [ylabels] * fig_number
        elif isinstance(ylabels, list) and len(ylabels) == 1:
            ylabels = ylabels * fig_number
        if not labels:
            labels = [None] * fig_number
        elif isinstance(labels, str):
            labels = [labels] * fig_number
        elif isinstance(labels, list) and len(labels) == 1:
            labels = labels * fig_number

        self.figsize = figsize
        self.dpi = dpi
        self.nrows = nrows
        self.ncols = ncols
        self.xlabels = xlabels
        self.ylabels = ylabels
        self.labels = labels
        self.colors = colors
        self.fontsize = fontsize
        self.font = font
        self.save_path = save_path

# ============================== Hyperparameter Module ==================================

# ============================== Data Module ==================================
excel_path = r"../jupyter_draw/J QED 筛选结果.xlsx"
new_excel_path = r"../jupyter_draw/J_FPZ.xlsx"
figsize_ = (20, 16)
dpi_ = 600
nrows_ = 1
ncols_ = 1
title_ = None
xlabels_ = None
ylabels_ = None
labels_ = None
fontsize_ = 18
colors_ = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'pink', 'gray', 'orange', 'purple', 'brown', 'olive', 'teal']
COLOR = "#000000"
n_clusters = 25
custom_labels_ = []
for i in range(n_clusters):
    custom_labels_.append(f'{i + 1}')
# ============================== Data Module ==================================
    

# ============================== Object Module ==================================
class DiagramObjects:
    def __init__(
            self,
            data: Any = None,
            # 0 图形参数
            figsize: Tuple[float, float] = (20, 16),
            dpi: int = 100,
            facecolor: str = 'white',
            edgecolor: str = 'white',
            linewidth: float = 0.2,
            frameon: bool = True,
            subplotpars: Optional[SubplotParams] = None,
            tight_layout: Optional[bool] = True,
            # constrained_layout: Optional[bool] = False,
            layout: Optional[str] = None,
            
            fontsize: int = 18,
            title: str = None,
            xlabels: str = None,
            ylabels: str = None,
            
            figure: Any = None,
            nrows: int = 1,
            ncols: int = 1,
            top: Optional[float] = None,
            bottom: Optional[float] = None,
            left: Optional[float] = None,
            right: Optional[float] = None,
            wspace: Optional[float] = None,
            hspace: Optional[float] = None,
            width_ratios: Any = None,
            height_ratios: Any = None,
            polar: bool = False,
            projection: str = None,
            sharex: Any = None,
            sharey: Any = None,
            labels: List = None,
            
            font: FontProperties = None
    ):
        if not labels:
            labels = [None] * nrows * ncols
        if not xlabels:
            xlabels = [None] * nrows * ncols
        elif isinstance(xlabels, str):
            xlabels = [xlabels] * nrows * ncols
        elif isinstance(xlabels, list) and len(xlabels) == 1:
            xlabels = xlabels * nrows * ncols
        if not ylabels:
            ylabels = [None] * nrows * ncols
        elif isinstance(ylabels, str):
            ylabels = [ylabels] * nrows * ncols
        elif isinstance(ylabels, list) and len(ylabels) == 1:
            ylabels = ylabels * nrows * ncols

        self.fontsize = fontsize
        self.plt = plt
        self.figure = self.plt.figure(
            figsize=figsize,
            dpi=dpi,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            frameon=frameon,
            subplotpars=subplotpars,
            tight_layout=tight_layout,
            # constrained_layout=constrained_layout,
            layout=layout,
        )
        self.gridspec = GridSpec(
            nrows=nrows,
            ncols=ncols,
            figure=self.figure,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            wspace=wspace,
            hspace=hspace,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
        self.ax = {}
        ax_ = None
        for i in range(nrows):
            sharex = ax_ if sharex else None
            sharey = ax_ if sharey else None
            for j in range(ncols):
                self.ax[(i, j)] = self.figure.add_subplot(
                    self.gridspec[i, j],
                    polar=polar,
                    projection=projection,
                    sharex=sharex,
                    sharey=sharey,
                    label=labels[i],
                    facecolor=facecolor,
                )
        self.set_font_sizes(title=title, xlabels=xlabels, ylabels=ylabels, font=font)

    def set_font_sizes(self, title: str = None, xlabels: List = None, ylabels: List = None, font: FontProperties = None):
        if not xlabels:
            xlabels = [None] * len(self.ax.items())
        if not ylabels:
            ylabels = [None] * len(self.ax.items())

        
            self.figure.suptitle(title, fontsize=32, fontproperties=font)

        index_number = 0
        for (i, j), ax in self.ax.items():
            ax.set_xlabel(xlabels[index_number], fontsize=self.fontsize, fontproperties=font)
            ax.set_ylabel(ylabels[index_number], fontsize=self.fontsize, fontproperties=font)

            
            ax.tick_params(axis='x', labelsize=30)
            ax.tick_params(axis='y', labelsize=30)


            if index_number < len(self.ax.items()) - 1:
                index_number += 1

# ============================== Object Module ==================================


def load_style(font_path: str = r"E:\Kinase_mapping\jupyter_draw\Times New Roman.ttf"):
    try:
        plt.style.use("chartlab.mplstyle")
    except:
        pass
    font = FontProperties(fname=font_path)
    # 设置字体为SimHei显示中文
    plt.rcParams['font.sans-serif'] = [font.get_name()]  # 用来正常显示中文标签[黑体]
    plt.rcParams["axes.unicode_minus"] = False
    # 设置全局字体颜色
    plt.rcParams['text.color'] = COLOR  # 设置所有文本的默认颜色为红色
    plt.rcParams['axes.labelcolor'] = COLOR  # 设置坐标轴标签的颜色为红色
    plt.rcParams['xtick.color'] = COLOR  # 设置x轴刻度标签的颜色为红色
    plt.rcParams['ytick.color'] = COLOR
    return font


def compute_mol_score(
        excel_path: str,
        n_components: int = 2,
        n_clusters: int = 15,
        random_state: int = 42,
        n_features: int = 1024,
        perplexity: int = 30,  # 默认 perplexity 值
):
    
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        raise ValueError(f"读取 Excel 文件失败: {e}")

    
    smiles_column_name = 'smiles'
    if smiles_column_name not in df.columns:
        raise ValueError(f"Excel 文件中未找到列 {smiles_column_name}")

    smiles_list = df[smiles_column_name].tolist()
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list if isinstance(smiles, str)]
    mols = [mol for mol in mols if mol is not None]
    
    fingerprints_mols = [GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_features) for mol in mols]
    fp_array_mols = np.array([list(fp) for fp in fingerprints_mols])

    
    if len(fp_array_mols) < 2:
        return fp_array_mols, [0] * len(fp_array_mols)  

    
    n_samples = len(fp_array_mols)
    if perplexity >= n_samples:
        perplexity = max(1, n_samples - 1)  

    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    fp_tsne = tsne.fit_transform(fp_array_mols)

    
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_components, random_state=random_state)
    kmeans.fit(fp_tsne)  
    labels = kmeans.labels_  
    return fp_tsne, labels

def morgana_plot(ax, data, new_data=None, text_location=(-0.10, 1.1), labels=None, custom_labels=None, fontsize=18, name='C.'):
    
    font = load_style()
    
    
    if custom_labels is None:
        custom_labels = []
        for i in range(len(data)):
            custom_labels.append(f'{i + 1}')  
    

    color_palette = [
        '#FF7F50',  
        '#FF00CC',  
        '#FF9966',  
        '#00FF7F',  
        '#9933FF',  
        '#BF8FED',  
        '#FF4500',  
        '#FF007F',  
        '#228B22',  
        '#FF4C4C',  
        '#FF7518',  
        '#90EE90',  
        '#EE82EE',  
        '#FF91A4',  
        '#FF69B4',  
        '#39FF14',  
        '#FFBC52',  
        '#66B3FF',  
        '#9ACD32',  
        '#B088FF',  
        '#FFB6C1',  
        '#FFD700',  
        '#00FFCC',  
        '#0000FF',  
        '#00BFFF',  
    ]
    
    
    scatter = sns.scatterplot(
        x=data[:, 0], y=data[:, 1],
        hue=labels,
        palette=color_palette,
        legend='full', ax=ax,s=100
    )
    
    
    ax.set_box_aspect(1)
    handles, _ = ax.get_legend_handles_labels()
    
    
    if new_data is not None:
        new_scatter = ax.scatter(
            new_data[:, 0], new_data[:, 1],
            color='red',
            marker='*',  
            s=600,  
            label='C12'  
        )
        
        new_handles, new_labels = ax.get_legend_handles_labels()
        new_handle = new_handles[-1]
        new_label = new_labels[-1]
        handles.insert(0, new_handle)  
        custom_labels.insert(0, new_label)  
    
    ax.set_xlabel('t-SNE 1', fontsize=30, fontproperties=font)
    ax.set_ylabel('t-SNE 2', fontsize=30, fontproperties=font)
    
    
    plt.subplots_adjust(
        bottom=0.2,
        wspace=0.5, hspace=0.1
    )
    
    return handles, custom_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Custom Image Save Path')
    parser.add_argument('--save_path', type=str, default=r"E:\Kinase_mapping\img\Morgan fingerprinting.png", help='Image Save Path')
    args = parser.parse_args()

    save_path_ = args.save_path
    arges = superArges(
        figsize=figsize_, dpi=dpi_, nrows=nrows_, ncols=ncols_,
        title=title_, xlabels=xlabels_, ylabels=ylabels_, labels=labels_,
        fontsize=fontsize_, colors=colors_,
        save_path=save_path_
    )
    arges.font = load_style()
    
    fp_tsne_, labels_ = compute_mol_score(excel_path=excel_path, n_clusters=n_clusters)

    new_df = pd.read_excel(new_excel_path)
    if len(new_df) == 1:
        smiles = new_df['smiles'].iloc[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fingerprint = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            new_fp_array = np.array([list(fingerprint)])
            new_fp_tsne_ = new_fp_array  
            new_labels_ = [0]  
        else:
            raise ValueError("新数据的 SMILES 格式无效")
    else:
        new_fp_tsne_, new_labels_ = compute_mol_score(excel_path=new_excel_path, n_clusters=n_clusters)


    img_ = DiagramObjects(
        figsize=arges.figsize,
        fontsize=arges.fontsize,
        dpi=arges.dpi,
        title=None,
        xlabels=arges.xlabels,
        ylabels=arges.ylabels,
        nrows=arges.nrows, ncols=arges.ncols,
    )
    
    handles, custom_labels = morgana_plot(
        ax=img_.ax[(0, 0)],
        data=fp_tsne_,
        new_data=new_fp_tsne_,  
        labels=labels_,
        custom_labels=custom_labels_
    )
    

    img_.ax[(0, 0)].legend(handles, custom_labels, loc=8, bbox_to_anchor=(0.5, -0.25), ncol=math.ceil(len(custom_labels)/3),prop={'size': 28},handletextpad=0.5, markerscale=1.5)
    img_.plt.savefig(arges.save_path, dpi=arges.dpi, bbox_inches='tight')  
    print("Image Save Location:", os.path.abspath(arges.save_path))