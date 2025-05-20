# -*- coding: utf-8 -*-
# -------------------------------- 功能模块 ---------------------------------
__all__ = [
    # 属性
    '',

    # 方法
    '',

    # 类
    '',

    # 对象
    '',
]

import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, QED, Descriptors
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdDepictor import Compute2DCoords

from tools.sascorer import calculateScore, readFragmentScores


# -------------------------------- 方法 ---------------------------------
def get_score(mol_file):
    """
        获取分子分数
    :Param mol_file: 分子文件[.sdf]
    :return:
    """
    # 第二个参数sanitize=True表示在读取后对分子进行清理和标准化处理
    suppl = SDMolSupplier(mol_file)
    # 遍历文件中的每个分子
    smi = None
    quality = 0
    qed = 0
    sa = 0
    # 加载分数文件
    file_path = './tools/fpscores'
    sa_params = readFragmentScores(file_path)
    for mol in suppl:
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            # 计算分子质量
            # quality = MolWt(mol)
            quality = MolWt(mol)
            # 计算QED
            # qed = QED.qed(mol)
            qed = QED.qed(mol)
            # 计算SA
            m = Chem.MolFromSmiles(smi)
            m = Chem.AddHs(m)
            Compute2DCoords(m)
            # sa = calculateScore(m)
            sa = calculateScore(m)
    print("smi =", smi)
    print("quality =", quality)
    print("qed =", qed)
    print("sa =", sa)
    return smi, quality, qed, sa

    # -------------------------------- 方法 ---------------------------------


# -------------------------------- 类 ---------------------------------

# -------------------------------- 类 ---------------------------------


# -------------------------------- 对象 ---------------------------------

# -------------------------------- 对象 ---------------------------------


# -------------------------------- 测试/执行 ---------------------------------
if __name__ == '__main__':
    inputDir = "/home/dldx/Projects/PocketFlow/mol"
    files = os.listdir(inputDir)
    mol_dataFrame = pd.DataFrame(columns=['mol', 'smiles', 'quality', 'QED', 'SA'])
    mol_list = []
    smiles_list = []
    quality_list = []
    qed_list = []
    sa_list = []
    for file in files:
        f = os.path.basename(file).replace('.sdf', '')
        sm, qu, qe, sa = get_score(os.path.join(inputDir, file))
        mol_list.append(f)
        smiles_list.append(sm)
        quality_list.append(qu)
        qed_list.append(qe)
        sa_list.append(sa)
    mol_dataFrame['mol'] = mol_list
    mol_dataFrame['smiles'] = smiles_list
    mol_dataFrame['quality'] = quality_list
    mol_dataFrame['QED'] = qed_list
    mol_dataFrame['SA'] = sa_list
    mol_dataFrame.to_csv(os.path.join(inputDir, 'mol.csv'))
    pass
# -------------------------------- 测试/执行 ---------------------------------
