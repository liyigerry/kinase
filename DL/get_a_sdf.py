# -*- coding: utf-8 -*-
import os
import shutil

from rdkit import Chem
from rdkit.Chem import SDMolSupplier

from main_generate import find_files


def split_sdf(input_sdf_file, last_name, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # elif os.listdir(output_dir) == 0:
    #     pass
    # else:
    #     shutil.rmtree(output_dir)
    #     os.makedirs(output_dir)

    # 创建一个SDMolSupplier来读取SDF文件
    suppl = SDMolSupplier(input_sdf_file)

    # 遍历SDF文件中的所有分子
    for i, mol in enumerate(suppl):
        if mol is None:
            continue  # 如果分子对象为空，则跳过

        # 构建输出文件名
        output_sdf_file = os.path.join(output_dir, f'No_{i}_{last_name}.sdf')

        # 使用SDWriter来写入新的SDF文件
        writer = Chem.SDWriter(output_sdf_file)
        props = mol.GetPropsAsDict()
        writer.write(mol)
        writer.close()


# -------------------------------- 测试/执行 ---------------------------------
if __name__ == '__main__':
    directory_path = '/home/dldx/Projects/PocketFlow/complexes_sdf'
    extension = '.sdf'
    files = find_files(directory_path, extension)
    for file in files:
        dir_name = file.split('/')
        pdb_name = dir_name[6]
        lastname = pdb_name.split('-')[0]
        split_sdf(
            input_sdf_file=file,
            last_name=lastname,
            output_dir=os.path.join(os.path.dirname(file), 'sdf_list'))
    pass
# -------------------------------- 测试/执行 ---------------------------------
