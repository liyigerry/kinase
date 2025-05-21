import argparse
import os.path
from pathlib import Path

from main_generate import find_files
from pocket_flow import Ligand, Protein, SplitPocket


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", type=str, default="./pocket/Bpocket/4M15.pdb", required=False, help="蛋白结构[.PDB]受体文件")
    parser.add_argument("--ligand", type=str, default="./pocket/Bpocket/QWS.sdf", required=False, help="化合物配体文件[.SDF]")
    parser.add_argument("--outdir", type=str, default="./complexes", required=False, help="化合物配体文件[.SDF]")
    args = parser.parse_args()

    return args


def main():
    directory_path = '/home/dldx/Projects/PocketFlow/pocket'  # 替换为实际目录路径
    extension = '.pdb'
    pdb_list = find_files(directory_path, '.pdb')
    sdf_list = find_files(directory_path, '.sdf')
    all_list = []
    out_dir = './complexes'
    if not os.path.exists('./complexes'):
        os.mkdir(out_dir)
    for i in range(len(pdb_list)):
        all_list.append((pdb_list[i], sdf_list[i]))
        print(pdb_list[i], sdf_list[i])
        args = arguments()
        args.protein = pdb_list[i]
        args.ligand = sdf_list[i]
        args.outdir = './complexes' if args.outdir is None else args.outdir
        if not os.path.exists('./complexes'):
            os.mkdir(out_dir)
        pdb_name = os.path.basename(pdb_list[i]).replace('.pdb', '')
        sdf_name = os.path.basename(sdf_list[i]).replace('.sdf', '')
        out_path = os.path.join(args.outdir, f"{pdb_name}-{sdf_name}.pdb")
        # output_path = f"{Path(args.protein).parent}/pocket.pdb"
        pro = Protein(args.protein)
        lig = Ligand(args.ligand)
        dist_cutoff = 10
        pocket_block, _ = SplitPocket._split_pocket_with_surface_atoms(
            pro, lig, dist_cutoff
        )
        with open(out_path, "w") as f:
            f.write(pocket_block)


if __name__ == "__main__":
    main()
