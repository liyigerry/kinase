pocket: ./pocket/Epocket/3PYY.pdb
ckpt: ./ckpt/ZINC-pretrained-255000.pt
num_gen: 10000
name: 3PYY
device: cuda:0
atom_temperature: 1.0
bond_temperature: 1.0
max_atom_num: 40
focus_threshold: 0.5
choose_max: True
min_dist_inter_mol: 3.0
bond_length_range: (1.0, 2.0)
max_double_in_6ring: 0
with_print: False
root_path: gen_results
readme: None