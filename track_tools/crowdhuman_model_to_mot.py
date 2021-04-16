import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pre_model', default='output_crowdhuman/checkpoint.pth', type=str)
parser.add_argument('--saved_model', default='crowdhuman_final.pth', type=str)
args = parser.parse_args()

pre_model = args.pre_model
aa = torch.load(pre_model)

keys_name = []
for key in aa.keys():
    keys_name.append(key)

for key in keys_name:
    if key != 'model':
        bb = aa.pop(key)
        
torch.save(aa, args.saved_model)
