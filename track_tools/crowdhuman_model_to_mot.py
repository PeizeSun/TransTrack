import torch

pre_model = 'output_crowdhuman/checkpoint.pth'
aa = torch.load(pre_model)

keys_name = []
for key in aa.keys():
    keys_name.append(key)

for key in keys_name:
    if key != 'model':
        bb = aa.pop(key)
        
torch.save(aa, 'crowdhuman_final.pth')