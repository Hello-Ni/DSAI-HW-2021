import torch

# create two sample vectors
inps = torch.randn([64, 161, 1])
d = torch.randn([64, 161])

# bring d into the same format, and then concatenate tensors
print(d.unsqueeze(2).shape)
new_inps = torch.cat((inps, d.unsqueeze(2)), dim=-1)
print(new_inps.shape)  # [64, 161, 2]
