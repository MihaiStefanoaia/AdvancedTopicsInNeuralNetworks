from typing import Union
from torch import Tensor
import torch


def get_normal_tensors(x: Tensor) -> Union[Tensor, None]:
    mean  = x.norm(dim=(1,2)).mean()
    st_dev = x.norm(dim=(1,2)).std()
    idk = [torch.std(gradient).item() for gradient in x]
    print(idk)
    maska = x.norm(dim=(1,2)) <= mean + st_dev * 1.5
    maskb = x.norm(dim=(1,2)) >= mean - st_dev * 1.5
    print(maska)
    print(maskb)
    return x[maska & maskb]

a = get_normal_tensors(torch.rand((100, 10, 256)))
print(a.shape)