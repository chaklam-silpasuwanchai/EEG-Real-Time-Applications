import torch
import torch.nn as nn

output_DBG = False


class ParallelModule(nn.Sequential):
    def __init__(self, *args):
        super(ParallelModule, self).__init__(*args)

    def forward(self, x):
        output = []
        for module in self:
            output.append(module(x))

        if output_DBG:
            for eo in output:
                print(eo.shape)

        return torch.cat(output, dim=1)
