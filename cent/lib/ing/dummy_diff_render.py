import sys 
from ingpy import DummyDiffRender as DummyDiffRenderApp

import torch 
import torch.nn as nn 

class _DummyDiffRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source_img, height, width, app):
        result_img = torch.zeros((3, height, width), dtype=torch.float32).cuda()
        app.forward(
            source_img.contiguous().data_ptr(), 
            height, 
            width,  
            result_img.contiguous().data_ptr())
        ctx.app = app 
        ctx.height = height
        ctx.width = width
        return result_img

    @staticmethod
    def backward(ctx, dL_dtpix):
        dL_dspix = torch.zeros((3, ctx.height, ctx.width), dtype=torch.float32).cuda()
        ctx.app.backward(dL_dtpix.contiguous().data_ptr(), 
                         dL_dspix.contiguous().data_ptr())
        # print(dL_dspix)
        return dL_dspix, None, None, None 

class DummyDiffRender(nn.Module):
    def __init__(self):
        super(DummyDiffRender, self).__init__()
        self.app = DummyDiffRenderApp()

    def forward(self, source_img, height, width):
        return _DummyDiffRender.apply(source_img, height, width, self.app)


