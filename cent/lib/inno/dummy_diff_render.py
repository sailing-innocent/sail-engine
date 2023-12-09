import sys 
from innopy import DummyDiffRenderApp

import torch 
import torch.nn as nn 

class _DummyDiffRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source_img, height, width, app):
        result_img = torch.zeros((height, width, 3), dtype=torch.float32).cuda()
        app.forward(height, width, source_img.contiguous().data_ptr(), result_img.contiguous().data_ptr())
        ctx.app = app 
        ctx.height = height
        ctx.width = width
        return result_img

    @staticmethod
    def backward(ctx, dL_dtpix):
        height, width = ctx.height, ctx.width
        # print("dL_dtpix")
        # print(dL_dtpix)
        dL_dspix = torch.zeros((height, width, 3), dtype=torch.float32).cuda()
        ctx.app.backward(height, width, 
                         dL_dtpix.contiguous().data_ptr(), 
                         dL_dspix.contiguous().data_ptr())
        # print("dL_dspix")
        # print(dL_dspix)

        return dL_dspix, None, None, None 

    
class DummyDiffRender(nn.Module):
    def __init__(self):
        super(DummyDiffRender, self).__init__()
        self.app = DummyDiffRenderApp()
        self.app.create(sys.path[-1], "cuda")

    def forward(self, source_img, height, width):
        return _DummyDiffRender.apply(source_img, height, width, self.app)


