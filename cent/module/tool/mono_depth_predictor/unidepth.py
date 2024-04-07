import torch 

class MonoDepthPredictor:
    def __init__(self, device="cuda"):
        self.model = torch.hub.load(
            "lpiccinelli-eth/unidepth",
            "UniDepthV1_ViTL14",
            pretrained=True,
            trust_repo=True,
            # force_reload=True,
        )
        self.model = self.model.to(device)
        self.model.eval()

    def predict(self, rgb, intr):
        predictions = self.model.infer(rgb, intr)
        depth_pred = predictions["depth"].squeeze()

        return depth_pred
    