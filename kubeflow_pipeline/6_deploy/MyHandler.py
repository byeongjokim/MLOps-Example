import logging
import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
import faiss
import json
import numpy as np

class MyHandler(BaseHandler):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.face_index, self.face_label = self.load_nn_model(
            "./faiss_index.bin",
            "./faiss_label.json"
        )

    def load_nn_model(self, faiss_model_path, faiss_label_path):
        face_index = faiss.read_index(faiss_model_path)
        
        with open(faiss_label_path, "r") as l:
            face_label = json.load(l)
        
        return face_index, np.asarray(face_label)

    def preprocess_one_image(self, req):
        image = req.get("data")
        if image is None:
            image = req.get("body")

        image = np.asarray(Image.open(io.BytesIO(image)).convert("L"))
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.unsqueeze(0).float()
        return image
    
    def preprocess(self, requests):
        images = [self.preprocess_one_image(req) for req in requests]
        images = torch.cat(images)
        return images

    def inference(self, x):
        embeddings = self.model.forward(x)
        
        dists, inds = self.face_index.search(embeddings.detach().cpu().numpy(), 3)
        
        dists = dists[:,0]
        preds = self.face_label[inds[:,0]]

        return dists.tolist(), preds.tolist()
    
    def postprocess(self, predictions):
        dists, preds = predictions

        res = []
        for dist, pred in zip(dists, preds):
            res.append({"dist":dist, "pred":pred})

        return res