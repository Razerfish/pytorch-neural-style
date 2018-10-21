import torch
from PIL import Image
import json

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

class json2args():
    def __init__(self, data):
        if data["subcommand"] == "eval":
            self.subcommand = "eval"
            self.content_image = str(data["content_image"])
            print(self.content_image)
            self.content_scale = float(data["content_scale"])
            self.output_image = str(data["output_image"])
            self.model = str(data["model"])
            self.cuda = int(data["cuda"])
            self.export_onnx = str(data["export_onnx"])
        else:
            self.subcommand = "train"
            self.epochs = int(data["self"])
            self.batch_size = int(data["batch_size"])
            self.dataset = str(data["dataset"])
            self.style_image = str(data["style_image"])
            self.save_model_dir = str(data["save_model_dir"])
            self.name = str(data["name"])
            self.checkpoint_model_dir = str(data["checkpoint_model_dir"])
            self.image_size = int(data["image_size"])
            self.style_size = int(data["style_size"])
            self.cuda = int(data["cuda"])
            self.seed = int(data["seed"])
            self.content_weight = float(data["content_weight"])
            self.style_weight = float(data["style_weight"])
            self.lr = float(data["lr"])
            self.log_interval = int(data["log_interval"])
            self.checkpoint_interval =  int(data["checkpoint_interval"])
            