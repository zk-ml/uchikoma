import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
from artbench import ArtBench10

from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torch.quantization.qconfig import QConfig
from torch.quantization import QuantStub, DeQuantStub

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.contrib import graph_executor

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.quantized.engine = 'fbgemm'

from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant, is_expr, rewrite, DFPatternCallback

class UpsampleCallback(DFPatternCallback):
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.x = wildcard()
        self.scale1 = is_constant()
        self.scale2 = is_constant()
        self.zero1 = is_constant()
        self.zero2 = is_constant()
        subtract = is_op("subtract")(is_op("cast")(self.x), self.zero1)
        upsample = is_op("round")(is_op("divide")(
            is_op("image.resize2d")(is_op("multiply")(is_op("cast")(subtract), self.scale1)) \
            , self.scale2))
        self.pattern = is_op("cast")(is_op("clip")(is_op("add")(upsample, self.zero2)))

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        #offset = node_map[self.zero2][0]
        #offset = relay.const(offset.data.numpy().astype("int32"), dtype="int32")

        curr_shape = (x._checked_type_.shape)
        size = (curr_shape[-2]*2, curr_shape[-1]*2)
        x = relay.image.resize2d(x, size=size, roi=[0.0, 0.0, 0.0, 0.0], method="nearest_neighbor", \
            coordinate_transformation_mode="asymmetric", rounding_method="", cubic_alpha=-0.75)
        
        #offset = x + offset
        clipped = relay.op.clip(x, a_min=0, a_max=255)
        return relay.op.cast(clipped, dtype="uint8")

#from torch.ao.quantization.observer import (
#    MovingAverageMinMaxObserver,
#)
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    #default_fused_wt_fake_quant,
)
from integer_only_observer import MovingAverageIntegerMinMaxObserver

#from fid_score import calculate_fid_given_paths

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=16, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

HIDDEN_DIM = 1
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Linear(opt.latent_dim, HIDDEN_DIM * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            #nn.BatchNorm2d(HIDDEN_DIM),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(1, 0.8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(HIDDEN_DIM, 0.8),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_DIM, opt.channels, 3, stride=1, padding=1),
        )

        self.tanh = nn.Tanh()

    def forward(self, z):
        z = self.quant(z)

        out = self.l1(z)
        out = out.view(out.shape[0], HIDDEN_DIM, self.init_size, self.init_size)
        img = self.conv_blocks(out)

        img = self.dequant(img)

        img = self.tanh(img)
        return img

# Initialize generator
generator = Generator()

qconfig = QConfig(activation=FakeQuantize.with_args(
                            observer=MovingAverageIntegerMinMaxObserver,
                            quant_min=0,
                            quant_max=255,
                            reduce_range=True),
                  weight=FakeQuantize.with_args(
                            observer=MovingAverageIntegerMinMaxObserver,
                            quant_min=-128,
                            quant_max=127,
                            dtype=torch.qint8,
                            qscheme=torch.per_tensor_symmetric
                ))
generator.qconfig = qconfig

generator.eval()
    
# fuse the activations to preceding layers, where applicable
# this needs to be done manually depending on the model architecture
generator = torch.quantization.fuse_modules(generator,
   [["conv_blocks.1", "conv_blocks.2", "conv_blocks.3"]
   ,["conv_blocks.5", "conv_blocks.6", "conv_blocks.7"]]
)

generator.train()

# Prepare the model for QAT. This inserts observers and fake_quants in
# the model that will observe weight and activation tensors during calibration.
generator = torch.quantization.prepare_qat(generator)

generator.load_state_dict(torch.load('generator.pt', map_location=torch.device('cpu')))
generator = generator.cpu()

# Convert to quantized model
torch.quantization.convert(generator, inplace=True)
print('QAT: Conversion done.')
print(generator)


# Configure data loader
os.makedirs("./data/artbench", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    ArtBench10(
        "./data/artbench",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=False,
    drop_last=True
)

imgs = list(enumerate(dataloader))[0][1]

dz = np.random.normal(0, 1, (64, opt.latent_dim))
z = torch.FloatTensor(dz)

script_module = torch.jit.trace(generator, example_inputs=[z]).eval()
script_result = script_module(z).numpy()
torch_result = generator(z).numpy()

print((script_result - torch_result).mean())

script_module.save("quantized_jit_generator.pt")

device = tvm.cpu()
target = "llvm"
input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shapes = [(input_name, (64, opt.latent_dim))]
mod, params = relay.frontend.from_pytorch(
    script_module, input_shapes, keep_quantized_weight=True
)
mod = relay.transform.InferType()(mod)
print(mod)
print("===========")
mod = relay.qnn.transform.CanonicalizeOps()(mod)
seq = tvm.transform.Sequential(
    [
        transform.CanonicalizeOps(),
        transform.InferType(),
        #transform.SimplifyInference(),
        transform.FoldConstant(),
        #transform.FoldScaleAxis(),
        #transform.SimplifyExpr(),
        #transform.FoldConstant(),
    ]
)
with tvm.transform.PassContext(opt_level=3):
    mod = seq(mod)
print(mod)
print("===========")
mod["main"] = rewrite(UpsampleCallback(), mod["main"])
mod = relay.transform.InferType()(mod)
print(mod)

json_str = mod.astext(show_meta_data=True) #tvm.ir.save_json(mod["main"])
with open("generator_tvm_ir.txt", "w") as fo:
    fo.write(json_str)
with open("generator_tvm.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))

with tvm.transform.PassContext(opt_level=1):
    func = relay.create_executor("graph", mod=mod, device=device, target=target).evaluate()


for i, (imgs, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
    # Sample noise as generator input
    z = np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))

    # Generate a batch of images
    input_dict = {input_name: z}
    # Generate a batch of images
    fake_imgs = func(**input_dict, **params).numpy()

    # print(fake_imgs.shape)
    for j in range(imgs.shape[0]):
        k = i * imgs.shape[0] + j
        save_image(torch.tensor(fake_imgs[j, :, :, :]), "./data/fid_eval/%d.png" % k)
        save_image(imgs.data[j], "./data/fid_real/%d.png" % k)

#fid = calculate_fid_given_paths(paths=('./data/fid_real', './data/fid_eval'), 
#        batch_size=256, device='cuda', dims=2048)
#print(f"FID after rewrite : {fid}")
