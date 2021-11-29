# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

# Copied and modified from https://github.com/nerdyrodent/VQGAN-CLIP/blob/4f380fc8ebd2d1993b6603cb583db36d6101142b/generate.py
# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

import tempfile
from pathlib import Path
import os
import sys

import librosa
from PIL import ImageFile, Image
import cog
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import kornia.augmentation as K
import numpy as np
import wav2clip
from omegaconf import OmegaConf

sys.path.append("/repos/CLIP")
import clip

sys.path.append("/repos/taming-transformers")
from taming.models import vqgan


IMAGE_SIZE = 512
DISPLAY_FREQ = 10
CLIP_MODEL = "ViT-B/32"
VQGAN_CONFIG = "/checkpoints/vqgan_imagenet_f16_16384.yaml"
VQGAN_CHECKPOINT = "/checkpoints/vqgan_imagenet_f16_16384.ckpt"
NUM_CUTS = 32
CUT_POWER = 1.0
AUGMENTS = [["Af", "Pe", "Ji", "Er"]]
SAMPLE_RATE = 16000


class Predictor(cog.Predictor):
    def setup(self):
        torch.backends.cudnn.benchmark = False  # NR: True is a bit faster, but can lead to OOM. False is more deterministic.
        # torch.use_deterministic_algorithms(True)	# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.device = torch.device("cuda")

        jit = True if float(torch.__version__[:3]) < 1.8 else False
        self.perceptor = (
            clip.load(CLIP_MODEL, jit=jit)[0]
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )

        self.wav2clip_model = wav2clip.get_model()

        config = OmegaConf.load(VQGAN_CONFIG)
        # always assume it's VQModel
        assert config.model.target == "taming.models.vqgan.VQModel"

        self.vqgan_model = vqgan.VQModel(**config.model.params).to(self.device)
        self.vqgan_model.eval().requires_grad_(False)
        self.vqgan_model.init_from_ckpt(VQGAN_CHECKPOINT)
        del self.vqgan_model.loss

        cut_size = self.perceptor.visual.input_resolution
        self.make_cutouts = MakeCutouts(AUGMENTS, cut_size, NUM_CUTS, CUT_POWER)

        f = 2 ** (self.vqgan_model.decoder.num_resolutions - 1)
        toksX, toksY = IMAGE_SIZE // f, IMAGE_SIZE // f
        self.sideX, self.sideY = toksX * f, toksY * f

        self.z_min = self.vqgan_model.quantize.embedding.weight.min(dim=0).values[
            None, :, None, None
        ]
        self.z_max = self.vqgan_model.quantize.embedding.weight.max(dim=0).values[
            None, :, None, None
        ]

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    @cog.input(
        "audio",
        type=cog.Path,
        default=None,
        help="Input audio file",
    )
    @cog.input("seed", type=int, default=-1, help="Random seed, -1 for random")
    @cog.input(
        "iterations", type=int, min=1, max=300, default=200, help="Number of iterations"
    )
    @cog.input(
        "learning_rate",
        type=float,
        min=0.0001,
        max=0.5,
        default=0.1,
        help="Learning rate",
    )
    def predict(self, audio, seed, iterations, learning_rate):
        """Run a single prediction on the model"""

        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
            print(f"Using seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        audio, _ = librosa.load(str(audio), sr=SAMPLE_RATE)
        embed = torch.from_numpy(wav2clip.embed_audio(audio, self.wav2clip_model)).to(
            self.device
        )
        prompt = Prompt(embed, float(1.0), float("-inf")).to(self.device)

        # assume init_noise == "pixels"
        img = random_noise_image(IMAGE_SIZE, IMAGE_SIZE)
        pil_image = img.convert("RGB")
        pil_image = pil_image.resize((self.sideX, self.sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = self.vqgan_model.encode(pil_tensor.to(self.device).unsqueeze(0) * 2 - 1)
        z.requires_grad_(True)

        opt = optim.Adam([z], lr=learning_rate)

        tmpdir = Path(tempfile.mkdtemp())
        out_file = Path(tempfile.mkdtemp()) / "tmp.png"
        out_file.touch()

        out = None
        for i in range(iterations):
            # Training time
            opt.zero_grad(set_to_none=True)
            z_q = vector_quantize(
                z.movedim(1, 3), self.vqgan_model.quantize.embedding.weight
            ).movedim(3, 1)
            out = clamp_with_grad(self.vqgan_model.decode(z_q).add(1).div(2), 0, 1)
            iii = self.perceptor.encode_image(
                self.normalize(self.make_cutouts(out))
            ).float()

            loss = prompt(iii)

            if i % DISPLAY_FREQ == 0:
                print(f"Iteration {i}; loss: {loss.item():g}")
                progress_path = tmpdir / "progress.png"
                TF.to_pil_image(out[0].cpu()).save(str(progress_path))
                yield progress_path

            loss.backward()
            opt.step()

            # with torch.no_grad():
            with torch.inference_mode():
                z.copy_(z.maximum(self.z_min).minimum(self.z_max))

        out_path = tmpdir / "output.png"
        TF.to_pil_image(out[0].cpu()).save(str(out_path))

        return out_path


# NR: Testing with different intital images
def random_noise_image(w, h):
    random_image = Image.fromarray(
        np.random.randint(0, 255, (w, h, 3), dtype=np.dtype("uint8"))
    )
    return random_image


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


def vector_quantize(x, codebook):
    d = (
        x.pow(2).sum(dim=-1, keepdim=True)
        + codebook.pow(2).sum(dim=1)
        - 2 * x @ codebook.T
    )
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1.0, stop=float("-inf")):
        super().__init__()
        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        self.register_buffer("stop", torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return (
            self.weight.abs()
            * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
        )


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        (input,) = ctx.saved_tensors
        return (
            grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
            None,
            None,
        )


clamp_with_grad = ClampWithGrad.apply


class MakeCutouts(nn.Module):
    def __init__(self, augments, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow  # not used with pooling

        # Pick your own augments & their order
        augment_list = []
        for item in augments[0]:
            if item == "Ji":
                augment_list.append(
                    K.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7
                    )
                )
            elif item == "Sh":
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == "Gn":
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1.0, p=0.5))
            elif item == "Pe":
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == "Ro":
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == "Af":
                augment_list.append(
                    K.RandomAffine(
                        degrees=15,
                        translate=0.1,
                        shear=5,
                        p=0.7,
                        padding_mode="zeros",
                        keepdim=True,
                    )
                )  # border, reflection, zeros
            elif item == "Et":
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == "Ts":
                augment_list.append(
                    K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7)
                )
            elif item == "Cr":
                augment_list.append(
                    K.RandomCrop(
                        size=(self.cut_size, self.cut_size),
                        pad_if_needed=True,
                        padding_mode="reflect",
                        p=0.5,
                    )
                )
            elif item == "Er":
                augment_list.append(
                    K.RandomErasing(
                        scale=(0.1, 0.4),
                        ratio=(0.3, 1 / 0.3),
                        same_on_batch=True,
                        p=0.7,
                    )
                )
            elif item == "Re":
                augment_list.append(
                    K.RandomResizedCrop(
                        size=(self.cut_size, self.cut_size),
                        scale=(0.1, 1),
                        ratio=(0.75, 1.333),
                        cropping_mode="resample",
                        p=0.5,
                    )
                )

        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        # self.noise_fac = False

        # Uncomment if you like seeing the list ;)
        # print(augment_list)

        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []

        for _ in range(self.cutn):
            # Use Pooling
            cutout = (self.av_pool(input) + self.max_pool(input)) / 2
            cutouts.append(cutout)

        batch = self.augs(torch.cat(cutouts, dim=0))

        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch
