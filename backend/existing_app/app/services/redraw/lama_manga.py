#!/usr/bin/env python3
"""
lama_manga.py - run mayocream/lama-manga (Big-LaMa fine-tuned on manga/anime) on CPU or GPU.

  pip install torch safetensors opencv-python numpy
  hf download mayocream/lama-manga lama-manga.safetensors --local-dir models

  # 1) check that the checkpoint matches this architecture (no image needed)
  python lama_manga.py --model models/lama-manga.safetensors --check

  # 2) erase whatever the mask marks (white / non-zero = fill)
  python lama_manga.py --model models/lama-manga.safetensors \
      --image 011.jpg --mask out/011_mask.png --out 011_lama.png

The network is the Big-LaMa FFCResNetGenerator (input RGB*(1-mask) + mask = 4 channels,
18 FFC residual blocks, sigmoid output). It is fully convolutional, so it runs at native
resolution; the image is processed in windows (--win) so big masks do not need the whole page
in memory and nothing is down-scaled to 512.

NOT verified against the real weights by the author of this file (no torch / network where it
was written). The tensor inventory implied by the code was checked against the published
numbers (989 tensors, ~204 MB fp32) but key names and the pre-processing were written from
the public LaMa design. Run --check first; if it prints missing/unexpected keys or the output
looks wrong, compare with config.json in the repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# =========================================================================== tiling (numpy only)
def inpaint_tiled(img: np.ndarray, mask: np.ndarray, fn, win: int = 512, margin: int = 48,
                  adaptive: bool = True, min_win: int = 128) -> np.ndarray:
    """Fill every pixel where mask>0, window by window, at native resolution.

    img  : HxWx3 uint8 (BGR)      mask : HxW (non-zero = fill)
    fn   : fn(crop_bgr_uint8, crop_mask_bool) -> crop_bgr_uint8   (the model)

    Each window is `win` px (or the whole image if smaller). Only pixels at least `margin` px
    away from a window edge are accepted (edges that touch the image border are exempt), the
    rest are left for the next window, which sees the already-filled pixels as context.

    adaptive: shrink the window to the pending mask extent (+ margin, multiple of 8, >= min_win).
    A 40 px mask then costs a ~136 px window instead of a full 512 px one (cost ~ window area).
    """
    H, W = mask.shape
    out = img.copy()
    todo = mask > 0
    guard = 0
    while todo.any():
        guard += 1
        if guard > 10_000:
            raise RuntimeError("tiling did not converge")
        ys, xs = np.nonzero(todo)
        y, xf = int(ys[0]), int(xs[0])                      # first pending pixel in raster order
        x = xf
        wx, wy = min(win, W), min(win, H)
        if adaptive:
            span = max(1, win - 2 * margin)                 # side of the area one full window can accept
            band = todo[y:y + span]
            cols = np.flatnonzero(band.any(axis=0))
            xa = int(cols[0])                               # leftmost pending column in this row band
            cols = cols[cols < xa + span]
            rows = np.flatnonzero(band[:, xa:xa + span].any(axis=1))
            bw, bh = int(cols[-1]) - xa + 1, int(rows[-1]) + 1
            r8 = lambda v: -(-v // 8) * 8
            wx = min(wx, max(min_win, r8(bw + 2 * margin)))
            wy = min(wy, max(min_win, r8(bh + 2 * margin)))
            x = xa
        x0 = int(min(max(0, x - margin), W - wx))
        y0 = int(min(max(0, y - margin), H - wy))
        x1, y1 = x0 + wx, y0 + wy
        res = fn(out[y0:y1, x0:x1], todo[y0:y1, x0:x1])
        ax0, ax1 = (x0 + margin if x0 > 0 else 0), (x1 - margin if x1 < W else W)
        ay0, ay1 = (y0 + margin if y0 > 0 else 0), (y1 - margin if y1 < H else H)
        accept = np.zeros((H, W), bool)
        accept[ay0:ay1, ax0:ax1] = True
        sel = todo & accept
        if not sel.any():                                   # safety: always make progress
            if adaptive:                                    # should not happen; fall back to fixed windows
                adaptive = False
                continue
            sel[y, xf] = True
        sub = sel[y0:y1, x0:x1]
        out[y0:y1, x0:x1][sub] = res[sub]
        todo &= ~sel
    return out


# =========================================================================== network (torch)
def build_network():
    """Big-LaMa generator. Layer indices inside `model` match the published checkpoint
    (model.1 .. model.34 after the 'model.generator.' prefix was stripped)."""
    import torch
    import torch.nn as nn

    class FourierUnit(nn.Module):
        def __init__(self, cin, cout, groups=1):
            super().__init__()
            self.conv_layer = nn.Conv2d(cin * 2, cout * 2, 1, 1, 0, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(cout * 2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            b = x.shape[0]
            f = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
            f = torch.stack((f.real, f.imag), dim=-1).permute(0, 1, 4, 2, 3).contiguous()
            f = f.view(b, -1, *f.shape[3:])
            f = self.relu(self.bn(self.conv_layer(f)))
            f = f.view(b, -1, 2, *f.shape[2:]).permute(0, 1, 3, 4, 2).contiguous()
            f = torch.complex(f[..., 0], f[..., 1])
            return torch.fft.irfftn(f, s=x.shape[-2:], dim=(-2, -1), norm="ortho")

    class SpectralTransform(nn.Module):
        def __init__(self, cin, cout, stride=1, groups=1):
            super().__init__()
            self.downsample = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()
            self.conv1 = nn.Sequential(nn.Conv2d(cin, cout // 2, 1, groups=groups, bias=False),
                                       nn.BatchNorm2d(cout // 2), nn.ReLU(inplace=True))
            self.fu = FourierUnit(cout // 2, cout // 2, groups)
            self.conv2 = nn.Conv2d(cout // 2, cout, 1, groups=groups, bias=False)

        def forward(self, x):
            x = self.conv1(self.downsample(x))
            return self.conv2(x + self.fu(x))               # enable_lfu=False in Big-LaMa

    class FFC(nn.Module):
        def __init__(self, cin, cout, k, rin, rout, stride=1, padding=0, dilation=1):
            super().__init__()
            icg = int(cin * rin); icl = cin - icg
            ocg = int(cout * rout); ocl = cout - ocg
            self.ratio_gout = rout
            self.global_in_num = icg
            conv = lambda a, b: nn.Identity() if (a == 0 or b == 0) else nn.Conv2d(
                a, b, k, stride, padding, dilation, 1, False, padding_mode="reflect")
            self.convl2l, self.convl2g, self.convg2l = conv(icl, ocl), conv(icl, ocg), conv(icg, ocl)
            self.convg2g = nn.Identity() if (icg == 0 or ocg == 0) else SpectralTransform(icg, ocg, stride)

        def forward(self, x):
            xl, xg = x if isinstance(x, tuple) else (x, 0)
            ol = og = 0
            if self.ratio_gout != 1:
                ol = self.convl2l(xl) + self.convg2l(xg)
            if self.ratio_gout != 0:
                og = self.convl2g(xl) + self.convg2g(xg)
            return ol, og

    class FFC_BN_ACT(nn.Module):
        def __init__(self, cin, cout, k, rin, rout, stride=1, padding=0, dilation=1):
            super().__init__()
            self.ffc = FFC(cin, cout, k, rin, rout, stride, padding, dilation)
            g = int(cout * rout)
            self.bn_l = nn.Identity() if rout == 1 else nn.BatchNorm2d(cout - g)
            self.bn_g = nn.Identity() if rout == 0 else nn.BatchNorm2d(g)
            self.act_l = nn.Identity() if rout == 1 else nn.ReLU(inplace=True)
            self.act_g = nn.Identity() if rout == 0 else nn.ReLU(inplace=True)

        def forward(self, x):
            xl, xg = self.ffc(x)
            return self.act_l(self.bn_l(xl)), self.act_g(self.bn_g(xg))

    class ResBlock(nn.Module):
        def __init__(self, dim, r):
            super().__init__()
            self.conv1 = FFC_BN_ACT(dim, dim, 3, r, r, padding=1)
            self.conv2 = FFC_BN_ACT(dim, dim, 3, r, r, padding=1)

        def forward(self, x):
            xl, xg = x if isinstance(x, tuple) else (x, 0)
            il, ig = xl, xg
            xl, xg = self.conv2(self.conv1((xl, xg)))
            return il + xl, ig + xg

    class Concat(nn.Module):
        def forward(self, x):
            xl, xg = x
            return xl if not torch.is_tensor(xg) else torch.cat((xl, xg), dim=1)

    class Generator(nn.Module):
        def __init__(self, inc=4, outc=3, ngf=64, nd=3, nb=18, maxf=1024, ratio=0.75):
            super().__init__()
            L = [nn.ReflectionPad2d(3), FFC_BN_ACT(inc, ngf, 7, 0, 0)]
            for i in range(nd):
                m = 2 ** i
                L.append(FFC_BN_ACT(min(maxf, ngf * m), min(maxf, ngf * m * 2), 3, 0,
                                    ratio if i == nd - 1 else 0, stride=2, padding=1))
            bott = min(maxf, ngf * 2 ** nd)
            L += [ResBlock(bott, ratio) for _ in range(nb)]
            L.append(Concat())
            for i in range(nd):
                m = 2 ** (nd - i)
                L += [nn.ConvTranspose2d(min(maxf, ngf * m), min(maxf, int(ngf * m / 2)), 3, 2, 1, output_padding=1),
                      nn.BatchNorm2d(min(maxf, int(ngf * m / 2))), nn.ReLU(True)]
            L += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, outc, 7), nn.Sigmoid()]
            self.model = nn.Sequential(*L)

        def forward(self, x):
            return self.model(x)

    return Generator()


class LamaManga:
    """Callable like the Lama class in manga_redraw.py: lama(image_bgr, mask) -> image_bgr."""

    def __init__(self, path: str, device: str = "cpu", win: int = 512, margin: int = 48, grow: int = 1,
                 adaptive: bool = True, threads: int | None = None):
        import torch
        from app.threading_utils import apply_torch_threads
        apply_torch_threads(threads or 0)   # một lần, dưới khoá chung
        from safetensors.torch import load_file
        self.torch, self.device = torch, torch.device(device)
        self.win, self.margin, self.grow, self.size = win, margin, grow, win
        self.adaptive = adaptive
        self.net = build_network()
        report = check_state(self.net, load_file(path))
        if report:
            raise RuntimeError("checkpoint does not match the network:\n" + report)
        self.net.load_state_dict(load_file(path), strict=True)
        self.net.eval().to(self.device)

    def _run(self, bgr: np.ndarray, m: np.ndarray) -> np.ndarray:
        torch = self.torch
        F = torch.nn.functional
        h, w = m.shape
        rgb = np.ascontiguousarray(bgr[..., ::-1]).astype(np.float32) / 255.0
        img = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(self.device)
        msk = torch.from_numpy((m > 0).astype(np.float32))[None, None].to(self.device)
        ph, pw = (-h) % 8, (-w) % 8                          # 3 stride-2 stages -> multiple of 8
        if ph or pw:
            img = F.pad(img, (0, pw, 0, ph), mode="reflect")
            msk = F.pad(msk, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            pred = self.net(torch.cat([img * (1 - msk), msk], dim=1))
            out = msk * pred + (1 - msk) * img
        out = out[0, :, :h, :w].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        return np.ascontiguousarray((out * 255.0 + 0.5).astype(np.uint8)[..., ::-1])

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        m = (mask > 0).astype(np.uint8)
        if self.grow > 0:
            k = 2 * self.grow + 1
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        from app.threading_utils import TORCH_INFER_LOCK

        def _locked_run(bgr, mm):          # từng ô một: giữ khoá ngắn để Manga-OCR / YOLO của luồng dịch xen vào được
            with TORCH_INFER_LOCK:
                return self._run(bgr, mm)
        return inpaint_tiled(img, m, _locked_run, self.win, self.margin, adaptive=self.adaptive)


def check_state(net, sd) -> str:
    """Return '' if the checkpoint matches the network, else a readable report."""
    mine = net.state_dict()
    miss = sorted(set(mine) - set(sd))
    extra = sorted(set(sd) - set(mine))
    bad = [f"{k}: checkpoint {tuple(sd[k].shape)} vs network {tuple(mine[k].shape)}"
           for k in sorted(set(mine) & set(sd)) if tuple(sd[k].shape) != tuple(mine[k].shape)]
    lines = []
    for title, items in (("missing in checkpoint", miss), ("unexpected in checkpoint", extra), ("shape mismatch", bad)):
        if items:
            lines.append(f"{title} ({len(items)}): " + "; ".join(items[:6]) + (" ..." if len(items) > 6 else ""))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="lama-manga inpainting (tiled, native resolution)")
    ap.add_argument("--model", required=True, help="lama-manga.safetensors")
    ap.add_argument("--image")
    ap.add_argument("--mask", help="non-zero = pixels to fill")
    ap.add_argument("--out", default="lama_out.png")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--margin", type=int, default=48)
    ap.add_argument("--grow", type=int, default=1, help="dilate the mask by N px (LaMa likes a small margin)")
    ap.add_argument("--no-adaptive", action="store_true", help="always use full --win windows")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch decides)")
    ap.add_argument("--check", action="store_true", help="only verify the checkpoint against the network")
    a = ap.parse_args()

    if a.check:
        from safetensors.torch import load_file
        net = build_network()
        sd = load_file(a.model)
        rep = check_state(net, sd)
        n = sum(1 for _ in sd)
        print(f"{n} tensors in checkpoint")
        print("OK: all keys and shapes match" if not rep else "MISMATCH:\n" + rep)
        sys.exit(0 if not rep else 1)
    if not (a.image and a.mask):
        sys.exit("--image and --mask are required (or use --check)")
    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        sys.exit("cannot read image or mask")
    if mask.shape != img.shape[:2]:
        sys.exit(f"mask {mask.shape} and image {img.shape[:2]} sizes differ")
    model = LamaManga(a.model, a.device, a.win, a.margin, a.grow, not a.no_adaptive, a.threads)
    cv2.imwrite(a.out, model(img, mask))
    print("saved", a.out)


if __name__ == "__main__":
    main()
