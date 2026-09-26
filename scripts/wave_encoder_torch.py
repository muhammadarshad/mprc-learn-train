"""QCMWaveEncoder — Pure-PyTorch port of wave_encode_numpy.

Input:  (B, 3, 113, 128) float32 in [0, 255]  ← aspect-fit RGB
Output: (B, 16, 113, 128) float32 in [0, 255]  ← 16-channel |ψ⟩ wave manifold

Implements exact Z₂₅₆ ring arithmetic to match the NumPy/Rust reference:
  Ch 0-2:  R, G, B              (pass-through)
  Ch 3:    Luma   = (77R + 150G + 29B) >> 8
  Ch 4:    Gray   = (R + G + B) // 3
  Ch 5:    Chroma = max(R,G,B) − min(R,G,B)
  Ch 6:    Phase  = ((R%7 + G%7 + B%7) × 14) & 0xFF
  Ch 7:    Winding = ((R+G+B) >> 8) × 85
  Ch 8:    d1     = (R − G) mod 256   (ADI wrap)
  Ch 9:    d2     = (G − B) mod 256   (ADI wrap)
  Ch 10-15: LoG × 6 scales {0,1,4,16,64,128} = _plaquette(_evolve(gray, t))

Physics constants FH=113, FW=128 are FIXED (derived from N=3 master quantum number).
This module has NO learnable parameters; all buffers are precomputed constants.

Parity guarantee:
  max(abs(QCMWaveEncoder()(x_float) - wave_encode_numpy(pil_img))) == 0
  for all pixel-integer inputs (integer arithmetic uses .floor() to replicate >>).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Physics constants — FIXED, do NOT change (CLAUDE.md derivation chain)
FH, FW = 113, 128             # manifold frame: height × width
SCALES = (0, 1, 4, 16, 64, 128)   # heat-evolution steps per LoG channel


class QCMWaveEncoder(nn.Module):
    """
    Pure-PyTorch wave encoder. No learnable parameters.

    Registered buffers:
        border_mask (1, 1, FH, FW) : 1 = interior pixel, 0 = border (vacuum margin stays 0)
        evolve_k    (1, 1, 3, 3)   : heat-diffusion stencil = [0,1,0;1,4,1;0,1,0] / 8
    """

    def __init__(self):
        super().__init__()

        # Border mask: interior pixels = 1, border = 0 (vacuum margin stays 0)
        mask = torch.zeros(1, 1, FH, FW)
        mask[:, :, 1:-1, 1:-1] = 1.0
        self.register_buffer("border_mask", mask)

        # Heat-diffusion kernel: (N + S + W + E + 4×center) / 8
        k = torch.tensor(
            [[[[0., 1., 0.],
               [1., 4., 1.],
               [0., 1., 0.]]]], dtype=torch.float32
        ) / 8.0
        self.register_buffer("evolve_k", k)

    # ------------------------------------------------------------------ #
    #  Internal helpers — integer-exact floor arithmetic                   #
    # ------------------------------------------------------------------ #

    def _evolve(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Heat-diffuse x for `steps` iterations.
        x: (B, 1, FH, FW) float32 with integer values in [0, 255].
        Border pixels are preserved unchanged (mirroring NumPy where only
        [1:-1, 1:-1] interior is updated). Note: row-0 can have real image
        data for tall portrait images (ay=0 in _frame_rgb), so borders must
        NOT be zeroed — they must be kept from the input x.
        Exact-integer parity with NumPy uint16 >> 3 via .floor().
        """
        for _ in range(steps):
            conv = F.conv2d(x, self.evolve_k, padding=1)           # (B, 1, FH, FW)
            # border_mask=1 at interior, 0 at border → keep original border pixels
            x = x * (1.0 - self.border_mask) + conv.floor() * self.border_mask
        return x

    def _plaquette(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wilson plaquette: (right + up − left − down + 128) mod 256.
        Border pixels → 128 (matches NumPy: np.full(..., 128) init).
        x: (B, 1, FH, FW) float32, integer values in [0, 255].
        """
        # Shift neighbours with zero-pad (border = 0 = vacuum)
        right = F.pad(x[:, :, :,  1:], [0, 1, 0, 0])    # x[i, j+1]
        left  = F.pad(x[:, :, :, :-1], [1, 0, 0, 0])    # x[i, j-1]
        up    = F.pad(x[:, :, :-1, :], [0, 0, 1, 0])    # x[i-1, j]  (pad top)
        down  = F.pad(x[:, :,  1:, :], [0, 0, 0, 1])    # x[i+1, j]  (pad bot)

        interior = right - left + up - down + 128.0
        # Border pixels → 128; interior → expr mod 256
        # Use x - 256*floor(x/256) instead of torch.remainder (not in CoreML MIL)
        out = 128.0 * (1.0 - self.border_mask) + interior * self.border_mask
        return out - 256.0 * (out / 256.0).floor()

    # ------------------------------------------------------------------ #
    #  Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, FH, FW) float32 in [0, 255]  — R, G, B channels
        returns: (B, 16, FH, FW) float32 in [0, 255]
        """
        r = x[:, 0:1]    # (B, 1, FH, FW)
        g = x[:, 1:2]
        b = x[:, 2:3]

        # Ch 3: Luma = (77R + 150G + 29B) >> 8  (BT.601, integer floor)
        luma = (77.0 * r + 150.0 * g + 29.0 * b).div(256.0).floor()

        # Ch 4: Gray = (R + G + B) // 3
        gray = (r + g + b).div(3.0).floor()

        # Ch 5: Chroma = max(R,G,B) − min(R,G,B)
        chroma = torch.max(torch.max(r, g), b) - torch.min(torch.min(r, g), b)

        # Ch 6: Phase = (R%7 + G%7 + B%7) × 14  [max = 18×14 = 252 < 256, no mod needed]
        # Use x - 7*floor(x/7) instead of fmod (not in CoreML MIL)
        def _mod7(v: torch.Tensor) -> torch.Tensor:
            return v - 7.0 * (v / 7.0).floor()
        phase = (_mod7(r) + _mod7(g) + _mod7(b)) * 14.0

        # Ch 7: Winding = ((R+G+B) >> 8) × 85  [result ∈ {0, 85, 170}]
        winding = (r + g + b).div(256.0).floor() * 85.0

        # Ch 8: d1 = (R − G) mod 256  (wrapping subtraction)
        # Use x - 256*floor(x/256) for negative-safe mod (not torch.remainder — not in CoreML MIL)
        def _mod256(v: torch.Tensor) -> torch.Tensor:
            return v - 256.0 * (v / 256.0).floor()
        d1 = _mod256(r - g)

        # Ch 9: d2 = (G − B) mod 256  (wrapping subtraction)
        d2 = _mod256(g - b)

        # Ch 10-15: LoG × 6 = _plaquette(_evolve(gray, t)) for t ∈ SCALES
        log_chs = [self._plaquette(self._evolve(gray, t)) for t in SCALES]

        return torch.cat([x, luma, gray, chroma, phase, winding, d1, d2] + log_chs, dim=1)
