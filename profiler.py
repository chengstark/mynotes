# mem_profiler_qtiny.py  (Python ≥3.8)
"""
Flash, SRAM *and* operation-count estimator for proprietary-quantised
PyTorch models.  Requires **no edits to your model**; you only need
to supply a `bit_getter` that understands your quantisation metadata.

• Flash  :   param.numel() * param.quant   (or dtype size)
• SRAM   :   peak Σ(input + output) across layers, using bit_getter
• MACs   :   Conv2d & Linear multiply–adds   (others easy to add)
"""

from __future__ import annotations
import math, torch
from typing import Callable, List, Tuple, Union, Optional       # <-- add Optional

_Tensor   = torch.Tensor
BitGetter = Callable[[_Tensor, Optional[torch.nn.Module]], Optional[int]]

import torch
import torch.nn as nn


class ActQuantizerII(nn.Module):
    """
    Dummy activation-quantizer used only for profiling tests.
    * quant_scheme = "float"  →  acts stay fp32
    * quant_scheme = "LSQ-4", "LSQ-6", "LSQ-8", …  →  treated as N-bit
    Nothing is really quantised—the tensor is passed through unchanged—
    but num_bits and quant_scheme behave as in your real code.
    """

    def __init__(self, quant_scheme: str = "float"):
        super().__init__()

        # default: no quantisation
        self.quant_scheme = "float"
        self.num_bits = 32

        # override if user sent e.g. "LSQ-4"
        if quant_scheme != "float":
            scheme, bits = quant_scheme.split("-")
            self.quant_scheme = scheme          # e.g. "LSQ"
            self.num_bits = int(bits)           # e.g. 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # real implementation would quantise; we just return x
        return x

class ModelProfiler:
    # --------------------------- init & public --------------------------- #
    def __init__(
        self,
        model: torch.nn.Module,
        example_input: Union[_Tensor, Tuple, List],
        bit_getter: BitGetter,
    ):
        self.model         = model.eval()
        self.example_input = example_input
        self.bit_getter    = bit_getter

        self._peak_bytes   = 0
        self._total_macs   = 0
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def flash_bytes(self) -> int:
        """
        Total Flash in bytes.

        • If a tensor carries .quant, use it.
        • Else look at its owning module’s .weight_quantizer.
        • Else fall back to dtype size.
        """
        total_bits = 0
        module_lookup = dict(self.model.named_modules())

        for pname, param in self.model.named_parameters():
            # split "layer.weight" → "layer"
            owner_name = pname.rsplit(".", 1)[0]
            owner      = module_lookup[owner_name]

            if hasattr(param, "quant"):
                bits = int(param.quant)

            elif hasattr(owner, "weight_quantizer"):
                wq = owner.weight_quantizer
                if isinstance(wq, ActQuantizerII) and wq.quant_scheme != "float":
                    bits = int(wq.num_bits)
                else:
                    bits = param.element_size() * 8
            else:
                bits = param.element_size() * 8
            # print(pname, wq.quant_scheme, bits)
            total_bits += param.numel() * bits

        return total_bits // 8

    def sram_bytes(self) -> int:
        if self._peak_bytes == 0:
            self._run_and_profile()
        return self._peak_bytes

    def macs(self) -> int:
        if self._total_macs == 0:
            self._run_and_profile()
        return self._total_macs

    def summary(self, name: str | None = None) -> str:
        def human(n):
            for u in ("B", "KB", "MB", "GB", "TB"):
                if n < 1024:
                    return f"{n:.2f} {u}"
                n /= 1024
        name = name or self.model.__class__.__name__
        return (
            f"── Memory/Compute summary for **{name}** ──\n"
            f"Flash / ROM : {human(self.flash_bytes())}\n"
            f"SRAM peak   : {human(self.sram_bytes())}\n"
            f"MACs / op   : {self.macs()/1e6:,.2f} M"
        )

    # --------------------------- internals ------------------------------ #
    def _run_and_profile(self):
        # attach hook to EVERY module except pure quantisers
        for m in self.model.modules():
            if not isinstance(m, ActQuantizerII):
                self._hooks.append(m.register_forward_hook(self._hook))

        with torch.no_grad():
            _ = self.model(self.example_input)

        for h in self._hooks:
            h.remove()

    def _hook(self, module, inputs, output):
        # ---- SRAM running max ------------------------------------------
        bytes_layer = (
            sum(self._tensor_bytes(t, module) for t in inputs)
            + self._tensor_bytes(output, module)
        )
        self._peak_bytes = max(self._peak_bytes, bytes_layer)

        # ---- MAC counter ----------------------------------------------
        self._total_macs += self._macs_of(module, output)

    # ---- helpers -------------------------------------------------------
    def _tensor_bytes(self, t: _Tensor, module) -> int:
        if not torch.is_tensor(t):
            return 0
        bits = self.bit_getter(t, module)
        if bits is None:
            bits = t.element_size() * 8
        return math.ceil(t.numel() * bits / 8)

    @staticmethod
    def _macs_of(module: torch.nn.Module, out: _Tensor) -> int:
        """Count multiply–adds for Conv2d & Linear (add more as needed)."""
        if isinstance(module, torch.nn.Conv2d):
            # NCHW output
            n, c_out, h, w = out.shape
            k_h, k_w        = module.kernel_size
            c_in            = module.in_channels
            groups          = module.groups
            macs = n * c_out * h * w * k_h * k_w * (c_in // groups)
            return int(macs)
        if isinstance(module, torch.nn.Linear):
            n, out_feats = out.shape
            in_feats     = module.in_features
            return int(n * in_feats * out_feats)
        return 0



def default_bit_getter(tensor, module):
    """
    Decide the *activation* bit-width for `tensor`.

    Works with your QConv2d / QLinear that own:
       • module.act_quantizer  : ActQuantizerII(...)
       • module.weight_quantizer: ActQuantizerII(...)
    """
    # Case A – this module *is itself* a quantiser leaf
    if isinstance(module, ActQuantizerII):
        return int(module.num_bits)

    # Case B – parent layer has an activation quantiser
    aq = getattr(module, "act_quantizer", None)
    if isinstance(aq, ActQuantizerII) and aq.quant_scheme != "float":
        return int(aq.num_bits)

    # (Optional) If you tag tensors during forward you can keep:
    if hasattr(tensor, "quant"):
        return int(tensor.quant)

    return None 


if __name__ == "__main__":
    class ToyNetQ(nn.Module):
        def __init__(self):
            super().__init__()

            # ---- quantised conv layer ------------------------------------
            self.conv = nn.Conv2d(3, 8, 3, 1, 1)
            self.conv.weight_quantizer = ActQuantizerII("LSQ-8")   # weights 8-bit
            self.conv.act_quantizer    = ActQuantizerII("LSQ-4")   # acts   4-bit

            # ---- global-pool + quantised linear --------------------------
            self.gap   = nn.AdaptiveAvgPool2d(1)

            self.fc    = nn.Linear(8, 10)
            self.fc.weight_quantizer = ActQuantizerII("LSQ-9")
            self.fc.act_quantizer    = ActQuantizerII("float")     # leave output fp32

        def forward(self, x):
            x = self.conv(x)
            x = self.conv.act_quantizer(x)          # quantise conv output
            x = self.gap(x).flatten(1)
            x = self.fc(x)
            return x

    model   = ToyNetQ()
    dummy   = torch.zeros(1, 3, 32, 32)

    profiler = ModelProfiler(model, dummy, default_bit_getter)
    print(profiler.summary("ToyNet-stub"))
