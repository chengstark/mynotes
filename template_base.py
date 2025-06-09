from __future__ import annotations
"""template_base.py – minimal primitives for defining searchable model families

* **GeneSpec**      – names an adjustable hyper-parameter and its discrete choices
* **Template**      – base class implementing tagged-bundle genome helpers
* **register_template / get_template** – lightweight registry so NAS engine can
  instantiate templates by name.

No DEAP, no torch – totally backend-agnostic.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple, Type
import random

# ---------------------------------------------------------------------------
# Registry ------------------------------------------------------------------
# ---------------------------------------------------------------------------
_TEMPLATE_REGISTRY: Dict[str, "Template"] = {}

def register_template(cls: Type["Template"]):
    _TEMPLATE_REGISTRY[cls.__name__] = cls
    return cls

def get_template(name: str) -> Type["Template"]:
    try:
        return _TEMPLATE_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown template '{name}'. Available: {list(_TEMPLATE_REGISTRY)}") from e

# ---------------------------------------------------------------------------
# Core dataclass -------------------------------------------------------------
# ---------------------------------------------------------------------------
class GeneSpec(NamedTuple):
    name: str
    choices: Sequence[Any]  # genome stores index into this list

# ---------------------------------------------------------------------------
# Tagged-bundle Template base -------------------------------------------------
# ---------------------------------------------------------------------------
class Template(ABC):
    """Base class for heterogeneous, variable-length architectures.

    Sub-classes **must** declare:
    * ``GENE_SCHEMA``  – tuple[GeneSpec] for scalar hyper-params
    * ``BUNDLE_TYPES`` – dict[tag → (human_name, tuple[GeneSpec, ...])]
    * ``MAX_BUNDLES``  – max length of the bundle list searched by NAS
    """

    GENE_SCHEMA: Tuple[GeneSpec, ...] = ()
    BUNDLE_TYPES: Dict[int, Tuple[str, Tuple[GeneSpec, ...]]] = {}
    MAX_BUNDLES: int = 0

    # ---- public ------------------------------------------------------
    @abstractmethod
    def build(self, genotype: Dict[str, Any]): ...
    def __call__(self, genotype: Dict[str, Any]):
        return self.build(genotype)

    # ---- genome helpers ---------------------------------------------
    @classmethod
    def encode(cls, g: Dict[str, Any]) -> List[int]:
        flat = [g[s.name] for s in cls.GENE_SCHEMA]
        for tag, params in g["bundles"]:
            flat.append(tag); flat.extend(params)
        return flat

    @classmethod
    def decode(cls, flat: Sequence[int]) -> Dict[str, Any]:
        g: Dict[str, Any] = {}
        idx = 0
        # scalars
        for spec in cls.GENE_SCHEMA:
            g[spec.name] = flat[idx] if idx < len(flat) else 0; idx += 1
        # bundles
        bundles: List[Tuple[int, List[int]]] = []
        while idx < len(flat) and len(bundles) < cls.MAX_BUNDLES:
            tag = flat[idx]; idx += 1
            if tag not in cls.BUNDLE_TYPES: break
            schema = cls.BUNDLE_TYPES[tag][1]
            n = len(schema)
            if idx + n > len(flat): break
            bundles.append((tag, list(flat[idx: idx + n])))
            idx += n
        if not bundles:
            tag0 = next(iter(cls.BUNDLE_TYPES))
            bundles = [(tag0, [0]*len(cls.BUNDLE_TYPES[tag0][1]))]
        g["bundles"] = bundles
        return g

    @classmethod
    def random_individual(cls) -> List[int]:
        flat = [random.randrange(len(spec.choices)) for spec in cls.GENE_SCHEMA]
        n = random.randint(1, cls.MAX_BUNDLES)
        for _ in range(n):
            tag = random.choice(list(cls.BUNDLE_TYPES))
            flat.append(tag)
            for spec in cls.BUNDLE_TYPES[tag][1]:
                flat.append(random.randrange(len(spec.choices)))
        return flat
