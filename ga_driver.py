from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Sequence
import random, copy, enum


# ——— GA operations you support ——————————————————————————
class Op(str, enum.Enum):
    MUTATE = "mutate"
    MATE   = "mate"
    CLONE  = "clone"


# ——— Your domain‐specific individual ————————————————————
@dataclass
class ModelResult:
    id:        int
    code_hash: str
    code:      str
    perf:      Dict[str, float]          # metric → value
    summary:   str
    is_probe:  bool = False              # (if you need it)

    def clone(self, new_id: int) -> "ModelResult":
        """Deep-copy with fresh ID, fitness cleared."""
        return ModelResult(
            id=new_id,
            code_hash=self.code_hash,
            code=self.code,
            perf={},                  # child not evaluated yet
            summary=self.summary,
            is_probe=self.is_probe,
        )

class GADriver:
    """
    Decides *which* individuals to mutate, mate, or clone.
    It never touches `code`, compiles anything, or evaluates performance.
    """

    def __init__(
        self,
        pop_size: int,
        selector_rng: random.Random | None = None,
        tournament_k: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate:  float = 0.15,
        elitism: int = 1,
    ):
        self.rng = selector_rng or random.Random()
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.mutation_rate  = mutation_rate
        self.elitism = elitism

        self._next_id = 0
        self.population: List[ModelResult] = []

        # replacement policy: generational by default
        self._next_generation: List[ModelResult] = []

    # ────────────────────────── external API ──────────────────────────
    def add_seed(self, model: ModelResult) -> None:
        """Call once at the beginning for each handcrafted seed."""
        self.population.append(model)

    def plan_evolution(self) -> List[Tuple[Op, Tuple[ModelResult, ...]]]:
        """
        Return a *worklist* that the caller will execute.
        Each tuple = (operation, parent(s))
        """
        if len(self.population) == 0:
            raise RuntimeError("Population is empty")

        worklist: List[Tuple[Op, Tuple[ModelResult, ...]]] = []

        # 1. keep elites → they go straight into next gen
        elites = self._elite_slice()
        self._next_generation = [e.clone(self._new_id()) for e in elites]

        # 2. fill plan until we have enough children
        while len(self._next_generation) + len(worklist) < len(self.population):
            if self.rng.random() < self.crossover_rate:
                # MATE
                p1, p2 = self._tournament(), self._tournament()
                worklist.append((Op.MATE, (p1, p2)))
            elif self.rng.random() < self.mutation_rate:
                # MUTATE
                parent = self._tournament()
                worklist.append((Op.MUTATE, (parent,)))
            else:
                # CLONE
                parent = self._tournament()
                worklist.append((Op.CLONE, (parent,)))

        return worklist

    def commit_offspring(self, children: Sequence[ModelResult]) -> None:
        """
        Caller hands back the newborns (already evaluated or not).
        Replaces current population with next_generation + children.
        """
        needed = len(self.population) - len(self._next_generation)
        if needed != len(children):
            raise ValueError(f"Expected {needed} children, got {len(children)}")

        self.population = self._next_generation + list(children)
        self._next_generation = []       # clear for next round

    def best(self, metric: str) -> ModelResult:
        return max(self.population, key=lambda m: m.perf.get(metric, float("-inf")))

    # ────────────────────────── internal helpers ──────────────────────────
    def _elite_slice(self) -> List[ModelResult]:
        return sorted(self.population,
                      key=lambda m: m.perf.get("fitness", float("-inf")),
                      reverse=True)[: self.elitism]

    def _tournament(self) -> ModelResult:
        k = min(self.tournament_k, len(self.population))
        subset = self.rng.sample(self.population, k)
        return max(subset, key=lambda m: m.perf.get("fitness", float("-inf")))

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id
