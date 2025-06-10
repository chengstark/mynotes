"""
ga_driver.py  —  Minimal Pareto-aware GA planner (NSGA-II-lite)

Author : ChatGPT example, 2025-06-10
License: MIT
"""

from __future__ import annotations
import random, copy, enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Sequence, Iterable


# ────────────────────────────────────────────────────────────────
# 1.  Basic data containers
# ────────────────────────────────────────────────────────────────

class Op(str, enum.Enum):
    """Evolutionary operation requested by the planner."""
    MUTATE = "mutate"
    MATE   = "mate"
    CLONE  = "clone"


@dataclass
class ModelResult:
    """
    One individual in the population.

    perf : dict with *all* objective metrics (keys = str, values = float)
           e.g. {"accuracy":0.91, "latency_ms":42.7, "size_mb":3.4}
    """
    id:        int
    code_hash: str
    code:      str
    perf:      Dict[str, float]
    summary:   str
    is_probe:  bool = False

    # ---- Pareto metadata (filled every generation) ----
    rank:      int   = 0
    crowding:  float = 0.0

    # utility ----------------------------------------------------
    def clone(self, new_id: int) -> "ModelResult":
        """Deep-copy with fresh ID and EMPTY perf (child not yet evaluated)."""
        return ModelResult(
            id=new_id,
            code_hash=self.code_hash,
            code=self.code,
            perf={},              # child must be (re)evaluated
            summary=self.summary,
            is_probe=self.is_probe,
        )


# ────────────────────────────────────────────────────────────────
# 2.  GA Driver (planner only)
# ────────────────────────────────────────────────────────────────

class GADriver:
    """
    NSGA-II-style planner: decides which parents take which operation,
    but *never* applies the operations or evaluations itself.
    """

    def __init__(
        self,
        population_size:     int,
        objectives:          List[str],          # e.g. ["accuracy", "-latency_ms"]
        crossover_rate:      float = 0.8,
        mutation_rate:       float = 0.15,
        tournament_k:        int   = 3,
        elitism:             int   = 1,
        rng: random.Random | None = None,
    ) -> None:
        if not objectives:
            raise ValueError("At least one objective must be supplied")

        self.population_size   = population_size
        self.objectives        = objectives      # '+' maximise, '-' minimise
        self.crossover_rate    = float(crossover_rate)
        self.mutation_rate     = float(mutation_rate)
        self.tournament_k      = int(tournament_k)
        self.elitism           = int(elitism)
        self.rng               = rng or random.Random()

        self.population: List[ModelResult] = []
        self._next_generation: List[ModelResult] = []    # elites buffered here
        self._next_id: int = 0

    # ─────────────────────────── public API ───────────────────────────

    # seeding .............................
    def add_seed(self, ind: ModelResult) -> None:
        """Insert an already-evaluated individual (fills .perf) before GA starts."""
        if len(self.population) >= self.population_size:
            raise ValueError("Population size already met")
        self.population.append(ind)

    # planning ............................
    def plan_evolution(self) -> List[Tuple[Op, Tuple[ModelResult, ...]]]:
        """
        Return a work-list of (Op, parents) tuples.
        Caller must honour the order: *one offspring per tuple*.

        Steps:
          1) Pareto-rank & crowding
          2) Copy elites to _next_generation
          3) Build work-list until the next generation will be full
        """
        if len(self.population) != self.population_size:
            raise RuntimeError(
                f"Population must contain exactly {self.population_size} "
                f"individuals (currently {len(self.population)})"
            )
        self._update_pareto()

        # ---- keep elites --------------------------------------------------
        elites = self._elite_slice()
        self._next_generation = [e.clone(self._new_id()) for e in elites]

        # ---- plan rest of children ----------------------------------------
        worklist: List[Tuple[Op, Tuple[ModelResult, ...]]] = []

        while len(self._next_generation) + len(worklist) < self.population_size:
            r = self.rng.random()
            if r < self.crossover_rate:           # Mate
                p1, p2 = self._tournament(), self._tournament()
                worklist.append((Op.MATE, (p1, p2)))
            elif r < self.crossover_rate + self.mutation_rate:  # Mutate
                parent = self._tournament()
                worklist.append((Op.MUTATE, (parent,)))
            else:                                  # Clone
                parent = self._tournament()
                worklist.append((Op.CLONE, (parent,)))

        return worklist

    # committing ..........................
    def commit_offspring(self, children: Sequence[ModelResult]) -> None:
        """
        Replace current population with elites + children.
        All children must have *perf* filled (but rank/crowding will be recomputed).
        """
        expected = self.population_size - len(self._next_generation)
        if len(children) != expected:
            raise ValueError(f"Expected {expected} children, got {len(children)}")

        self.population = self._next_generation + list(children)
        self._next_generation = []      # reset buffer

    # helpers .............................
    def best(self, metric: str) -> ModelResult:
        """Return individual with max *raw* metric value (direction ignored)."""
        return max(self.population, key=lambda m: m.perf.get(metric, float("-inf")))

    # ─────────────────────────── internal helpers ───────────────────────────

    # IDs .................................
    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # Pareto ranking ......................
    def _update_pareto(self) -> None:
        """
        Fast non-dominated sort + crowding distance for every individual.
        Results stored in m.rank (int) and m.crowding (float).
        """
        objectives = self.objectives

        # ---------- fast non-dominated sort ----------
        fronts: List[List[ModelResult]] = []
        S: Dict[ModelResult, List[ModelResult]] = {}
        n_dom: Dict[ModelResult, int] = {}

        fronts.append([])   # first front

        for p in self.population:
            S[p], n_dom[p] = [], 0
            for q in self.population:
                if p is q:
                    continue
                if self._dominates(p, q, objectives):
                    S[p].append(q)
                elif self._dominates(q, p, objectives):
                    n_dom[p] += 1
            if n_dom[p] == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        q.rank = i + 1
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            i += 1

        # ---------- crowding distance ----------
        for front in fronts:
            self._crowding_distance(front, objectives)

    def _metric_value(self, ind: ModelResult, obj: str) -> float:
        """
        Return metric transformed so that *larger* is always *better*.
        If obj begins with '-', we minimise -> invert sign.
        """
        if obj.startswith("-"):
            return -ind.perf[obj[1:]]
        return ind.perf[obj]

    def _dominates(self, p: ModelResult, q: ModelResult, objectives: List[str]) -> bool:
        better, worse = False, False
        for obj in objectives:
            p_val = self._metric_value(p, obj)
            q_val = self._metric_value(q, obj)
            if p_val > q_val:
                better = True
            elif p_val < q_val:
                worse = True
        return better and not worse

    def _crowding_distance(self, front: List[ModelResult], objectives: List[str]) -> None:
        if not front:
            return
        for ind in front:
            ind.crowding = 0.0

        for obj in objectives:
            front.sort(key=lambda m: self._metric_value(m, obj))
            front[0].crowding = front[-1].crowding = float("inf")

            obj_values = [self._metric_value(m, obj) for m in front]
            min_v, max_v = obj_values[0], obj_values[-1]
            if max_v == min_v:                           # all same ⇒ skip
                continue
            denom = max_v - min_v

            for i in range(1, len(front) - 1):
                next_v = obj_values[i + 1]
                prev_v = obj_values[i - 1]
                front[i].crowding += (next_v - prev_v) / denom

    # selection helpers ...................
    def _elite_slice(self) -> List[ModelResult]:
        """Return the top-N by (rank, -crowding)."""
        return sorted(
            self.population,
            key=lambda m: (m.rank, -m.crowding)
        )[: self.elitism]

    def _tournament(self) -> ModelResult:
        """Pareto tournament (size k). Lower rank better; on tie, higher crowding."""
        subset = self.rng.sample(self.population, self.tournament_k)
        return min(subset, key=lambda m: (m.rank, -m.crowding))


# ────────────────────────────────────────────────────────────────
# 3.  Quick demo (run `python ga_driver.py` to see it work)
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Dummy mutate / mate / evaluate for illustration ----------------------
    def random_code() -> str:
        return "".join(random.choice("ACGT") for _ in range(8))

    def eval_perf(code: str) -> Dict[str, float]:
        """Toy: accuracy = fraction of 'A'; latency inversely related."""
        acc = code.count("A") / len(code)
        latency = 100 - acc * 80 + random.gauss(0, 2)   # ms
        return {"accuracy": acc, "latency_ms": latency}

    def mutate(parent: ModelResult, new_id: int) -> ModelResult:
        child = parent.clone(new_id)
        pos = random.randrange(len(child.code))
        child.code = child.code[:pos] + random.choice("ACGT") + child.code[pos + 1 :]
        child.perf = eval_perf(child.code)
        return child

    def mate(p1: ModelResult, p2: ModelResult, new_id: int) -> ModelResult:
        cut = random.randrange(1, len(p1.code) - 1)
        child_code = p1.code[:cut] + p2.code[cut:]
        child = ModelResult(
            id=new_id,
            code_hash="",
            code=child_code,
            perf=eval_perf(child_code),
            summary="child"
        )
        return child

    # ---------------------------------------------------------------------
    POP = 30
    driver = GADriver(
        population_size=POP,
        objectives=["accuracy", "-latency_ms"],   # maximise accuracy, minimise latency
        crossover_rate=0.7,
        mutation_rate=0.25,
        elitism=2,
        rng=random.Random(42),
    )

    # seed initial population
    for _ in range(POP):
        code = random_code()
        perf = eval_perf(code)
        driver.add_seed(
            ModelResult(
                id=driver._new_id(),
                code_hash="",
                code=code,
                perf=perf,
                summary="seed",
            )
        )

    # main loop
    for gen in range(30):
        plan = driver.plan_evolution()
        children: List[ModelResult] = []

        for op, parents in plan:
            if op is Op.MUTATE:
                p = parents[0]
                children.append(mutate(p, driver._new_id()))
            elif op is Op.MATE:
                p1, p2 = parents
                children.append(mate(p1, p2, driver._new_id()))
            elif op is Op.CLONE:
                p = parents[0]
                child = p.clone(driver._new_id())
                child.perf = eval_perf(child.code)
                children.append(child)
            else:
                raise ValueError(op)

        driver.commit_offspring(children)

        best_acc = driver.best("accuracy")
        print(
            f"Gen {gen:02d} "
            f"best acc={best_acc.perf['accuracy']:.3f} "
            f"lat={best_acc.perf['latency_ms']:.1f}ms "
            f"rank={best_acc.rank}"
        )
