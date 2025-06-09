from __future__ import annotations
"""nas_ga.py – GA orchestrator for *any* Template subclass.

Relies solely on `template_base.Template` APIs (encode/decode/random_individual)
so concrete templates remain completely declarative.
"""
from typing import List, Type
import random

from template_base import Template, get_template  # type: ignore


def _lazy_deap():
    from deap import base, creator, tools  # noqa: F401 – runtime import
    return base, creator, tools


class NAS:
    """Evolutionary search wrapper (DEAP backend)."""

    def __init__(self, template: str | Type[Template], *, num_classes: int = 10):
        if isinstance(template, str):
            template = get_template(template)
        self.tpl_cls: Type[Template] = template
        self.tpl: Template = template(num_classes=num_classes)  # type: ignore[arg-type]
        self.best_model_ = None
        self.best_score_: float | None = None

    # ------------------------------------------------------------------
    def evolve(
        self,
        *,
        population: int = 32,
        generations: int = 10,
        train_fn,  # callable(model) -> fitness float
        cx_pb: float = 0.5,
        mut_pb: float = 0.2,
    ):
        base, creator, tools = _lazy_deap()
        if "FitnessMax" not in creator.__dict__:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", lambda: creator.Individual(self.tpl_cls.random_individual()))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self._cx_bundle)
        toolbox.register("mutate", self._mutate, indpb=mut_pb)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", lambda ind: (train_fn(self.tpl(self.tpl_cls.decode(ind))),))

        pop = toolbox.population(population)
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
        for gen in range(generations):
            offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
            # crossover -------------------------------------------------
            for i1, i2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_pb:
                    toolbox.mate(i1, i2)
                    del i1.fitness.values, i2.fitness.values
            # mutation --------------------------------------------------
            for ind in offspring:
                if random.random() < mut_pb:
                    toolbox.mutate(ind)
                    del ind.fitness.values
            # evaluation ----------------------------------------------
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)
            pop[:] = offspring
            best = max(pop, key=lambda ind: ind.fitness.values[0])
            if self.best_score_ is None or best.fitness.values[0] > self.best_score_:
                self.best_score_ = best.fitness.values[0]
                self.best_model_ = self.tpl(self.tpl_cls.decode(best))
            print(f"Gen {gen:02d} → best {self.best_score_:.4f}")
        return self

    # ------------------------------------------------------------------
    # bundle-aware operators -------------------------------------------
    # ------------------------------------------------------------------
    def _bundle_boundaries(self, ind: List[int]) -> List[int]:
        """Return indices where a bundle *starts* (including tail end)."""
        tpl = self.tpl_cls
        idx = len(tpl.GENE_SCHEMA)
        starts = [idx]
        while idx < len(ind):
            tag = ind[idx]
            if tag not in tpl.BUNDLE_TYPES:
                break
            idx += 1 + len(tpl.BUNDLE_TYPES[tag][1])
            starts.append(idx)
        return starts

    def _cx_bundle(self, ind1: List[int], ind2: List[int]):
        bounds1 = self._bundle_boundaries(ind1)
        bounds2 = self._bundle_boundaries(ind2)
        if len(bounds1) < 2 or len(bounds2) < 2:
            return ind1, ind2  # nothing to swap
        cut1 = random.choice(bounds1[:-1])
        cut2 = random.choice(bounds2[:-1])
        ind1[cut1:], ind2[cut2:] = ind2[cut2:], ind1[cut1:]
        return ind1, ind2

    def _mutate(self, ind: List[int], *, indpb: float):
        tpl = self.tpl_cls
        scalars_len = len(tpl.GENE_SCHEMA)
        # scalars -------------------------------------------------------
        for i, spec in enumerate(tpl.GENE_SCHEMA):
            if random.random() < indpb:
                ind[i] = random.randrange(len(spec.choices))
        # decode bundles for easier manipulation ------------------------
        bundles = tpl.decode(ind)["bundles"]
        # add bundle ----------------------------------------------------
        if random.random() < indpb and len(bundles) < tpl.MAX_BUNDLES:
            tag = random.choice(list(tpl.BUNDLE_TYPES))
            new_params = [random.randrange(len(s.choices)) for s in tpl.BUNDLE_TYPES[tag][1]]
            bundles.insert(random.randrange(len(bundles)+1), (tag, new_params))
        # drop bundle ---------------------------------------------------
        if len(bundles) > 1 and random.random() < indpb:
            del bundles[random.randrange(len(bundles))]
        # mutate bundles ------------------------------------------------
        for bi, (tag, params) in enumerate(bundles):
            # mutate tag (swap entire schema)
            if random.random() < indpb:
                tag = random.choice(list(tpl.BUNDLE_TYPES))
                params = [random.randrange(len(s.choices)) for s in tpl.BUNDLE_TYPES[tag][1]]
                bundles[bi] = (tag, params)
            else:
                schema = tpl.BUNDLE_TYPES[tag][1]
                for pi, spec in enumerate(schema):
                    if random.random() < indpb:
                        params[pi] = random.randrange(len(spec.choices))
        # re-encode -----------------------------------------------------
        new_tail: List[int] = []
        for tag, params in bundles:
            new_tail.append(tag); new_tail.extend(params)
        ind[:] = ind[:scalars_len] + new_tail
        return (ind,)