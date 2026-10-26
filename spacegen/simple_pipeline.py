"""Small demo pipeline used by the presentation version of SpaceGen AI.

It deliberately has one generator and one learned model:

request -> rule-based candidate sampler -> CNN evaluator -> hard-rule checker -> top 3

The CNN is an evaluator, never the authority on safety.  The existing CVAE research pipeline
is retained in :mod:`spacegen.pipeline`, but is not needed to run this demonstrable path.
"""
from __future__ import annotations

import time
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from spacegen.catalog import RoomCatalog
from spacegen.generator import Condition, generate_layout, load_generator_config
from spacegen.metrics import layout_distance
from spacegen.models.evaluator import Evaluator, evaluator_config_from_record
from spacegen.quality import quality_score
from spacegen.raster import RasterConfig, rasterize_layouts
from spacegen.rules import Rules, check_layout, door_geometry, footprint_ratio, reachability


@dataclass(frozen=True)
class Request:
    """The small demo's room, door, furniture, and optional budget input."""
    width: float
    depth: float
    door_wall: str
    door_offset: float
    items: dict[str, str | None]
    budget: float | None = None
    pins: dict = field(default_factory=dict)


@dataclass
class Candidate:
    layout: object
    quality: float
    evaluator_score: float | None
    evaluator_valid: float | None


def check_request(request: Request, catalog: RoomCatalog, rules: Rules) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    names = {slot.name for slot in catalog.slots}
    errors += [f"unknown item {name!r}" for name in request.items if name not in names]
    errors += [f"the {slot.name.replace('_', ' ')} is required" for slot in catalog.slots
               if slot.mandatory and slot.name not in request.items]
    for name, variant in request.items.items():
        if name in names and variant is not None and variant not in {v.id for v in catalog.slot(name).variants}:
            errors.append(f"unknown variant {variant!r} for the {name.replace('_', ' ')}")
    if min(request.width, request.depth) <= 0:
        errors.append("the room needs a positive width and depth")
    else:
        try:
            door_geometry(request.width, request.depth, request.door_wall, request.door_offset, rules.door)
        except ValueError as error:
            errors.append(str(error))
    return errors, warnings


def choose_variants(request: Request, catalog: RoomCatalog, rules: Rules) -> tuple[Condition | None, str | None]:
    options = [[catalog.slot(name).variant(v)] if v else list(catalog.slot(name).variants)
               for name, v in request.items.items()]
    names, combos = list(request.items), list(itertools.product(*options))
    price = [sum(v.price for v in combo) for combo in combos]
    area = [sum(v.area for v in combo) for combo in combos]
    ratio = [footprint_ratio(value, request.width, request.depth, rules.door) for value in area]
    fits = [i for i in range(len(combos)) if ratio[i] <= rules.f_max and
            (request.budget is None or price[i] <= request.budget)]
    if fits:
        best = max(fits, key=lambda i: (area[i], -price[i]))
        return Condition(request.width, request.depth, request.door_wall, request.door_offset,
                         {name: v.id for name, v in zip(names, combos[best])}), None
    if request.budget is not None and min(price) > request.budget:
        return None, f"the cheapest choice costs {min(price):,} INR, over the budget"
    return None, "the furniture covers too much of the free floor; remove an optional item or choose a larger room"


def select_diverse(layouts: list, k: int = 3, threshold: float = 0.3) -> list[int]:
    picks = []
    for i, layout in enumerate(layouts):
        if all(layout_distance(layout, layouts[j]) >= threshold for j in picks):
            picks.append(i)
            if len(picks) == k:
                break
    return picks


@dataclass
class SimpleResult:
    request: Request
    condition: Condition | None
    top: list[Candidate] = field(default_factory=list)
    message: str | None = None
    warnings: list[str] = field(default_factory=list)
    candidates: int = 0
    valid: int = 0
    seconds: dict[str, float] = field(default_factory=dict)


def generate_candidates(request: Request, catalog: RoomCatalog, rules: Rules, rng: np.random.Generator,
                        candidates: int = 32) -> tuple[Condition | None, list, str | None, list[str]]:
    """Generate candidate layouts with the existing understandable rule-based sampler."""
    errors, warnings = check_request(request, catalog, rules)
    if errors:
        return None, [], "; ".join(errors), warnings
    if request.pins:
        return None, [], "pinning is deferred in the simplified demo; remove the pin and try again", warnings
    condition, message = choose_variants(request, catalog, rules)
    if condition is None:
        return None, [], message, warnings
    config = load_generator_config()
    layouts = []
    for _ in range(candidates):
        generated = generate_layout(condition, rng, catalog, rules, config)
        if generated.layout is not None:
            layouts.append(generated.layout)
    return condition, layouts, None, warnings


def generate_simple(request: Request, catalog: RoomCatalog, rules: Rules, evaluator: Evaluator,
                    raster: RasterConfig, seed: int = 0, candidates: int = 32,
                    device: str | torch.device = "cpu") -> SimpleResult:
    """Generate, score, independently validate, and diversify up to three layouts."""
    started = time.perf_counter()
    condition, layouts, message, warnings = generate_candidates(
        request, catalog, rules, np.random.default_rng(seed), candidates)
    timings = {"generate": time.perf_counter() - started}
    if condition is None:
        return SimpleResult(request, None, message=message, warnings=warnings, seconds=timings)

    # CNN sees every sampled candidate as the same 4-channel top-down raster used in training.
    started = time.perf_counter()
    with torch.no_grad():
        logits, predicted_quality = evaluator.eval()(rasterize_layouts(layouts, catalog, rules, raster, device)) if layouts else (
            torch.empty(0), torch.empty(0))
    probabilities = torch.sigmoid(logits).cpu().tolist()
    predicted_quality = predicted_quality.cpu().tolist()
    timings["cnn"] = time.perf_counter() - started

    # H1 boundaries, H2 overlap, H3 door clearance and H4 reachability are final authority.
    started = time.perf_counter()
    valid: list[Candidate] = []
    for layout, probability, cnn_quality in zip(layouts, probabilities, predicted_quality):
        reach = reachability(layout, catalog, rules)
        if check_layout(layout, catalog, rules, reach=reach).valid:
            rule_quality = quality_score(layout, catalog, rules, reach=reach).total
            valid.append(Candidate(layout, float(rule_quality), float(cnn_quality), float(probability)))
    timings["rules"] = time.perf_counter() - started

    ordered = sorted(valid, key=lambda item: item.evaluator_score if item.evaluator_score is not None else item.quality,
                     reverse=True)
    picks = select_diverse([item.layout for item in ordered])
    top = [ordered[i] for i in picks]
    note = None if len(top) == 3 else f"Only {len(top)} sufficiently different valid layouts were found from {candidates} attempts."
    return SimpleResult(request, condition, top, note, warnings, candidates, len(valid),
                        {key: round(value, 3) for key, value in timings.items()})


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[Evaluator, RasterConfig]:
    """Load the compatible saved CNN checkpoint and its raster convention."""
    record = json.loads((path / "run.json").read_text(encoding="utf-8"))
    config = record["evaluator"]
    model = Evaluator(evaluator_config_from_record(config), pixels=record["raster"]["pixels"])
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval(), RasterConfig(**record["raster"])
