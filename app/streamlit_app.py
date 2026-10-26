"""Presentation-friendly Streamlit interface for the simplified SpaceGen AI workflow."""
from __future__ import annotations

import sys
import io
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from spacegen.catalog import load_room_catalog
from spacegen.layout import layout_to_dict
from spacegen.paths import REPO_ROOT
from spacegen.quality import quality_score
from spacegen.rules import WALLS, load_rules
from spacegen.simple_pipeline import Request, generate_simple, load_checkpoint
from spacegen.rules import check_layout, reachability
from spacegen.viz import SURFACE, plot_layout


def facts_for(layout):
    reach = reachability(layout, catalog, rules)
    checked = check_layout(layout, catalog, rules, reach=reach)
    return {"quality": quality_score(layout, catalog, rules, reach=reach).total,
            "checks": {"H1 inside room": checked.in_room, "H2 no overlap": checked.no_overlap,
                       "H3 door clear": checked.door_clear, "H4 reachable": checked.reachable}}


def layout_png(layout, title):
    figure = plot_layout(layout, catalog, rules, title=title, width=3.4)
    output = io.BytesIO()
    figure.savefig(output, format="png", facecolor=SURFACE)
    plt.close(figure)
    return output.getvalue()

st.set_page_config(page_title="SpaceGen AI", layout="wide")


@st.cache_resource
def resources():
    catalog, rules = load_room_catalog("living_room"), load_rules()
    model, raster = load_checkpoint(REPO_ROOT / "checkpoints" / "evaluator" / "e9a")
    return catalog, rules, model, raster


catalog, rules, evaluator, raster = resources()
room = rules.rooms[catalog.room_type]

with st.sidebar:
    st.header("Room input")
    width = st.slider("Width (m)", 3.5, 7.0, 5.0, 0.1)
    depth = st.slider("Depth (m)", 3.0, 6.0, 4.0, 0.1)
    door = st.selectbox("Door wall", WALLS, index=WALLS.index("W"))
    offset = st.slider("Door position", 0.0, 1.0, 0.5, 0.05)
    st.header("Furniture")
    items = {"sofa": st.selectbox("Sofa", ["sofa_3seater", "sofa_2seater"]),
             "tv_unit": st.selectbox("TV unit", ["tv_unit_large", "tv_unit_standard"])}
    if st.checkbox("Add coffee table", value=True):
        items["coffee_table"] = "coffee_table_standard"
    if st.checkbox("Add armchair"):
        items["armchair"] = "armchair_standard"
    budget = st.number_input("Budget (INR, 0 = no limit)", 0, 200000, 0, 1000)
    candidates = st.slider("Candidate attempts", 8, 64, 24, 8)
    seed = int(st.number_input("Seed", 0, 9999, 0))

request = Request(width, depth, door, offset, items, float(budget) if budget else None)
st.title("SpaceGen AI: Simple CNN Layout Evaluator")
st.caption("Rule-based candidate generator → CNN raster evaluator → independent hard-constraint checker → diverse top 3")
st.info("The CNN predicts validity and layout quality, but H1–H4 rules make the final validity decision.")

if st.button("Generate layouts", type="primary"):
    with st.spinner("Generating candidates, evaluating raster images, and checking constraints..."):
        st.session_state.result = generate_simple(request, catalog, rules, evaluator, raster, seed, candidates)

result = st.session_state.get("result")
if result is None:
    st.write("Choose the room and furniture, then generate layouts.")
elif result.condition is None:
    st.error(result.message)
else:
    for warning in result.warnings:
        st.warning(warning)
    st.caption(f"{result.valid} rule-valid layouts from {result.candidates} attempts. Timings: {result.seconds}")
    if result.message:
        st.warning(result.message)
    for rank, (column, candidate) in enumerate(zip(st.columns(max(1, len(result.top))), result.top), 1):
        with column:
            facts = facts_for(candidate.layout)
            st.image(layout_png(candidate.layout, f"Layout {rank}"), width="stretch")
            st.metric("CNN validity probability", f"{candidate.evaluator_valid:.2f}")
            st.metric("CNN quality prediction", f"{candidate.evaluator_score:.2f}")
            st.metric("Rule quality", f"{facts['quality']:.2f}")
            st.dataframe(pd.DataFrame({"Hard constraint": list(facts["checks"]), "Passed": list(facts["checks"].values())}),
                         hide_index=True, width="stretch")
            st.download_button("Download layout JSON", json.dumps(layout_to_dict(candidate.layout, catalog, rules.door.width, facts), indent=2),
                               file_name=f"spacegen_layout_{rank}.json")
