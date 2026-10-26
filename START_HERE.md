# SpaceGen AI - Simplified Project

This folder is the standalone presentation version. It contains the rule-based candidate generator,
the trained CNN evaluator checkpoint, rasterizer, independent hard-constraint checker, and Streamlit UI.

Run:

```powershell
python run.py app
```

The app's actual flow is:

```text
Room input -> rule-based candidate sampler -> 4-channel raster image -> CNN
-> H1/H2/H3/H4 rule checker -> diverse top 3 layouts
```

The CNN is not trusted as the final safety decision. The checker rejects layouts outside the room,
with overlap, blocking the door, or without reachability.

The omitted research modules are CVAE generation, latent optimization, comparison experiments,
and other dashboards. The original full project remains one directory above this folder.
