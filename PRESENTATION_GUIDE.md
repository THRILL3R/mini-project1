# SpaceGen AI: presentation guide

## What runs in the app

1. The user enters room size, door wall/location, furniture, and optional budget.
2. `spacegen.simple_pipeline.generate_candidates` reuses the existing rule-based furniture generator to make a small set of candidate layouts.
3. `spacegen.raster.rasterize_layouts` converts each layout to a 4-channel, 128 x 128 top-down image: room, furniture footprints, furniture fronts, and door-clearance zone.
4. The saved CNN evaluates every candidate. It outputs a validity logit (shown as sigmoid probability) and a predicted quality score.
5. `spacegen.rules.check_layout` independently applies H1 inside room, H2 no overlap, H3 door clearance, and H4 reachability. A CNN-positive layout still fails if any hard rule fails.
6. Valid layouts are ranked by CNN quality and a diversity filter selects at most three.

## CNN retained

`spacegen/models/evaluator.py` has four Conv-BatchNorm-ReLU-MaxPool blocks, then Flatten, Dropout, Dense-ReLU, and two linear output heads. `head_valid` is trained with binary cross-entropy with logits; `head_score` is trained with MSE quality loss. Their summed loss is optimized by Adam. The saved checkpoint has 585,682 parameters and was trained on the labelled Set B split (70/15/15 train/validation/test). `spacegen/evaluator_report.py` reports precision, recall, F1, confusion matrix, accuracy and ROC-AUC.

## Commands

```bash
.venv\Scripts\python.exe run.py app
.venv\Scripts\python.exe -m pytest -q tests/test_simple_pipeline.py tests/test_raster.py tests/test_rules.py
python run.py train-evaluator --epochs 30 --name my_cnn
python run.py evaluator-report --run my_cnn
```

## Files changed

- `spacegen/simple_pipeline.py`: small end-to-end candidate, CNN, rule-check, ranking workflow.
- `app/simple_streamlit_app.py`: focused Streamlit UI.
- `run.py`: `app` now opens the simplified UI; `simple-app` is an explicit alias.
- `tests/test_simple_pipeline.py`: inference and checkpoint-loading checks.
- `README.md`, `Architecture.md`: current workflow and commands.

## Deferred features

CVAE generation, latent optimization, five-method comparison, pinning, headline experiments, and 3D/experimental work are retained but not used in the demo. Pinning is intentionally unavailable in the simplified UI.

## Limitations to say honestly

The CNN recognizes patterns from its generated labelled training data; it is not a constraint solver. The rule checker makes the final decision. The furniture dimensions and prices are project assumptions, not commercial specifications. The current app supports the living-room catalog only.

## Short viva explanation

“The generator proposes layouts using understandable placement rules. Every layout becomes an image. The CNN learns spatial patterns such as overlap and door blockage through convolution, ReLU and pooling. During training, BCEWithLogitsLoss teaches valid versus invalid classification and MSE teaches the optional quality score; Adam updates weights through backpropagation. At inference I do not trust the CNN alone: the geometry checker verifies boundaries, overlap, door clearance and reachability. Finally, I rank the safe layouts with the CNN and show up to three different ones.”
