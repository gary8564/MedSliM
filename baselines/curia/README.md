# Curia classifier baseline

Frozen 2D Curia features plus a lightweight 3D head (attention pooling and a
linear classifier), following the [Curia](https://arxiv.org/abs/2509.06830)
classifier recipe. The same idea as [AnyMC3D](https://arxiv.org/abs/2512.12887):
keep the 2D foundation model frozen and adapt to a 3D task with a small plugin.

This is a paper baseline, not part of the `med_slim` package. For Curia as a
MedSliM slice encoder, use `build_slice_encoder(name="curia")`.

```bash
python -m baselines.curia.classifier \
  --config baselines/curia/configs/kneeMRI.yml \
  --n-folds 3
```

Or submit `scripts/eval/curia_classifier.sh`. 
