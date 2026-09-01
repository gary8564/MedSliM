# AnyMC3D classifier baseline

Frozen 2D foundation-model slice features plus AnyMC3D task-query pooling and a
linear classifier ([AnyMC3D](https://arxiv.org/abs/2512.12887), Eq. 5):

```
a = softmax(H q / sqrt(d))
v = a^T H
```

The 2D backbone stays frozen in the MedSliM feature cache. There is no sequence
encoder (no Mamba2 / ABMIL) and no LoRA.

This is a paper baseline, not part of the `med_slim` package. It is not the
Curia reproduction (`baselines/curia`): Curia uses Curia tokens and MHA pooling.

MRI-CORE CLS features are the default. Any other single-FM MedSliM cache works
if `embed_dim` matches that encoder.

```bash
python -m baselines.anymc3d.classifier \
  --config baselines/anymc3d/configs/kneeMRI.yml \
  --n-folds 3
```
