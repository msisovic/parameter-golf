# Experiment Results (8xH100, New GPTQ Baseline)

## Goal

Retune the delayed dual-recurrence recipe against the AR self-generated GPTQ + BigramHash `3072x112` stack.

Fixed settings for this sweep:

- Seed: `314`
- `BIGRAM_VOCAB_SIZE=3072`
- `BIGRAM_DIM=112`
- `TTT_ENABLED=0` during training runs
- Sweep `RECUR_START_STEP == WARMDOWN_ITERS` over `{2500, 3000, 3500}`

## Experiment 1: Equal Start/Warmdown Sweep

| Setup | Log | Final post-EMA BPB | Final int6 sliding-window BPB | Post-TTT BPB | Notes |
|-------|-----|--------------------|--------------------------------|--------------|-------|
| `RECUR_START_STEP=2500`, `WARMDOWN_ITERS=2500` | [sweep_logs/newbase_s314_eq2500.log](/root/parameter-golf/sweep_logs/newbase_s314_eq2500.log) | `1.1356` | `1.11615505` | pending | Clearly worse than 3000/3500. Earlier activation hurts. |
| `RECUR_START_STEP=3000`, `WARMDOWN_ITERS=3000` | [sweep_logs/newbase_s314_eq3000.log](/root/parameter-golf/sweep_logs/newbase_s314_eq3000.log) | `1.1344` | `1.11550547` | pending | Best of the seed-314 equal sweep, but still worse than the earlier seed-1337-style `1.11471611`. |
| `RECUR_START_STEP=3500`, `WARMDOWN_ITERS=3500` | [sweep_logs/newbase_s314_eq3500.log](/root/parameter-golf/sweep_logs/newbase_s314_eq3500.log) | `1.1343` | `1.11589748` | pending | Slightly worse than 3000 despite slightly better post-EMA. |

### Short Findings

- Equal `2500/2500`, `3000/3000`, `3500/3500` confirms the old center still holds: `3000/3000` is best among these three.
- The gain is not recovered by simple retuning alone on seed `314`; all three are worse than the earlier `1.11471611` no-TTT run.
- `3500/3500` improving post-EMA but regressing after GPTQ suggests the remaining gap is not purely a training-side issue.

## Experiment 2: Broad XSA, Skip Recurrent Layers

| Setup | Log | Final post-EMA BPB | Final int6 sliding-window BPB | Post-TTT BPB | Notes |
|-------|-----|--------------------|--------------------------------|--------------|-------|
| `RECUR_START_STEP=3000`, `WARMDOWN_ITERS=3000`, `XSA_LAST_N=13`, `XSA_SKIP_RECUR=1` | [logs/xsa_all_skip_recur.txt](/root/parameter-golf/logs/xsa_all_skip_recur.txt) | `1.1339` | `1.11477270` | `1.11452572` | Best structural tweak so far; broad XSA helps, but not on recurrent layers. |
| Same as above, rerun with scale dump | [logs/xsa_all_skip_recur_scales.txt](/root/parameter-golf/logs/xsa_all_skip_recur_scales.txt) | `1.1335` | `1.11425194` | pending | Faster rerun with slightly better final score; likely runtime variance, not an intentional training-path change. |

### Scale Dump Notes

- `attn_scale` and `mlp_scale` are smooth and positive across all virtual layers; there is no obvious dead branch.
- Repeated recurrent passes show a mild MLP increase relative to their first pass, but attention remains substantial:
  - phys 4 first pass: `attn=0.4100`, `mlp=0.2414`
  - phys 4 repeated: `attn=0.3858`, `mlp=0.2845`
  - phys 5 first pass: `attn=0.3548`, `mlp=0.2574`
  - phys 5 repeated: `attn=0.5062`, `mlp=0.3033`
- `final_int6` scale dumps exactly match `post_ema`, which is expected because these per-block scale vectors are not part of the int6-quantized weights.

### Updated Findings

- Skipping XSA on recurrent layers while using broad XSA elsewhere is the clearest recurrence-specific win on the new baseline.
- The scale dump does not support a simple story that repeated passes are "MLP only"; attention still carries a large residual weight on recurrent passes.
- The improved rerun score (`1.11425194`) is real, but the speedup appears to be run-to-run/runtime variance rather than a known code change in the training path.

## Experiment 3: Layer 0 MLP-Only (Provisional)

| Setup | Log | Status | Notes |
|-------|-----|--------|-------|
| `LAYER0_MLP_ONLY=1` on top of `XSA_LAST_N=13`, `XSA_SKIP_RECUR=1` | [logs/layer0_mlp_only_xsa_skip_recur.txt](/root/parameter-golf/logs/layer0_mlp_only_xsa_skip_recur.txt) | partial | Early training signal shows no obvious quality collapse and a real pre-recurrence speedup (`80.48ms` at step `3000` vs `85.06ms` baseline). The post-activation region is noisy because the recurrent compiled path changes. |

## Experiment 4: New Anchor Defaults + Skip Dump

| Setup | Log | Final post-EMA BPB | Final int6 sliding-window BPB | Notes |
|-------|-----|--------------------|--------------------------------|-------|
| `BIGRAM=3072x112`, `XSA_LAST_N=13`, `XSA_SKIP_RECUR=1`, `LAYER0_MLP_ONLY=1` | [logs/anchor_skip_dump.txt](/root/parameter-golf/logs/anchor_skip_dump.txt) | `1.1334` | `1.11450899` | New anchor run with block-scale and skip-weight dump. Faster and simpler than the earlier broad-XSA rerun, but slightly worse than the lucky `1.11425194` run. |

### Anchor Dump Notes

- Layer 0 attention can be removed without obvious collapse; pre-recurrence step time drops to about `80.46ms` by step `3000`.
- The dump strongly suggests the weakest skip paths are the shallowest encoder skips, not the recurrent skips.
- Learned skip weights at `post_ema`:
  - `skip:00 = 0.2382`
  - `skip:01 = 0.3264`
  - `skip:02 = 0.3859`
  - `skip:03 = 0.2941`
  - `skip:04 = 0.0909`
  - `skip:05 = 0.0327`
- Under the default recurrence topology for `RECUR_LAYERS=4,5`, these correspond to:
  - `skip:00` repeated phys `4 <- 5`
  - `skip:01` repeated phys `5 <- 4`
  - `skip:02` phys `6 <- 3`
  - `skip:03` phys `7 <- 2`
  - `skip:04` phys `8 <- 1`
  - `skip:05` phys `9 <- 0`
- Interpretation:
  - recurrent skips are real and used
  - the strongest non-recurrent later skips are the mid-depth ones (`3` and `2`)
  - the shallowest skips (`1` and `0`) are nearly dead

## Experiment 5: Fixed Skip Topology

| Setup | Log | Final post-EMA BPB | Final int6 sliding-window BPB | Notes |
|-------|-----|--------------------|--------------------------------|-------|
| Anchor-like setup plus `FIXED_SKIP_TOPOLOGY=1` | [logs/fixed_skip_topology_xsa_skip_recur.txt](/root/parameter-golf/logs/fixed_skip_topology_xsa_skip_recur.txt) | `1.1339` | `1.11501332` | The baseline-preserving skip remap did not help. It was slightly behind before quantization and worse after GPTQ. |

### Fixed Skip Takeaway

- Skip-topology discontinuity still feels like a real conceptual issue, but this particular baseline-canonical remap is not the answer.
- The stronger remaining skip hypothesis is a recurrence-local/self-anchor scheme, not further tuning of the old U-Net stack routing.

## Experiment 6: Recurrence Self-Skip Only (Started)

| Setup | Log | Status | Notes |
|-------|-----|--------|-------|
| `RECUR_SELF_SKIP=1`, `RECUR_SELF_SKIP_ONLY=1` on the new anchor | [logs/recur_self_skip_only.txt](/root/parameter-golf/logs/recur_self_skip_only.txt) | running | Each repeated recurrent layer gets its own first-pass activation; generic decoder skips are disabled. Early training is fast (`80.00ms` at step `3000`) and the recurrence transition cost looks comparable to the anchor so far. |

### Current Working Conclusions

- The best clean anchor right now is:
  - `BIGRAM_VOCAB_SIZE=3072`
  - `BIGRAM_DIM=112`
  - `XSA_LAST_N=13`
  - `XSA_SKIP_RECUR=1`
  - `LAYER0_MLP_ONLY=1`
  - `RECUR_START_STEP=3000`
  - `WARMDOWN_ITERS=3000`
- Recurrence is still doing real work: recurrent skips are not dead.
- The generic skip stack is probably too broad; the model wants recurrent-local and mid-depth skips, not shallow ones.
