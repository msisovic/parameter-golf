# Fused Lane Estimate

Using the profiled training shape per rank:

- `M = 98304`
- `K = 512`
- `V = 8192`

Dense logits shape:

- `logits = [M, V] = [98304, 8192]`
- elements: `805,306,368`
- BF16 size: about `1.61 GB`

## Current Dense Intermediate Traffic

Approximate dense `logits` / `grad_logits` traffic in the current "kinda fused" lane:

1. projection writes `logits`: `1.61 GB`
2. CE fwd reads `logits`: `1.61 GB`
3. CE bwd reads `logits`: `1.61 GB`
4. CE bwd writes `grad_logits`: `1.61 GB`
5. `grad_input` reads `grad_logits`: `1.61 GB`
6. `grad_weight` reads `grad_logits`: `1.61 GB`

Total:

- about `9.66 GB` per rank per step

At rough H100 HBM peak bandwidth of `~3.0 TB/s`, the pure bandwidth floor for that traffic is:

- `9.66 GB / 3.0 TB/s ~= 3.22 ms`

So the current lane is paying at least about `3.2 ms` just from dense intermediate traffic.

## Recompute Cost

One projection-sized GEMM:

- `2 * M * K * V ~= 0.825 TFLOP`

A true no-logit-store backward would likely add about one extra projection’s worth of recompute.

Reasonable lower-bound estimate for that extra recompute on Hopper:

- about `0.4 ms` to `0.8 ms`

## Net Upside Estimate

If true full fusion removes the dense `logits` / `grad_logits` traffic and pays recompute instead:

- traffic removed floor: about `3.2 ms`
- recompute added: about `0.4` to `0.8 ms`
- net recovered: about `2.4` to `2.8 ms`

If step time is about `120 ms`, that implies:

- about `2.0%` to `2.3%` end-to-end

At about `4800` steps in budget, that is roughly:

- `+95` to `+110` steps

## Caveat

This is a roofline-style estimate, not a hardware-counter proof.

It uses:

- real tensor sizes
- `nsys` timing for the explicit CE kernels
- first-principles byte counting for the dense intermediate traffic

It does not yet use `ncu`, so it should be treated as a plausible upper-mid estimate rather than a confirmed measurement.
