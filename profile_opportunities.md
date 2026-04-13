# Profile Opportunities From `nsys_exp_fused_fp8_softcap_ce.nsys-rep`

This file summarizes the largest GPU-time kernels seen in the steady-state `nsys` run of the
experimental fused FP8 LM-head + softcap + CE path with:

- `FP8_LM_HEAD=1`
- `EXPERIMENTAL_FUSED_FP8_SOFTCAP_CE=1`
- `FUSED_SOFTCAP_CE=0`
- capture window starting at step `200`

These percentages are from `nsys stats --report cuda_gpu_kern_sum`.

## Top Kernels

1. `cutlass::device_kernel<...FlashAttnBwdSm90...>`: `13.2%`
   - Area: FlashAttention backward
   - Room: likely low
   - Note: this is already a specialized FA3/SM90 path; not the place to hunt first

2. `linear_leaky_relu_square_kernel`: `12.9%`
   - Area: model-specific fused block kernel
   - Room: unknown, potentially moderate
   - Note: large enough to matter; worth profiling separately if logits/loss lane stalls out

3. `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT`: `8.3%`
   - Area: GEMM family
   - Room: unknown
   - Note: likely one of the main math kernels in the model; needs timeline mapping before action

4. `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_splitK_NTT`: `6.6%`
   - Area: GEMM family
   - Room: unknown
   - Note: same comment as above

5. `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT`: `5.6%`
   - Area: GEMM family
   - Room: unknown
   - Note: same comment as above

6. `cutlass::device_kernel<...FlashAttnFwdSm90...>`: `5.3%`
   - Area: FlashAttention forward
   - Room: low
   - Note: specialized FA3 path; probably not worth targeting first

7. `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL`: `2.0%`
   - Area: distributed comms
   - Room: low to moderate
   - Note: only worth attention if overlap/scheduling changes are on the table

8. `nvjet_tst_128x128_64x6_2x1_v_bz_splitK_NTT`: `1.9%`
   - Area: GEMM family
   - Room: unknown
   - Note: likely part of major matmul lanes; needs timeline mapping

9. `cutlass::device_kernel<...FlashAttnBwdPreprocess...>`: `1.9%`
   - Area: FlashAttention backward support
   - Room: low
   - Note: part of FA3 stack; probably not a custom optimization target

10. `nvjet_tst_128x256_64x4_2x1_v_bz_coopA_TNN`: `1.4%`
    - Area: GEMM family
    - Room: unknown
    - Note: another material math kernel

## Logits/Loss Lane

The explicit softcap+CE kernels in this run are:

- `fused_softcap_ce_bwd_kernel`: `1.2%`
- `fused_softcap_ce_fwd_kernel`: `0.9%`

Combined explicit CE share:

- about `2.1%`

Interpretation:

- the explicit CE kernels are real but not dominant at whole-step level
- the remaining potential upside in the logits/loss lane depends on whether some of the large
  `nvjet_*` kernels are substantially part of the LM-head projection path
- `nsys` summary alone does not label those GEMMs semantically, so a timeline mapping or targeted
  `ncu` pass is needed before attributing them to LM-head work

## Practical Ranking

Most credible optimization opportunities from this profile:

1. `linear_leaky_relu_square_kernel`
2. whichever `nvjet_*` GEMMs map to the LM-head projection or other large model matmuls
3. only then the explicit `fused_softcap_ce_*` kernels

Least likely high-alpha targets from this profile:

- FA3 forward/backward kernels
- FA3 preprocess/postprocess support kernels
- small elementwise/helper kernels
