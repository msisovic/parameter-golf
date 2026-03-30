## Quick note

- Disabling layer-0 attention looks safe and saves time. Its learned `attn_scale` was only about `0.08`, versus roughly `0.4-0.8` in later layers.
- Increasing depth still helped with layer-0 attention removed, which supports the conclusion that extra layers are still buying useful capacity.
- Logged metrics for the layer-0-attention-disabled run:
  `DIAGNOSTIC post_ema val_loss:1.9102 val_bpb:1.1313 eval_time:2140ms`
  `final_int8_zlib_roundtrip_exact val_loss:1.87718387 val_bpb:1.11177697`
- New baseline comparison:
  `DIAGNOSTIC post_ema val_loss:1.9146 val_bpb:1.1340 eval_time:2062ms`
  `final_int8_zlib_roundtrip_exact val_loss:1.88156874 val_bpb:1.11437394`
- Relative to the new baseline, disabling layer-0 attention improved post-EMA by `0.0027` bpb and final roundtrip by `0.0026` bpb.
- Final recurrence rebase result:
  `DIAGNOSTIC post_ema val_loss:1.9117 val_bpb:1.1322 eval_time:2306ms`
  `final_int8_zlib_roundtrip_exact val_loss:1.88039677 val_bpb:1.11367983`
- Relative to the fresh baseline, this is `0.0018` bpb better post-EMA and `0.0007` bpb better on the final roundtrip metric.
- Full untie of the repeated layer-4 MLP improved further:
  `DIAGNOSTIC post_ema val_loss:1.9077 val_bpb:1.1299 eval_time:2310ms`
  `final_int8_zlib_roundtrip_exact val_loss:1.87680398 val_bpb:1.11155197`
- Relative to the fresh baseline, this is `0.0041` bpb better post-EMA and `0.0028` bpb better on the final roundtrip metric.
- Relative to the previous recurrence rebase result, this is another `0.0023` bpb better post-EMA and `0.0021` bpb better on the final roundtrip metric.
- 10-layer reallocation test (`NUM_LAYERS=10`, `RECUR_LAYERS=3,4,5`, `REPEAT_UNTIE_MLP=down` on all three repeated layers) regressed:
  `DIAGNOSTIC post_ema val_loss:1.9113 val_bpb:1.1320 eval_time:2327ms`
  `final_int8_zlib_roundtrip_exact val_loss:1.88045827 val_bpb:1.11371625`
- Relative to the fresh baseline, this is only `0.0020` bpb better post-EMA and `0.0007` bpb better on the final roundtrip metric.
- Relative to the 11-layer full-untie recurrence result, it is `0.0021` bpb worse post-EMA and `0.0022` bpb worse on the final roundtrip metric.
- It also still missed the artifact cap: `Total submission size int6+lzma: 16041646 bytes`, which is `41,646` bytes over the `16,000,000`-byte limit.
- Conclusion: reallocating one physical layer into a wider recurrent band (`3,4,5`) does not preserve the gain from the sharper 11-layer `4,5` setup. The next experiments should stay on the 11-layer `RECUR_LAYERS=4,5` geometry and work on byte-efficient specialization or pruning/export targeting the real artifact cap.
- Repeated-pass LoRA on top of the size-safe 11-layer recurrence setup also regressed badly:
  `DIAGNOSTIC post_ema val_loss:1.9133 val_bpb:1.1332 eval_time:2325ms`
- Config: `NUM_LAYERS=11`, `RECUR_LAYERS=4,5`, `REPEAT_UNTIE_MLP=down`, `REPEAT_UNTIE_MLP_LAYERS=4`, `REPEAT_LORA_RANK=4`, `REPEAT_LORA_ALPHA=4`, `REPEAT_LORA_LR=3e-4`, `WARMDOWN_ITERS=3000`, `SEED=314`.
- Relative to the fresh baseline, this is only `0.0008` bpb better post-EMA. Relative to the non-LoRA 11-layer recurrence result (`1.1322`) it is `0.0010` worse, and relative to the full repeated-MLP untie result (`1.1299`) it is `0.0033` worse.
- The run log stops after GPTQ starts, so no final artifact metric was captured, but the training-side result is already clearly non-competitive.
- Conclusion: repeated-pass LoRA is not recovering the lost capacity here and should be deprioritized.
