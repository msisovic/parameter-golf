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
