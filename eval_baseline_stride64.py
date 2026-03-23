"""Quick no-TTT baseline eval with stride 64 at various depths."""
import math, os, time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import sentencepiece as spm
from train_gpt_noinject import (
    GPT, CastedLinear, build_sentencepiece_luts, load_validation_tokens,
    restore_low_dim_params_to_fp32,
)

def main():
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt_path = os.environ.get("CHECKPOINT_PATH", "ttt_checkpoint_noinject.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]
    state_dict = ckpt["model_state_dict"]

    seq_len = 2048
    eval_stride = 64
    vocab_size = model_cfg["vocab_size"]
    depths = [int(d) for d in os.environ.get("DEPTHS", "3,4,5,6,8,10").split(",")]

    model = GPT(
        vocab_size=model_cfg["vocab_size"], model_dim=model_cfg["model_dim"],
        num_heads=model_cfg["num_heads"], num_kv_heads=model_cfg["num_kv_heads"],
        mlp_mult=model_cfg["mlp_mult"], tie_embeddings=model_cfg["tie_embeddings"],
        tied_embed_init_std=model_cfg["tied_embed_init_std"],
        logit_softcap=model_cfg["logit_softcap"], rope_base=model_cfg["rope_base"],
        qk_gain_init=model_cfg["qk_gain_init"],
        bigram_vocab_size=model_cfg.get("bigram_vocab_size", 0),
        bigram_dim=model_cfg.get("bigram_dim", 128),
        rope_dims=model_cfg.get("rope_dims", 0),
        ln_scale=model_cfg.get("ln_scale", False),
        mean_depth=model_cfg.get("mean_depth", 3),
        num_entry_blocks=model_cfg.get("num_entry_blocks", 2),
        num_exit_blocks=model_cfg.get("num_exit_blocks", 2),
        num_recurrent_blocks=model_cfg.get("num_recurrent_blocks", 3),
    ).to(device).bfloat16()
    for mod in model.modules():
        if isinstance(mod, CastedLinear):
            mod.float()
    restore_low_dim_params_to_fp32(model)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sp = spm.SentencePieceProcessor(model_file="./data/tokenizers/fineweb_1024_bpe.model")
    val_tokens = load_validation_tokens(
        os.path.join("./data/datasets/fineweb10B_sp1024", "fineweb_val_*.bin"), seq_len
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, vocab_size, device
    )
    total_val = val_tokens.numel() - 1
    per_rank = total_val // world_size
    rank_start = rank * per_rank
    rank_end = (rank + 1) * per_rank if rank < world_size - 1 else total_val

    if master:
        print(f"Baseline stride-64 eval | tokens per rank: {rank_end - rank_start} | depths: {depths}")

    for depth in depths:
        model.n_recurrent_iters = depth
        nll_accum = torch.zeros((), device=device, dtype=torch.float64)
        bytes_accum = torch.zeros((), device=device, dtype=torch.float64)
        t0 = time.perf_counter()

        pos = rank_start
        win_count = 0
        while pos < rank_end:
            end = min(pos + seq_len, rank_end)
            wlen = end - pos
            if wlen < 1:
                break
            chunk = val_tokens[pos:end + 1].to(dtype=torch.int64, device=device)
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = model.tok_emb(x)
                if model.bigram is not None:
                    h = h + model.bigram(x)
                h = F.rms_norm(h, (h.size(-1),))
                h = model.smear(h)
                h0 = h
                for block in model.entry_blocks:
                    h = block(h, h0)
                for _ in range(model.n_recurrent_iters):
                    for block in model.recurrent_blocks:
                        h = block(h, h0)
                for block in model.exit_blocks:
                    h = block(h, h0)
                h = model.final_norm(h)
                if model.tie_embeddings:
                    logits = F.linear(h, model.tok_emb.weight)
                else:
                    logits = model.lm_head(h)
                logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
            nll = F.cross_entropy(logits.reshape(-1, vocab_size).float(), y.reshape(-1), reduction="none")
            s = 0 if pos == rank_start else max(wlen - eval_stride, 0)
            scored = nll[s:wlen]
            tgt = y[0, s:wlen]
            prev = x[0, s:wlen]
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
            nll_accum += scored.to(torch.float64).sum()
            bytes_accum += tb.sum()
            pos += eval_stride
            win_count += 1
            if master and win_count % 5000 == 0:
                running_bpb = (nll_accum.item() / math.log(2)) / max(bytes_accum.item(), 1)
                elapsed = time.perf_counter() - t0
                print(f"    depth={depth} win={win_count} bpb={running_bpb:.6f} elapsed={elapsed:.0f}s", flush=True)

        if distributed:
            dist.all_reduce(nll_accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(bytes_accum, op=dist.ReduceOp.SUM)
        bpb = (nll_accum.item() / math.log(2)) / bytes_accum.item() if bytes_accum.item() > 0 else float("inf")
        elapsed = time.perf_counter() - t0
        if master:
            print(f"  depth={depth}: BPB={bpb:.6f} ({elapsed:.0f}s)")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
