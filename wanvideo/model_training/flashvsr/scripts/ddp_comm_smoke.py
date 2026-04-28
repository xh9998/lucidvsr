import os
import socket
import time

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    start = time.time()
    dist.init_process_group(backend="nccl")
    x = torch.ones(1, device=f"cuda:{local_rank}") * (rank + 1)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    expected = world_size * (world_size + 1) / 2
    elapsed = time.time() - start
    if rank == 0:
        print(
            f"[ddp_comm_smoke] host={socket.gethostname()} world_size={world_size} "
            f"sum={float(x.item())} expected={expected} elapsed={elapsed:.2f}s",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
