import torch
import torch.distributed as dist
import os
from util.utils import AverageMeter


def main():
    param = AverageMeter()
    num_gpus = torch.cuda.device_count()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    
    print("current rank is {}, world size is {}".format(rank, world_size))
    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )
    print(dist.get_world_size())
    
    x = torch.tensor([1,2]).cuda()
    
    dist.all_reduce(x)

    param.update(x / world_size)
    print(param.sum, param.count)


if __name__ == "__main__":
    main()