import functools
import io
import os

import torch
import torch.distributed as dist
import hostlist

_LOCAL_PROCESS_GROUP = None


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """

    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")

    return dist.group.WORLD


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    cpu_group = None
    if os.getenv("MDETR_CPU_REDUCE") == "1":
        cpu_group = _get_global_gloo_group()

    buffer = io.BytesIO()
    torch.save(data, buffer)
    data_view = buffer.getbuffer()
    device = "cuda" if cpu_group is None else "cpu"
    tensor = torch.ByteTensor(data_view).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    size_list = [
        torch.tensor([0], device=device, dtype=torch.long) for _ in range(world_size)
    ]
    if cpu_group is None:
        dist.all_gather(size_list, local_size)
    else:
        print("gathering on cpu")
        dist.all_gather(size_list, local_size, group=cpu_group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    assert isinstance(local_size.item(), int)
    local_size = int(local_size.item())

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device=device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    if cpu_group is None:
        dist.all_gather(tensor_list, tensor)
    else:
        dist.all_gather(tensor_list, tensor, group=cpu_group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        tensor = torch.split(tensor, [size, max_size - size], dim=0)[0]
        buffer = io.BytesIO(tensor.cpu().numpy())
        obj = torch.load(buffer)
        data_list.append(obj)

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    # print("Finding World_size in reduce_dict: ",world_size)
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        # print("Finding NAN in reduce_dict: ",values)
        dist.all_reduce(values)
        
        #debug NaN
        # print("After all_reduce Finding NAN in reduce_dict: ",values)
        # non_nan_values = values[~torch.isnan(values)]
        # mean_non_nan = torch.mean(non_nan_values)
        # values[torch.isnan(values)] = mean_non_nan
        # ##########


        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """
    Returns:
        True if distributed training is enabled
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Returns:
        The number of processes in the process group
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Returns:
        The rank of the current process within the global process group.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process():
    """Return true if the current process is the main one"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Utility function to save only from the main process"""
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """Initialize distributed training, if appropriate"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
        # CLUSTER SPECIFIC
        args.world_size = int(os.environ["SLURM_NTASKS"])
        hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
        gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")
        os.environ["MASTER_ADDR"] = hostnames[0]
        os.environ["MASTER_PORT"] = str(
            12345 + int(min(gpu_ids))
        )  # to avoid port conflict on the same node
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output