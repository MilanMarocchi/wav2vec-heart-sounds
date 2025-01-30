import random

import numpy as np
import torch

SEED = 0x5EED  # ayy

# torch.use_deterministic_algorithms(True)

MAKE_REPRODUCIBLE = False

if MAKE_REPRODUCIBLE:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.benchmark = True

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


torch_gen = torch.Generator()
torch_gen.manual_seed(SEED)
