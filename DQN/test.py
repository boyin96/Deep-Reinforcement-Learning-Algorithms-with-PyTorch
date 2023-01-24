from torch.utils.tensorboard import SummaryWriter
import numpy as np




writer = SummaryWriter()
x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)
writer.close()