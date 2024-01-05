import torch
import numpy as np
from sdf.network_tcnn import SDFNetwork

checkpoint = torch.load("test_out/0/checkpoints/ngp.pth.tar")
model = SDFNetwork()
model.load_state_dict(checkpoint['model'])
model.eval()
dummy_input = torch.randn(1000000, 3, dtype=torch.float).cuda()

# init loggers
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))

for _ in range(10):
    _ = model(dummy_input)

with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

print(np.min(timings))