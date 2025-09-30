# rectified_flow/preprocess/test_gpu.py
import time, torch, torch.nn.functional as F
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

device = "cuda"
N, Cin, H, W = 64, 64, 256, 256   # bump up/down if needed
Cout, K = 128, 3
iters = 100000

x = torch.randn(N, Cin, H, W, device=device)
w = torch.randn(Cout, Cin, K, K, device=device)

# warmup
for _ in range(10):
    y = F.conv2d(x, w, padding=1)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(iters):
    y = F.conv2d(x, w, padding=1)
torch.cuda.synchronize()
t1 = time.time()

print(f"OK. time={t1-t0:.2f}s, y.mean={y.mean().item():.4f}")
