# gpu_smoke_conv.py
import time, torch, torch.nn.functional as F

assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # ok to omit

device = "cuda"
# Big input to keep SMs busy
N, Cin, H, W = 128, 64, 256, 256
Cout, K = 256, 3
iters = 100000000

# Preallocate once (avoid CPU work each step)
x = torch.randn(N, Cin, H, W, device=device)
w = torch.randn(Cout, Cin, K, K, device=device)

# Warmup
for _ in range(10):
    y = F.conv2d(x, w, padding=1)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(iters):
    y = F.conv2d(x, w, padding=1)
torch.cuda.synchronize()
t1 = time.time()

pixels = H*W
macs_per_out = Cin*K*K
ops_per_conv = 2 * N * Cout * pixels * macs_per_out  # 2 for MAC -> FLOPs
tflops = (ops_per_conv * iters) / (t1 - t0) / 1e12

print(f"Time: {t1-t0:.2f}s  |  ~{tflops:.1f} TFLOP/s  |  y.mean={y.mean().item():.4f}")
