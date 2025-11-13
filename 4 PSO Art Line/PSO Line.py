# ============================================================
# PSO-PAINT: Particle-Swarm Stroke Painting (progress strip)
# Style: PSO-based stroke painter that approximates a target image.
# Each run is cumulative: add one optimized stroke at a time.
# ------------------------------------------------------------
# Requirements: pillow, numpy, matplotlib (optional for final save)
# Parallel: multiprocessing for fitness eval batches (toggle PARALLEL)
# Printing: clear stage logs; warnings suppressed.
# ============================================================

import os, time, warnings, math, multiprocessing as mp
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ------------------------- CONFIG -------------------------
TARGET_PATH      = "target.jpg"   
OUTPUT_STRIP     = "pso_paint_strip.png"
W, H             = 512, 512       # working resolution (smaller = faster)
DOWNSCALE_FIT    = 2              # fitness computed on (W//D, H//D)
N_STROKES        = 600            # total strokes to paint
POP_SIZE         = 64             # PSO particles per stroke
PSO_ITERS        = 40             # PSO iterations per stroke
SNAPSHOT_EVERY   = 75             # save a panel every K strokes
PARALLEL         = True           # multiprocessing for fitness per iter
RNG              = np.random.default_rng(int.from_bytes(os.urandom(8),'little'))

# PSO hyperparams
W_INERTIA        = 0.72
C1               = 1.6
C2               = 1.6
VEL_CLAMP        = 0.2

# Stroke parameter bounds (normalized)
# [x, y, len, angle, thickness, r, g, b, alpha]
BOUNDS_LO = np.array([0, 0, 0.02, 0.0, 0.002, 0.0, 0.0, 0.0, 0.05], dtype=np.float32)
BOUNDS_HI = np.array([1, 1, 0.40, 2*math.pi, 0.030, 1.0, 1.0, 1.0, 0.40], dtype=np.float32)

# ----------------------------------------------------------
def clamp01(a): return np.minimum(1.0, np.maximum(0.0, a))
def to_uint8(a): return (clamp01(a)*255.0 + 0.5).astype(np.uint8)

def load_target(path, size):
    im = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32)/255.0
    return arr

def downsample(arr, f):
    if f <= 1: return arr
    h, w, c = arr.shape
    nh, nw = h//f, w//f
    return arr.reshape(nh, f, nw, f, c).mean(axis=(1,3))

# ----------------- Alpha compositing (numpy) ----------------
def alpha_blend_rgb(base_rgb, over_rgb, alpha):  # all float32 [0..1]
    # out = over*alpha + base*(1-alpha)
    return over_rgb*alpha + base_rgb*(1.0 - alpha)

# ----------------------- Stroke renderer --------------------
def render_stroke_on(canvas_rgb, params):
    """
    Draw one anti-aliased RGBA line stroke onto a copy of canvas.
    params: [x, y, len, angle, thickness, r,g,b,alpha] in normalized units.
    """
    h, w, _ = canvas_rgb.shape
    x = float(params[0] * w)
    y = float(params[1] * h)
    L = float(params[2] * max(w, h))
    ang = float(params[3])
    th = float(max(1.0, params[4] * max(w, h)))  # px
    col = tuple(to_uint8(params[5:8]))
    al  = float(params[8])

    x2 = x + L*math.cos(ang)
    y2 = y + L*math.sin(ang)

    # Draw to separate RGBA layer
    layer = Image.new("RGBA", (w, h), (0,0,0,0))
    draw  = ImageDraw.Draw(layer, "RGBA")
    draw.line((x, y, x2, y2), fill=(col[0], col[1], col[2], int(al*255)), width=int(th), joint="curve")

    lay = np.asarray(layer).astype(np.float32)/255.0
    lay_rgb = lay[..., :3]
    lay_a   = lay[..., 3:4]

    out = alpha_blend_rgb(canvas_rgb, lay_rgb, lay_a)
    return out

# ---------------------- Fitness function --------------------
def fitness_of_params(params, canvas_small, target_small, scale_factor):
    # Render on a high-res canvas clone, then downsample for fitness
    canvas_hr = canvas_small.copy() if scale_factor==1 else None
    # For speed, render at high-res canvas directly (we’ll pass full-res)
    out_hr = render_stroke_on(canvas_small, params)
    out_small = downsample(out_hr, scale_factor)
    diff = (out_small - target_small)
    return float(np.mean(diff*diff)), out_hr

def _eval_particle(args):
    params, canvas_full, target_small, scale = args
    f, _ = fitness_of_params(params, canvas_full, target_small, scale)
    return f

# -------------------------- PSO loop ------------------------
def pso_optimize_stroke(canvas_full, target_small, scale_factor):
    D = len(BOUNDS_LO)
    # init swarm in normalized box
    X = RNG.uniform(BOUNDS_LO, BOUNDS_HI, size=(POP_SIZE, D)).astype(np.float32)
    V = RNG.normal(0, 0.1, size=(POP_SIZE, D)).astype(np.float32)
    pbest = X.copy()
    # Evaluate initial
    if PARALLEL:
        with mp.Pool() as pool:
            fvals = pool.map(_eval_particle, [(X[i], canvas_full, target_small, scale_factor) for i in range(POP_SIZE)])
        pbest_f = np.array(fvals, dtype=np.float32)
    else:
        pbest_f = np.zeros(POP_SIZE, np.float32)
        for i in range(POP_SIZE):
            pbest_f[i], _ = fitness_of_params(X[i], canvas_full, target_small, scale_factor)

    gbest_idx = int(np.argmin(pbest_f))
    gbest = pbest[gbest_idx].copy()
    gbest_f = float(pbest_f[gbest_idx])

    for it in range(PSO_ITERS):
        # velocity & position
        r1 = RNG.random((POP_SIZE, D)).astype(np.float32)
        r2 = RNG.random((POP_SIZE, D)).astype(np.float32)
        V = (W_INERTIA*V + C1*r1*(pbest - X) + C2*r2*(gbest - X))
        V = np.clip(V, -VEL_CLAMP, VEL_CLAMP)
        X = X + V
        X = np.minimum(BOUNDS_HI, np.maximum(BOUNDS_LO, X))

        # evaluate
        if PARALLEL:
            with mp.Pool() as pool:
                fvals = pool.map(_eval_particle, [(X[i], canvas_full, target_small, scale_factor) for i in range(POP_SIZE)])
            f = np.array(fvals, dtype=np.float32)
        else:
            f = np.zeros(POP_SIZE, np.float32)
            for i in range(POP_SIZE):
                f[i], _ = fitness_of_params(X[i], canvas_full, target_small, scale_factor)

        # update bests
        improved = f < pbest_f
        pbest[improved] = X[improved]
        pbest_f[improved] = f[improved]
        gi = int(np.argmin(pbest_f))
        if pbest_f[gi] < gbest_f:
            gbest_f = float(pbest_f[gi])
            gbest = pbest[gi].copy()

    # Return best stroke and its rendered canvas
    _, painted = fitness_of_params(gbest, canvas_full, target_small, scale_factor)
    return gbest, painted, gbest_f

# --------------------- Progress strip builder -------------------
def make_strip(panels, save_path, gap=32, bg=1.0):
    if len(panels) == 0: return
    h, w, c = panels[0].shape
    strip = np.ones((h, w*len(panels) + gap*(len(panels)-1), 3), np.float32) * bg
    for i, p in enumerate(panels):
        x0 = i*(w+gap)
        strip[:, x0:x0+w, :] = p
    Image.fromarray(to_uint8(strip)).save(save_path)

# ------------------------------ RUN -----------------------------
def main():
    t0 = time.time()
    print("== PSO-PAINT start ==")
    target_full = load_target(TARGET_PATH, (W, H))
    target_small = downsample(target_full, DOWNSCALE_FIT)
    canvas = np.ones_like(target_full)  # white canvas

    panels = []
    best_err = np.mean((downsample(canvas, DOWNSCALE_FIT) - target_small)**2)
    print(f"Init error: {best_err:.6f}")

    for s in range(1, N_STROKES+1):
        stroke, canvas_new, err = pso_optimize_stroke(canvas, target_small, DOWNSCALE_FIT)
        canvas = canvas_new
        if s % 5 == 0:
            cur_err = np.mean((downsample(canvas, DOWNSCALE_FIT) - target_small)**2)
            print(f"[{s:04d}/{N_STROKES}] stroke added | err={cur_err:.6f}")

        if (s % SNAPSHOT_EVERY == 0) or (s == N_STROKES):
            panels.append(canvas.copy())

    # Build the evolution strip
    make_strip(panels, OUTPUT_STRIP, gap=24, bg=1.0)
    print(f"== Saved progress strip: {OUTPUT_STRIP}")
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
