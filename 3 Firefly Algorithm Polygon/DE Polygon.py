# ============================================================
# DE-PAINTER (Parallel, Variable Polygons 3–7, Fast)
# - Reconstructs a target image with DE-optimized polygons
# - Variable polygon sides per stroke (3..7), filled color + alpha
# - Joblib parallelism (uses all CPU cores)
# ============================================================

import os, time, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw
from joblib import Parallel, delayed, cpu_count

# --------------------------- CONFIG ---------------------------
TARGET_PATH     = "target.jpg"        
W, H            = 256, 256            # small & fast
DPI_META        = (200, 200)          # metadata only
N_STROKES       = 700                 # strokes to add (more = better)
POP_SIZE        = 64                  # DE population
DE_GENS         = 50                  # DE generations per stroke
F_SCALE         = 0.65                # DE mutation factor (F)
CROSS_RATE      = 0.9                 # DE crossover rate (CR)
VEL_CLAMP       = 0.25                # not used in DE (kept for ref)
DOWNSCALE_FIT   = 2                   # compute fitness on downsampled
N_CORES         = max(1, cpu_count() - 1)
SEED            = int.from_bytes(os.urandom(8), "little")
RNG             = np.random.default_rng(SEED)

OUT_FINAL       = "de_poly_final.png"
OUT_STRIP       = "de_poly_strip.png"
SNAPSHOT_EVERY  = 40

# --------------------------- UTILS ----------------------------
def clamp01(a): return np.minimum(1.0, np.maximum(0.0, a))
def to_uint8(a): return (clamp01(a)*255.0 + 0.5).astype(np.uint8)

def load_target(path, size):
    im = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    return (np.asarray(im).astype(np.float32)/255.0)

def downsample(arr, f):
    if f <= 1: return arr
    h, w, c = arr.shape
    nh, nw = h//f, w//f
    return arr.reshape(nh, f, nw, f, c).mean(axis=(1,3))

def mse(a, b):
    d = a - b
    return float(np.mean(d*d))

# ---------------------- PARAM ENCODING -----------------------
# Vector layout (D = 1 + 14 + 4 = 19):
# [0]   k_raw in [3,7] (rounded to ints 3..7)
# [1:15]  (x1,y1, x2,y2, ..., x7,y7) in [0,1]
# [15:18] (r,g,b) in [0,1]
# [18]    alpha in [0.05,0.5]
LO = np.array([3.0] + [0.0]*14 + [0.0,0.0,0.0, 0.05], np.float32)
HI = np.array([7.0] + [1.0]*14 + [1.0,1.0,1.0, 0.50], np.float32)
D  = len(LO)

def decode_polygon(p, w, h):
    k = int(np.round(np.clip(p[0], 3.0, 7.0)))
    coords = []
    xy = p[1:15]
    for i in range(k):
        xi = float(xy[2*i]   * w)
        yi = float(xy[2*i+1] * h)
        coords.append((xi, yi))
    color = tuple(to_uint8(p[15:18]))
    alpha = float(p[18])
    return k, coords, color, alpha

# ----------------------- RENDERING ---------------------------
def render_polygon(canvas_rgb, p):
    h, w, _ = canvas_rgb.shape
    k, pts, col, al = decode_polygon(p, w, h)

    if k < 3:  # guard
        return canvas_rgb

    layer = Image.new("RGBA", (w, h), (0,0,0,0))
    draw  = ImageDraw.Draw(layer, "RGBA")
    draw.polygon(pts, fill=(col[0], col[1], col[2], int(al*255)))

    lay = np.asarray(layer).astype(np.float32)/255.0
    lay_rgb, lay_a = lay[...,:3], lay[...,3:4]
    return lay_rgb*lay_a + canvas_rgb*(1.0 - lay_a)

# ------------------------ FITNESS ---------------------------
def fitness_for_params(p, canvas_full, target_small, scale):
    out_full = render_polygon(canvas_full, p)
    out_small = downsample(out_full, scale)
    return mse(out_small, target_small), out_full

def _eval_particle(args):
    p, canvas_full, target_small, scale = args
    f, _ = fitness_for_params(p, canvas_full, target_small, scale)
    return f

# -------------------- DIFFERENTIAL EVOLUTION ----------------
def de_optimize_polygon(canvas_full, target_small, scale):
    # init population
    X = RNG.uniform(LO, HI, size=(POP_SIZE, D)).astype(np.float32)

    # evaluate initial (parallel)
    fvals = Parallel(n_jobs=N_CORES)(
        delayed(_eval_particle)((X[i], canvas_full, target_small, scale))
        for i in range(POP_SIZE)
    )
    F = np.array(fvals, np.float32)
    best_idx = int(np.argmin(F))
    best_p = X[best_idx].copy()
    best_f = float(F[best_idx])

    for _ in range(DE_GENS):
        trial = np.empty_like(X)
        for i in range(POP_SIZE):
            # mutation: choose 3 distinct indices a,b,c
            idxs = np.arange(POP_SIZE)
            RNG.shuffle(idxs)
            a, b, c = idxs[0], idxs[1], idxs[2]
            while a == i or b == i or c == i:
                RNG.shuffle(idxs)
                a, b, c = idxs[0], idxs[1], idxs[2]
            mutant = X[a] + F_SCALE * (X[b] - X[c])

            # crossover (binomial)
            cross_mask = RNG.random(D) < CROSS_RATE
            # ensure at least one dim from mutant
            cross_mask[RNG.integers(0, D)] = True
            t = np.where(cross_mask, mutant, X[i])

            # bound constraints
            trial[i] = np.minimum(HI, np.maximum(LO, t)).astype(np.float32)

        # evaluate trial pop (parallel)
        tvals = Parallel(n_jobs=N_CORES)(
            delayed(_eval_particle)((trial[i], canvas_full, target_small, scale))
            for i in range(POP_SIZE)
        )
        T = np.array(tvals, np.float32)

        # selection
        better = T < F
        X[better] = trial[better]
        F[better] = T[better]

        # track best
        bi = int(np.argmin(F))
        if F[bi] < best_f:
            best_f = float(F[bi]); best_p = X[bi].copy()

    # render best polygon onto full-res canvas
    _, painted = fitness_for_params(best_p, canvas_full, target_small, scale)
    return best_p, painted, best_f

# ---------------------- PROGRESS STRIP ----------------------
def make_strip(panels, save_path, gap=12, bg=1.0):
    if not panels: return
    h, w, _ = panels[0].shape
    out = np.ones((h, w*len(panels) + gap*(len(panels)-1), 3), np.float32) * bg
    for i, p in enumerate(panels):
        x0 = i*(w+gap)
        out[:, x0:x0+w, :] = p
    Image.fromarray(to_uint8(out)).save(save_path, dpi=DPI_META)

# ------------------------------ RUN -------------------------
def main():
    t0 = time.time()
    print("== DE-PAINTER (Polygons 3–7, Parallel) start ==")
    target_full = load_target(TARGET_PATH, (W, H))
    target_small = downsample(target_full, DOWNSCALE_FIT)
    canvas = np.ones_like(target_full)  # white

    panels = []
    err0 = mse(downsample(canvas, DOWNSCALE_FIT), target_small)
    print(f"Init error: {err0:.6f} | cores={N_CORES}")

    for s in range(1, N_STROKES+1):
        _, canvas, err = de_optimize_polygon(canvas, target_small, DOWNSCALE_FIT)
        if s % 5 == 0:
            cur = mse(downsample(canvas, DOWNSCALE_FIT), target_small)
            print(f"[{s:04d}/{N_STROKES}] err={cur:.6f}")

        if (s % SNAPSHOT_EVERY == 0) or (s == N_STROKES):
            panels.append(canvas.copy())

    Image.fromarray(to_uint8(canvas)).save(OUT_FINAL, dpi=DPI_META)
    make_strip(panels, OUT_STRIP, gap=10, bg=1.0)
    print(f"Saved {OUT_FINAL} & {OUT_STRIP} (dpi={DPI_META[0]})")
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
