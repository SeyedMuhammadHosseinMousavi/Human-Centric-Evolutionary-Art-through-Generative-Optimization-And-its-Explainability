# ============================================================
# HILL-PAINT (Rectangles + Random Local Search, Parallel)
# - Fastest evolutionary art baseline using rectangles
# - Approximates a target image via RLS with gradual refinement
# - Parallelized across all CPU cores
# ============================================================

import os, time, math, warnings, multiprocessing as mp
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw

# --------------------------- CONFIG ---------------------------
TARGET_PATH      = "target.jpg"     # Reference image path
WORK_W, WORK_H   = 256, 256         # working size for speed/quality balance
N_RECTS          = 400              # number of rectangles to evolve
ITER_PER_RECT    = 200              # refinement steps per rectangle
PARALLEL         = True             # enable parallel local search
OUTPUT_CANVAS    = "hill_rect_final.png"
OUTPUT_STRIP     = "hill_rect_strip.png"
DPI_META         = (300, 300)

RNG = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))

# ---------------------- Utility / IO ----------------------
def clamp01(a): return np.minimum(1.0, np.maximum(0.0, a))
def to_uint8(a): return (clamp01(a) * 255.0 + 0.5).astype(np.uint8)

def load_target(path, size):
    im = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr

def alpha_blend_rgb(base_rgb, over_rgb, alpha):
    return over_rgb * alpha + base_rgb * (1.0 - alpha)

# ---------------------- Rectangle Rendering ---------------------
# RECT params: [x, y, w, h, angle, r, g, b, alpha]
RECT_LO = np.array([0, 0, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0, 0.05], np.float32)
RECT_HI = np.array([1, 1, 0.5, 0.5, 2*math.pi, 1.0, 1.0, 1.0, 0.5], np.float32)

def render_rect(canvas_rgb, p):
    h, w, _ = canvas_rgb.shape
    cx = float(p[0]*w)
    cy = float(p[1]*h)
    rw = float(p[2]*w)
    rh = float(p[3]*h)
    ang = float(p[4])
    col = tuple(to_uint8(p[5:8]))
    al  = float(p[8])

    # create rectangle
    layer = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(layer, "RGBA")
    box = [cx - rw/2, cy - rh/2, cx + rw/2, cy + rh/2]
    draw.rectangle(box, fill=(col[0], col[1], col[2], int(al*255)))
    # rotate around center
    layer = layer.rotate(ang * 180.0/math.pi, resample=Image.BICUBIC, center=(cx, cy))
    lay = np.asarray(layer).astype(np.float32)/255.0
    return alpha_blend_rgb(canvas_rgb, lay[..., :3], lay[..., 3:4])

# ---------------------- Fitness (MSE) ----------------------
def mse(a, b):
    d = a - b
    return float(np.mean(d*d))

# ---------------------- Random Local Search ----------------------
def local_search_rect(canvas_rgb, target_rgb):
    """Hill climbing for one rectangle."""
    D = len(RECT_LO)
    p = RNG.uniform(RECT_LO, RECT_HI).astype(np.float32)
    best_f, best_im = fitness(p, canvas_rgb, target_rgb)
    for _ in range(ITER_PER_RECT):
        cand = p + RNG.normal(0, 0.05, size=D).astype(np.float32)
        cand = np.clip(cand, RECT_LO, RECT_HI)
        f, im = fitness(cand, canvas_rgb, target_rgb)
        if f < best_f:
            p, best_f, best_im = cand, f, im
    return p, best_im, best_f

def fitness(params, canvas_rgb, target_rgb):
    out = render_rect(canvas_rgb, params)
    return mse(out, target_rgb), out

def _worker_eval(args):
    canvas_rgb, target_rgb = args
    return local_search_rect(canvas_rgb, target_rgb)

# ---------------------- Progress Strip ----------------------
def make_strip(panels, save_path, gap=16, bg=1.0):
    if not panels: return
    h, w, _ = panels[0].shape
    out = np.ones((h, w*len(panels) + gap*(len(panels)-1), 3), np.float32) * bg
    for i, p in enumerate(panels):
        x0 = i*(w+gap)
        out[:, x0:x0+w, :] = p
    Image.fromarray(to_uint8(out)).save(save_path, dpi=DPI_META)

# ------------------------------ MAIN -----------------------------
def main():
    t0 = time.time()
    print("== HILL-PAINT (Rectangles + RLS) start ==")
    target = load_target(TARGET_PATH, (WORK_W, WORK_H))
    canvas = np.ones_like(target)
    panels = []

    err = mse(canvas, target)
    print(f"Init error: {err:.6f}")

    for s in range(1, N_RECTS+1):
        if PARALLEL:
            with mp.Pool() as pool:
                results = pool.map(_worker_eval, [(canvas, target)]*mp.cpu_count())
            best_p, best_img, best_f = min(results, key=lambda x: x[2])
        else:
            best_p, best_img, best_f = local_search_rect(canvas, target)

        canvas = best_img
        if s % 5 == 0:
            print(f"[{s:04d}/{N_RECTS}] err={best_f:.6f}")
        if (s % 50 == 0) or (s == N_RECTS):
            panels.append(canvas.copy())

    Image.fromarray(to_uint8(canvas)).save(OUTPUT_CANVAS, dpi=DPI_META)
    make_strip(panels, OUTPUT_STRIP)
    print(f"Saved: {OUTPUT_CANVAS}, {OUTPUT_STRIP}")
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
