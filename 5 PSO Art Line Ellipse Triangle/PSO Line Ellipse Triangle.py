# ============================================================
# PSO-PAINT (Fast, Mixed Strokes, 300dpi) — Full Pipeline
# - Approximates a target image with PSO-optimized strokes
# - Uses MIXED stroke types per run: line, ellipse, triangle
# - Much faster defaults + CPU parallel per-stroke (multiprocessing)
# - Saves final canvas and a progress strip (both tagged 300 dpi)
# ============================================================

import os, time, math, warnings, multiprocessing as mp
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image, ImageDraw

# --------------------------- CONFIG ---------------------------
TARGET_PATH      = "target.jpg"    
WORK_W, WORK_H   = 256, 256         # working resolution (fast + good)
N_STROKES        = 300              # total strokes (fewer → faster)
POP_SIZE         = 32               # PSO particles per stroke
PSO_ITERS        = 25               # PSO iterations per stroke
SNAPSHOT_EVERY   = 50               # panels in progress strip
PARALLEL         = True             # per-iter particle evals in parallel

OUTPUT_CANVAS    = "pso_paint_final.png"
OUTPUT_STRIP     = "pso_paint_strip.png"
DPI_META         = (300, 300)       # PNG/JPG dpi metadata

# Stroke type probabilities (sum to 1.0)
P_LINE, P_ELLIPSE, P_TRI = 0.5, 0.3, 0.2

# RNG
RNG = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))

# ---------------------- Utility / IO ----------------------
def clamp01(a): return np.minimum(1.0, np.maximum(0.0, a))
def to_uint8(a): return (clamp01(a) * 255.0 + 0.5).astype(np.uint8)

def load_target(path, size):
    im = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr

# ---------------------- Stroke Bounds ---------------------
# We pick a stroke TYPE randomly each stroke.
# Then PSO optimizes ONLY that type’s parameter vector, using these bounds.

# LINE params: [x, y, len, angle, thickness, r, g, b, alpha]
LINE_LO = np.array([0, 0, 0.03, 0.0, 0.002, 0.0, 0.0, 0.0, 0.05], np.float32)
LINE_HI = np.array([1, 1, 0.40, 2*math.pi, 0.030, 1.0, 1.0, 1.0, 0.40], np.float32)

# ELLIPSE params: [cx, cy, rx, ry, angle, r, g, b, alpha]
ELL_LO  = np.array([0, 0, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.05], np.float32)
ELL_HI  = np.array([1, 1, 0.30, 0.20, 2*math.pi, 1.0, 1.0, 1.0, 0.40], np.float32)

# TRIANGLE params: [x1,y1, x2,y2, x3,y3, r,g,b, alpha]
TRI_LO  = np.array([0,0, 0,0, 0,0, 0.0,0.0,0.0, 0.05], np.float32)
TRI_HI  = np.array([1,1, 1,1, 1,1, 1.0,1.0,1.0, 0.40], np.float32)

# --------------------- Alpha Compositing -------------------
def alpha_blend_rgb(base_rgb, over_rgb, alpha):  # float32 in [0..1]
    return over_rgb * alpha + base_rgb * (1.0 - alpha)

# --------------------- Stroke Rendering -------------------
def render_line(canvas_rgb, p):
    h, w, _ = canvas_rgb.shape
    x  = float(p[0] * w)
    y  = float(p[1] * h)
    L  = float(p[2] * max(w, h))
    a  = float(p[3])
    th = float(max(1.0, p[4] * max(w, h)))
    col = tuple(to_uint8(p[5:8]))
    al  = float(p[8])

    x2 = x + L * math.cos(a)
    y2 = y + L * math.sin(a)

    layer = Image.new("RGBA", (w, h), (0,0,0,0))
    draw  = ImageDraw.Draw(layer, "RGBA")
    draw.line((x, y, x2, y2), fill=(col[0], col[1], col[2], int(al*255)), width=int(th), joint="curve")

    lay = np.asarray(layer).astype(np.float32)/255.0
    return alpha_blend_rgb(canvas_rgb, lay[..., :3], lay[..., 3:4])

def render_ellipse(canvas_rgb, p):
    h, w, _ = canvas_rgb.shape
    cx = float(p[0] * w)
    cy = float(p[1] * h)
    rx = float(max(1.0, p[2] * w))
    ry = float(max(1.0, p[3] * h))
    ang = float(p[4])
    col = tuple(to_uint8(p[5:8]))
    al  = float(p[8])

    # axis-aligned bbox before rotation
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]

    # draw filled ellipse on a layer; approximate rotation by rotating layer
    layer = Image.new("RGBA", (w, h), (0,0,0,0))
    draw  = ImageDraw.Draw(layer, "RGBA")
    draw.ellipse(bbox, fill=(col[0], col[1], col[2], int(al*255)))

    # rotate around center (expand=False keeps size)
    layer = layer.rotate(ang * 180.0/math.pi, resample=Image.BICUBIC, center=(cx, cy))
    lay = np.asarray(layer).astype(np.float32)/255.0
    return alpha_blend_rgb(canvas_rgb, lay[..., :3], lay[..., 3:4])

def render_triangle(canvas_rgb, p):
    h, w, _ = canvas_rgb.shape
    x1, y1 = float(p[0]*w), float(p[1]*h)
    x2, y2 = float(p[2]*w), float(p[3]*h)
    x3, y3 = float(p[4]*w), float(p[5]*h)
    col = tuple(to_uint8(p[6:9]))
    al  = float(p[9])

    layer = Image.new("RGBA", (w, h), (0,0,0,0))
    draw  = ImageDraw.Draw(layer, "RGBA")
    draw.polygon([(x1,y1), (x2,y2), (x3,y3)], fill=(col[0], col[1], col[2], int(al*255)))

    lay = np.asarray(layer).astype(np.float32)/255.0
    return alpha_blend_rgb(canvas_rgb, lay[..., :3], lay[..., 3:4])

def render_stroke(canvas_rgb, stroke_type, params):
    if stroke_type == "line":
        return render_line(canvas_rgb, params)
    elif stroke_type == "ellipse":
        return render_ellipse(canvas_rgb, params)
    else:  # "tri"
        return render_triangle(canvas_rgb, params)

# ---------------------- Fitness (MSE) ----------------------
def mse(a, b):  # a,b float32 [0..1]
    d = a - b
    return float(np.mean(d*d))

def fitness_for_params(params, stroke_type, canvas_rgb, target_rgb):
    out = render_stroke(canvas_rgb, stroke_type, params)
    return mse(out, target_rgb), out

# -------- Multiprocessing helper: avoid big copies -----------
def _worker_eval(args):
    params, stroke_type, canvas_rgb, target_rgb = args
    f, _ = fitness_for_params(params, stroke_type, canvas_rgb, target_rgb)
    return f

# -------------------------- PSO core -----------------------
def pso_optimize_stroke(canvas_rgb, target_rgb, stroke_type, lo, hi):
    D = len(lo)
    X = RNG.uniform(lo, hi, size=(POP_SIZE, D)).astype(np.float32)
    V = RNG.normal(0, 0.1, size=(POP_SIZE, D)).astype(np.float32)

    # init personal best
    if PARALLEL:
        with mp.Pool() as pool:
            fvals = pool.map(_worker_eval, [(X[i], stroke_type, canvas_rgb, target_rgb) for i in range(POP_SIZE)])
        pbest_f = np.array(fvals, dtype=np.float32)
    else:
        pbest_f = np.zeros(POP_SIZE, np.float32)
        for i in range(POP_SIZE):
            pbest_f[i], _ = fitness_for_params(X[i], stroke_type, canvas_rgb, target_rgb)

    pbest = X.copy()
    gi = int(np.argmin(pbest_f))
    gbest = pbest[gi].copy()
    gbest_f = float(pbest_f[gi])

    for _ in range(PSO_ITERS):
        r1 = RNG.random((POP_SIZE, D)).astype(np.float32)
        r2 = RNG.random((POP_SIZE, D)).astype(np.float32)
        V = 0.72*V + 1.6*r1*(pbest - X) + 1.6*r2*(gbest - X)
        V = np.clip(V, -0.25, 0.25)
        X = np.clip(X + V, lo, hi)

        if PARALLEL:
            with mp.Pool() as pool:
                fvals = pool.map(_worker_eval, [(X[i], stroke_type, canvas_rgb, target_rgb) for i in range(POP_SIZE)])
            f = np.array(fvals, dtype=np.float32)
        else:
            f = np.zeros(POP_SIZE, np.float32)
            for i in range(POP_SIZE):
                f[i], _ = fitness_for_params(X[i], stroke_type, canvas_rgb, target_rgb)

        improved = f < pbest_f
        pbest[improved] = X[improved]
        pbest_f[improved] = f[improved]
        gi = int(np.argmin(pbest_f))
        if pbest_f[gi] < gbest_f:
            gbest_f = float(pbest_f[gi])
            gbest = pbest[gi].copy()

    _, painted = fitness_for_params(gbest, stroke_type, canvas_rgb, target_rgb)
    return gbest, painted, gbest_f

# --------------------- Progress strip builder -------------------
def make_strip(panels, save_path, gap=16, bg=1.0):
    if not panels: return
    h, w, _ = panels[0].shape
    out = np.ones((h, w*len(panels) + gap*(len(panels)-1), 3), np.float32) * bg
    for i, p in enumerate(panels):
        x0 = i*(w+gap)
        out[:, x0:x0+w, :] = p
    Image.fromarray(to_uint8(out)).save(save_path, dpi=DPI_META)

# ------------------------------ RUN -----------------------------
def sample_stroke_type():
    r = RNG.random()
    if r < P_LINE: return "line", LINE_LO, LINE_HI
    r -= P_LINE
    if r < P_ELLIPSE: return "ellipse", ELL_LO, ELL_HI
    return "tri", TRI_LO, TRI_HI

def main():
    t0 = time.time()
    print("== PSO-PAINT (Fast, Mixed Strokes) start ==")
    target = load_target(TARGET_PATH, (WORK_W, WORK_H))
    canvas = np.ones_like(target)  # white canvas

    panels = []
    err = mse(canvas, target)
    print(f"Init error: {err:.6f}")

    for s in range(1, N_STROKES+1):
        stype, lo, hi = sample_stroke_type()
        _, canvas, err = pso_optimize_stroke(canvas, target, stype, lo, hi)

        if s % 5 == 0:
            print(f"[{s:04d}/{N_STROKES}] {stype:<7} | err={err:.6f}")

        if (s % SNAPSHOT_EVERY == 0) or (s == N_STROKES):
            panels.append(canvas.copy())

    # Save final canvas
    Image.fromarray(to_uint8(canvas)).save(OUTPUT_CANVAS, dpi=DPI_META)
    make_strip(panels, OUTPUT_STRIP, gap=20, bg=1.0)

    print(f"Saved: {OUTPUT_CANVAS} (300 dpi), {OUTPUT_STRIP} (300 dpi)")
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
