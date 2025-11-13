# ============================================================
# GA-PAINT (Triangles Only, Fast, 300dpi)
# - Reconstructs a target image using GA-optimized triangles
# - Uses multiprocessing for parallel fitness evaluation
# - Small resolution for faster computation
# ============================================================

import os, time, math, warnings, multiprocessing as mp
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw

# --------------------------- CONFIG ---------------------------
TARGET_PATH      = "target.jpg"     # target image path
WORK_W, WORK_H   = 256, 256         # working resolution (fast + good)
N_STROKES        = 300              # total triangles (fewer → faster)
POP_SIZE         = 32               # GA population per stroke
GA_ITERS         = 25               # generations per stroke
MUT_RATE         = 0.2              # mutation rate
SNAPSHOT_EVERY   = 50               # panels in progress strip
PARALLEL         = True             # enable multiprocessing
OUTPUT_CANVAS    = "ga_triangle_final.png"
OUTPUT_STRIP     = "ga_triangle_strip.png"
DPI_META         = (300, 300)

# RNG
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

# ---------------------- Triangle Params ---------------------
# TRI params: [x1,y1, x2,y2, x3,y3, r,g,b, alpha]
TRI_LO = np.array([0,0, 0,0, 0,0, 0.0,0.0,0.0, 0.05], np.float32)
TRI_HI = np.array([1,1, 1,1, 1,1, 1.0,1.0,1.0, 0.40], np.float32)

# --------------------- Triangle Rendering -------------------
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

# ---------------------- Fitness (MSE) ----------------------
def mse(a, b):
    d = a - b
    return float(np.mean(d*d))

def fitness_for_params(params, canvas_rgb, target_rgb):
    out = render_triangle(canvas_rgb, params)
    return mse(out, target_rgb), out

def _worker_eval(args):
    params, canvas_rgb, target_rgb = args
    f, _ = fitness_for_params(params, canvas_rgb, target_rgb)
    return f

# -------------------------- GA Core -------------------------
def ga_optimize_triangle(canvas_rgb, target_rgb):
    D = len(TRI_LO)
    pop = RNG.uniform(TRI_LO, TRI_HI, size=(POP_SIZE, D)).astype(np.float32)

    # Initial fitness
    if PARALLEL:
        with mp.Pool() as pool:
            fvals = pool.map(_worker_eval, [(pop[i], canvas_rgb, target_rgb) for i in range(POP_SIZE)])
        fitness = np.array(fvals, np.float32)
    else:
        fitness = np.zeros(POP_SIZE, np.float32)
        for i in range(POP_SIZE):
            fitness[i], _ = fitness_for_params(pop[i], canvas_rgb, target_rgb)

    for _ in range(GA_ITERS):
        # Selection (tournament)
        parents = []
        for _ in range(POP_SIZE):
            i, j = RNG.integers(0, POP_SIZE, 2)
            parents.append(pop[i] if fitness[i] < fitness[j] else pop[j])
        parents = np.array(parents)

        # Crossover (blend)
        children = np.empty_like(parents)
        for i in range(0, POP_SIZE, 2):
            p1, p2 = parents[i], parents[(i+1) % POP_SIZE]
            alpha = RNG.random(D)
            children[i] = alpha*p1 + (1-alpha)*p2
            children[(i+1)%POP_SIZE] = alpha*p2 + (1-alpha)*p1

        # Mutation
        mut_mask = RNG.random(children.shape) < MUT_RATE
        children = np.clip(children + mut_mask*RNG.normal(0, 0.05, children.shape), TRI_LO, TRI_HI)

        # Evaluate children
        if PARALLEL:
            with mp.Pool() as pool:
                fvals = pool.map(_worker_eval, [(children[i], canvas_rgb, target_rgb) for i in range(POP_SIZE)])
            new_fit = np.array(fvals, np.float32)
        else:
            new_fit = np.zeros(POP_SIZE, np.float32)
            for i in range(POP_SIZE):
                new_fit[i], _ = fitness_for_params(children[i], canvas_rgb, target_rgb)

        # Elitism: keep best half from combined pool
        all_pop = np.vstack((pop, children))
        all_fit = np.hstack((fitness, new_fit))
        idx = np.argsort(all_fit)
        pop = all_pop[idx[:POP_SIZE]]
        fitness = all_fit[idx[:POP_SIZE]]

    # Return best
    best_idx = int(np.argmin(fitness))
    best = pop[best_idx]
    _, painted = fitness_for_params(best, canvas_rgb, target_rgb)
    return best, painted, float(fitness[best_idx])

# --------------------- Progress strip builder -------------------
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
    print("== GA-PAINT (Triangles Only) start ==")
    target = load_target(TARGET_PATH, (WORK_W, WORK_H))
    canvas = np.ones_like(target)
    panels = []

    err = mse(canvas, target)
    print(f"Init error: {err:.6f}")

    for s in range(1, N_STROKES+1):
        _, canvas, err = ga_optimize_triangle(canvas, target)

        if s % 5 == 0:
            print(f"[{s:04d}/{N_STROKES}] err={err:.6f}")

        if (s % SNAPSHOT_EVERY == 0) or (s == N_STROKES):
            panels.append(canvas.copy())

    Image.fromarray(to_uint8(canvas)).save(OUTPUT_CANVAS, dpi=DPI_META)
    make_strip(panels, OUTPUT_STRIP, gap=20, bg=1.0)

    print(f"Saved: {OUTPUT_CANVAS}, {OUTPUT_STRIP} (300 dpi)")
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
