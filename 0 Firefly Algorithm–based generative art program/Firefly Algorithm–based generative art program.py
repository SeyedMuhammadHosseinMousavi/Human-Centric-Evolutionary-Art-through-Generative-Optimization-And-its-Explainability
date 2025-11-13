# ============================================================
# FIREFLY-AESTHETIC (Parallel, Fast, 150 dpi)
# Generates abstract evolutionary art using the Firefly Algorithm.
# ============================================================

import os, time, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
from joblib import Parallel, delayed, cpu_count

# --------------------------- CONFIG ---------------------------
W, H = 512, 512
DPI = (150, 150)
POP_SIZE = 40
FF_ITERS = 100
GAMMA = 1.0           # Light absorption coefficient
BETA0 = 1.5           # Base attraction
ALPHA = 0.2           # Randomness factor
N_CORES = max(1, cpu_count() - 1)
RNG = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))

# --------------------- PARAMETER SPACE -----------------------
LO = np.array([0.0, 0.10, 0.0, 0.20, 0.00, 0.50, 1, 1, 1, 1,
               0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 1.0, 0.0], dtype=np.float32)
HI = np.array([1.0, 0.90, 1.0, 1.60, 2.00, 3.00, 10,10,10,10,
               6.0, 6.0, 6.0, 6.0, 2*np.pi,2*np.pi,2*np.pi,2*np.pi, 12.0, 0.80], dtype=np.float32)

# --------------------- COLOR UTILITIES -----------------------
def hsv_to_rgb(h, s, v):
    i = np.floor(h*6).astype(int)
    f = h*6 - i
    p = v*(1 - s)
    q = v*(1 - f*s)
    t = v*(1 - (1-f)*s)
    i_mod = i % 6
    r = np.select([i_mod==0, i_mod==1, i_mod==2, i_mod==3, i_mod==4, i_mod==5],
                  [v, q, p, p, t, v])
    g = np.select([i_mod==0, i_mod==1, i_mod==2, i_mod==3, i_mod==4, i_mod==5],
                  [t, v, v, q, p, p])
    b = np.select([i_mod==0, i_mod==1, i_mod==2, i_mod==3, i_mod==4, i_mod==5],
                  [p, p, t, v, v, q])
    return np.stack([r,g,b], axis=-1)

def palette(h, base, spread, rot):
    h1 = (base + h*spread) % 1.0
    h2 = (h1 + 0.33 + rot*0.1) % 1.0
    h3 = (h1 + 0.66 - rot*0.1) % 1.0
    w1 = 0.5 + 0.5*np.cos(2*np.pi*h + 0.0)
    w2 = 0.5 + 0.5*np.cos(2*np.pi*h + 2.1)
    w3 = 1.0 - np.maximum(w1,w2)*0.4
    wsum = w1+w2+w3 + 1e-6
    hmix = (h1*w1 + h2*w2 + h3*w3)/wsum
    s = 0.65 + 0.35*np.cos(2*np.pi*(h+rot))
    v = 0.55 + 0.45*np.sin(2*np.pi*(h+0.25))
    return hsv_to_rgb(hmix%1.0, np.clip(s,0,1), np.clip(v,0,1))

# --------------------- PATTERN RENDERER ---------------------
def render(params, w=W, h=H):
    (base, spread, rot, rdecay, swirl, spow,
     k1,k2,k3,k4, f1,f2,f3,f4, p1,p2,p3,p4, sharp, vign) = params

    y, x = np.linspace(-1,1,h), np.linspace(-1,1,w)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2) + 1e-6
    th = np.arctan2(yy, xx)
    th_s = th + swirl * (rr**spow)

    ang = (np.cos(k1*th_s + p1) +
           np.cos(k2*th_s + p2) +
           np.cos(k3*th_s + p3) +
           np.cos(k4*th_s + p4)) / 4.0
    rad = (np.cos(f1*rr + p1) +
           np.cos(f2*rr + p2) +
           np.cos(f3*rr + p3) +
           np.cos(f4*rr + p4)) / 4.0

    combined = np.tanh(sharp * (0.55*ang + 0.45*rad))
    combined *= np.exp(-rdecay * rr)
    v = (combined - combined.min()) / (np.ptp(combined) + 1e-6)

    rgb = palette(v, base, spread, rot)
    vig = 1.0 - vign*(rr**1.5)
    rgb = np.clip(rgb * vig[...,None], 0, 1)
    return rgb.astype(np.float32)

# --------------------- AESTHETIC METRICS -------------------
def symmetry_score(img):
    diff_h = np.mean(np.abs(img - img[:, ::-1, :]))
    diff_v = np.mean(np.abs(img - img[::-1, :, :]))
    return 1.0 - 0.5*(diff_h + diff_v)

def edge_flow_score(img):
    g = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2])
    Gy, Gx = np.gradient(g)
    mag = np.sqrt(Gx*Gx + Gy*Gy)
    dens = np.mean(mag > (mag.mean() + mag.std()))
    return 1.0 - abs(dens - 0.18)

def color_var_score(img):
    var = np.mean(np.var(img.reshape(-1,3), axis=0))
    return 1.0 - np.exp(-6.0*var)

def thirds_contrast_score(img):
    h, w, _ = img.shape
    x1, x2 = w//3, 2*w//3
    y1, y2 = h//3, 2*h//3
    g = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2])
    c = 0.0
    for xs in (x1, x2):
        c += np.mean(np.abs(np.diff(g[:, max(xs-1,0):min(xs+1,w)], axis=1)))
    for ys in (y1, y2):
        c += np.mean(np.abs(np.diff(g[max(ys-1,0):min(ys+1,h), :], axis=0)))
    c /= 4.0
    return 1.0 - np.exp(-12.0*c)

def aesthetic_fitness(img):
    w_sym, w_edge, w_col, w_third = 0.30, 0.25, 0.25, 0.20
    s1 = symmetry_score(img)
    s2 = edge_flow_score(img)
    s3 = color_var_score(img)
    s4 = thirds_contrast_score(img)
    return w_sym*s1 + w_edge*s2 + w_col*s3 + w_third*s4

# ------------------------ FIREFLY ALGORITHM ------------------
def evaluate(params):
    img = render(params)
    return aesthetic_fitness(img), img

def firefly_optimize():
    D = len(LO)
    X = np.clip(RNG.uniform(LO, HI, size=(POP_SIZE, D)).astype(np.float32), LO, HI)

    # Initial evaluation (parallel)
    results = Parallel(n_jobs=N_CORES)(delayed(evaluate)(X[i]) for i in range(POP_SIZE))
    scores, images = zip(*results)
    scores = np.array(scores, np.float32)

    print(f"Init fitness: {np.max(scores):.4f} | using {N_CORES} cores")

    for it in range(1, FF_ITERS+1):
        # Move less bright fireflies toward brighter ones
        for i in range(POP_SIZE):
            for j in range(POP_SIZE):
                if scores[j] > scores[i]:
                    r = np.linalg.norm(X[i] - X[j])
                    beta = BETA0 * np.exp(-GAMMA * r*r)
                    step = beta * (X[j] - X[i]) + ALPHA * RNG.normal(0, 0.05, size=D)
                    X[i] = np.clip(X[i] + step, LO, HI)

        # Re-evaluate (parallel)
        results = Parallel(n_jobs=N_CORES)(delayed(evaluate)(X[i]) for i in range(POP_SIZE))
        scores, images = zip(*results)
        scores = np.array(scores, np.float32)

        if it % max(1, FF_ITERS//6) == 0:
            print(f"[{it:03d}/{FF_ITERS}] best={np.max(scores):.4f}")

    best_idx = int(np.argmax(scores))
    return X[best_idx], float(scores[best_idx]), images[best_idx]

# ------------------------- UNIQUE FILENAME -------------------
def unique_final_name(prefix="firefly_aesthetic_final", ext=".png"):
    # Use timestamp to avoid overwrite (same folder)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{ts}{ext}"
    # In rare case of same-second collision, add a counter
    if not os.path.exists(name):
        return name
    k = 1
    while True:
        alt = f"{prefix}_{ts}_{k}{ext}"
        if not os.path.exists(alt):
            return alt
        k += 1

# ------------------------------ RUN -----------------------------
def main():
    t0 = time.time()
    print("== FIREFLY-AESTHETIC (Parallel) start ==")
    gbest, best_score, best_img = firefly_optimize()
    final_name = unique_final_name()  # saves ONLY one final image, unique name
    Image.fromarray((np.clip(best_img,0,1)*255+0.5).astype(np.uint8)).save(final_name, dpi=DPI)
    print(f"Saved {final_name} (150 dpi) | fitness={best_score:.4f}")
    print(f"Done in {time.time()-t0:.1f}s | used {N_CORES} cores")

if __name__ == "__main__":
    main()
