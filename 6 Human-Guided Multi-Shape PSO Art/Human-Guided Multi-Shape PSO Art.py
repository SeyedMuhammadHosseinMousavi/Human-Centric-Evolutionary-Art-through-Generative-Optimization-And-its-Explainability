# =====================================================================
# HUMAN-GUIDED MULTI-SHAPE PSO ART (FINAL + METRICS)
# =====================================================================

import os, time, math, warnings, multiprocessing as mp, signal
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
TARGET_PATH       = "target.jpg"
WORK_W, WORK_H    = 128, 128
N_SHAPES          = 160
PSO_ITERS         = 30
POP_SIZE          = 24
PARALLEL          = True
SAVE_CANVAS_STEPS = True
MODEL_SAVE_PATH   = "pso_human_multishape_metrics_model.npz"
DPI_META          = (100, 100)
N_FEEDBACKS       = 5
HUMAN_FEEDBACK    = True
ALPHA_FITNESS     = 0.85
SHAPE_TYPES       = ["rectangle","circle","triangle","line"]

RECT_LO = np.array([0,0,0.02,0.02,0.0,0.0,0.0,0.0,0.05],np.float32)
RECT_HI = np.array([1,1,0.5,0.5,2*math.pi,1.0,1.0,1.0,0.5],np.float32)

stop_requested=False
def handle_interrupt(signum=None, frame=None):
    global stop_requested
    stop_requested=True
    print("\n🛑 Stop signal received — finishing current iteration...")
signal.signal(signal.SIGINT, handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)

# ---------------- UTILS ----------------
def clamp01(a): return np.minimum(1.0, np.maximum(0.0, a))
def to_uint8(a): return (clamp01(a)*255+0.5).astype(np.uint8)
def mse(a,b): return float(np.mean((a-b)**2))
def load_target(path,size):
    im=Image.open(path).convert("RGB").resize(size,Image.LANCZOS)
    return np.asarray(im).astype(np.float32)/255.0
def alpha_blend_rgb(base,over,alpha): return over*alpha+base*(1-alpha)

# ---------------- METRICS ----------------
def compute_metrics(canvas, target, errors, delta_errors):
    final_mse = mse(canvas, target)
    psnr_val = psnr(target, canvas, data_range=1.0)
    ssim_val = ssim(target, canvas, channel_axis=2, data_range=1.0)
    mean_delta = np.mean(delta_errors) if delta_errors else 0.0
    std_delta  = np.std(delta_errors) if delta_errors else 0.0
    conv_rate  = (errors[0] - errors[-1]) / max(1, len(errors))
    return dict(
        final_mse=final_mse,
        psnr=psnr_val,
        ssim=ssim_val,
        mean_delta=mean_delta,
        std_delta=std_delta,
        conv_rate=conv_rate
    )

# ---------------- DRAW ----------------
def render_shape(canvas,p,shape_type):
    h,w,_=canvas.shape
    cx,cy=p[0]*w,p[1]*h
    rw,rh=p[2]*w,p[3]*h
    ang=p[4]; col=tuple(to_uint8(p[5:8])); al=p[8]
    layer=Image.new("RGBA",(w,h),(0,0,0,0))
    d=ImageDraw.Draw(layer,"RGBA")

    if shape_type=="rectangle":
        box=[cx-rw/2,cy-rh/2,cx+rw/2,cy+rh/2]
        d.rectangle(box,fill=(col[0],col[1],col[2],int(al*255)))
    elif shape_type=="circle":
        box=[cx-rw/2,cy-rh/2,cx+rw/2,cy+rh/2]
        d.ellipse(box,fill=(col[0],col[1],col[2],int(al*255)))
    elif shape_type=="triangle":
        pts=[(cx,cy-rh/2),(cx-rw/2,cy+rh/2),(cx+rw/2,cy+rh/2)]
        d.polygon(pts,fill=(col[0],col[1],col[2],int(al*255)))
    elif shape_type=="line":
        d.line([cx-rw/2,cy,cx+rw/2,cy],
               fill=(col[0],col[1],col[2],int(al*255)),
               width=max(1,int(rh)))
    layer=layer.rotate(ang*180/math.pi,resample=Image.BICUBIC,center=(cx,cy))
    lay=np.asarray(layer).astype(np.float32)/255.0
    return alpha_blend_rgb(canvas,lay[...,:3],lay[...,3:4])

# ---------------- FEEDBACK ----------------
def get_human_feedback(img,step):
    img_pil=Image.fromarray(to_uint8(img))
    path=f"preview_{step}.png"; img_pil.save(path); img_pil.show(title=f"Preview {step}")
    print(f"\n🧠 Feedback at step {step}")
    try:
        score=float(input("Rate image (0–1, 1=excellent): ") or "0.5")
        score=np.clip(score,0,1)
    except: score=0.5

    print("Choose area to improve: [1] Shape [2] Color [3] Opacity [4] Size [5] Skip")
    try: choice=int(input("Enter number (1–5): ") or "5")
    except: choice=5

    fb={"type":"none","detail":"none","ratio":1.0}
    if choice==1:
        print("   [0] Rectangle [1] Circle [2] Triangle [3] Line")
        idx=int(input("   Choose shape: ") or "0")
        fb["type"]="shape"; fb["detail"]=SHAPE_TYPES[idx%4]
        fb["ratio"]=float(input("   Ratio (0–1): ") or "1.0")
    elif choice==2:
        print("   [0] Warmer [1] Cooler [2] Contrast")
        cidx=int(input("   Choose: ") or "0")
        fb["type"]="color"; fb["detail"]=["warmer","cooler","contrast"][cidx%3]
        fb["ratio"]=float(input("   Ratio (0–1): ") or "1.0")
    elif choice==3:
        fb["type"]="opacity"; fb["detail"]=input("   inc/dec: ") or "inc"
    elif choice==4:
        fb["type"]="size"; fb["detail"]=input("   larger/smaller: ") or "larger"
    print(f"✅ Feedback recorded: {fb}")
    return score,fb

def apply_feedback_bias(p,fb):
    if fb["type"]=="color":
        r=fb["ratio"]
        if fb["detail"]=="warmer": p[5]=clamp01(p[5]+0.2*r)
        elif fb["detail"]=="cooler": p[7]=clamp01(p[7]+0.2*r)
        elif fb["detail"]=="contrast": p[5:8]=clamp01(0.5+(p[5:8]-0.5)*(1+0.4*r))
    elif fb["type"]=="opacity":
        p[8]=clamp01(p[8]+(0.15 if fb["detail"].startswith("inc") else -0.15))
    elif fb["type"]=="size":
        scale=1.25 if fb["detail"].startswith("larg") else 0.75
        p[2]=clamp01(p[2]*scale); p[3]=clamp01(p[3]*scale)
    return p

# ---------------- FITNESS + PSO ----------------
def fitness(params, canvas_rgb, target_rgb, shape_type):
    out = render_shape(canvas_rgb, params, shape_type)
    return mse(out, target_rgb), out

def pso_search(canvas_rgb, target_rgb, seed, shape_type, fb=None):
    """PSO-based optimization for one shape."""
    rng = np.random.default_rng(seed)
    D = len(RECT_LO)
    X = rng.uniform(RECT_LO, RECT_HI, size=(POP_SIZE, D)).astype(np.float32)
    if fb:
        for i in range(POP_SIZE):
            X[i] = apply_feedback_bias(X[i], fb)
    V = rng.normal(0, 0.1, size=(POP_SIZE, D)).astype(np.float32)

    pbest = X.copy()
    pbest_f = np.zeros(POP_SIZE)
    pbest_imgs = [None]*POP_SIZE
    for i in range(POP_SIZE):
        f, im = fitness(X[i], canvas_rgb, target_rgb, shape_type)
        pbest_f[i], pbest_imgs[i] = f, im
    gidx = int(np.argmin(pbest_f))
    gbest, gbest_f, gbest_im = pbest[gidx].copy(), pbest_f[gidx], pbest_imgs[gidx]

    for _ in range(PSO_ITERS):
        r1, r2 = rng.random((POP_SIZE, D)), rng.random((POP_SIZE, D))
        V = 0.72*V + 1.4*r1*(pbest - X) + 1.4*r2*(gbest - X)
        X = np.clip(X + V, RECT_LO, RECT_HI)
        for i in range(POP_SIZE):
            f, im = fitness(X[i], canvas_rgb, target_rgb, shape_type)
            if f < pbest_f[i]:
                pbest_f[i], pbest[i], pbest_imgs[i] = f, X[i].copy(), im
        gidx = int(np.argmin(pbest_f))
        if pbest_f[gidx] < gbest_f:
            gbest, gbest_f, gbest_im = pbest[gidx].copy(), pbest_f[gidx], pbest_imgs[gidx]
    return gbest, gbest_im, gbest_f

def _worker_eval(args):
    canvas_rgb, target_rgb, seed, shape_type, fb = args
    return pso_search(canvas_rgb, target_rgb, seed, shape_type, fb)

# ---------------- MAIN ----------------
def main():
    t0=time.time()
    print("== HUMAN-GUIDED MULTI-SHAPE PSO (METRICS) ==")

    target=load_target(TARGET_PATH,(WORK_W,WORK_H))
    canvas=np.ones_like(target)
    shapes,errors,deltas,imgs,feedbacks=[],[],[],[],[]
    err=mse(canvas,target)
    print(f"Initial error: {err:.6f}")

    current_shape="rectangle"
    fb_last=None
    feedback_interval=max(1,N_SHAPES//N_FEEDBACKS)

    for i in range(1,N_SHAPES+1):
        if stop_requested: break
        seeds=np.random.randint(0,99999999,size=mp.cpu_count())
        if PARALLEL:
            with mp.Pool() as pool:
                results=pool.map(_worker_eval,
                    [(canvas,target,s,current_shape,fb_last) for s in seeds])
            best_p,best_img,best_f=min(results,key=lambda x:x[2])
        else:
            best_p,best_img,best_f=pso_search(canvas,target,seeds[0],current_shape,fb_last)

        shapes.append((best_p,current_shape))
        deltas.append(err-best_f)
        errors.append(best_f)
        canvas=best_img

        if SAVE_CANVAS_STEPS: imgs.append(canvas.copy())
        if i%5==0: print(f"[{i:03d}/{N_SHAPES}] err={best_f:.6f}")

        if i % feedback_interval == 0 or i == N_SHAPES:
            metrics = compute_metrics(canvas, target, errors, deltas)
            print(f"\n📊 Metrics @ {i}: "
                  f"MSE={metrics['final_mse']:.6f}, "
                  f"PSNR={metrics['psnr']:.2f}, "
                  f"SSIM={metrics['ssim']:.4f}")

        if HUMAN_FEEDBACK and (i%feedback_interval==0 or i==N_SHAPES):
            score,fb=get_human_feedback(canvas,i)
            feedbacks.append((i,score,fb))
            if fb["type"]=="shape": current_shape=fb["detail"]
            else: fb_last=fb
            err=ALPHA_FITNESS*err+(1-ALPHA_FITNESS)*(1-score)

            # --- CONTINUOUS COLLAGE SAVE AT FEEDBACK ---
            if SAVE_CANVAS_STEPS and len(imgs) > 0:
                try:
                    n = 8  # number of snapshots to include
                    # determine new segment since last feedback
                    start_idx = (i - feedback_interval) if (i - feedback_interval) >= 0 else 0
                    end_idx = len(imgs)
                    segment = imgs[start_idx:end_idx]
            
                    if len(segment) >= n:
                        idxs = np.linspace(0, len(segment)-1, n, dtype=int)
                    else:
                        idxs = range(len(segment))
            
                    frames = [Image.fromarray(to_uint8(segment[j])) for j in idxs]
                    w, h = frames[0].width, frames[0].height
                    collage = Image.new("RGB", (w * len(frames), h))
                    for k, im in enumerate(frames):
                        collage.paste(im, (k * w, 0))
                    collage.save(f"progress_collage_step_{i}.png", dpi=DPI_META)
                    print(f"🖼️ Continuous collage saved — evolution from last feedback to step {i}")
                except Exception as e:
                    print(f"⚠️ Collage save failed: {e}")
            # --- END COLLAGE SAVE ---


    metrics = compute_metrics(canvas, target, errors, deltas)
    print("\n📈 Final Metrics:")
    for k,v in metrics.items():
        print(f"   {k:15s}: {v:.6f}")

    Image.fromarray(to_uint8(canvas)).save("canvas_final.png",dpi=DPI_META)
    np.savez_compressed(MODEL_SAVE_PATH,
        shapes=np.array([s[1] for s in shapes]),
        params=np.array([s[0] for s in shapes],dtype=np.float32),
        errors=np.array(errors,dtype=np.float32),
        delta=np.array(deltas,dtype=np.float32),
        human_feedback=feedbacks,
        metrics=metrics,
        canvas_final=canvas,
        canvas_snapshots=np.array(imgs,dtype=np.float32) if SAVE_CANVAS_STEPS else None)
    print(f"\n✅ Completed in {time.time()-t0:.1f}s | 💾 Saved model: {MODEL_SAVE_PATH}")

if __name__=="__main__":
    main()
