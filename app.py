#!/usr/bin/env python3
"""
Small Gradio UI for Boogu-Image-0.1 (Base / Turbo / Edit) on a multi-GPU server.

It shells out to the proven inference.py / inference_turbo.py CLIs (same code path as
run_boogu.sh) so behaviour matches the working command line exactly. The model is
(re)loaded per generation in a subprocess — load is cheap (~a few s, mmap); inference
dominates.

Launch:  cd ~/Boogu-Image && . .venv/bin/activate && python app.py
Then open the printed http://<server-ip>:<port> from any machine on the LAN.
"""
import os, sys, socket, subprocess, time, datetime, shutil, tempfile, math, threading
import gradio as gr
from PIL import Image as PILImage

REPO = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUTDIR = os.path.join(REPO, "outputs", "ui")
os.makedirs(OUTDIR, exist_ok=True)

MODELS = {
    "Turbo (fast, 4 steps)": dict(path="models/Boogu-Image-0.1-Turbo", script="inference_turbo.py", steps=4,  cfg=1.0, edit=False),
    "Base (T2I, dense text)": dict(path="models/Boogu-Image-0.1-Base",  script="inference.py",       steps=28, cfg=4.0, edit=False),
    "Edit (image editing)":   dict(path="models/Boogu-Image-0.1-Edit",  script="inference.py",       steps=20, cfg=5.0, edit=True),
}
EDIT_LABEL = "Edit (image editing)"

# --- GPU auto-selection: most-free, flash-attn-capable (skip Turing RTX 8000) ----------
def pick_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.free",
             "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return "0"
    best, best_free = "0", -1
    for line in out.strip().splitlines():
        idx, name, free = [x.strip() for x in line.split(",")]
        if "RTX 8000" in name or "Tesla K" in name:  # not flash-attn (sm<80)
            continue
        if int(free) > best_free:
            best, best_free = idx, int(free)
    return best

def list_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.free",
             "--format=csv,noheader,nounits"], text=True)
        return [f"{i.split(',')[0].strip()}: {i.split(',')[1].strip()} ({i.split(',')[2].strip()} MiB free)"
                for i in out.strip().splitlines()]
    except Exception:
        return []

def gpu_free_mib(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(idx), "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0

# Preferred GPU index (with CUDA_DEVICE_ORDER=PCI_BUS_ID this == the nvidia-smi index).
# "auto" lets the pool pick the most-free flash-attn-capable card. Override via env.
PINNED_GPU = os.environ.get("BOOGU_PIN_GPU", "auto")
# Min free VRAM (MiB) a card needs to host the 10B model (model-cpu-offload peak ~22 GB).
MIN_FREE_MIB = int(os.environ.get("BOOGU_MIN_FREE_MIB", "23000"))

# --- GPU pool: lets parallel generations land on DIFFERENT free cards ------------------
_gpu_lock = threading.Lock()
_in_use = set()

def _query_gpus():
    """Return {idx: (name, free_mib)} from nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.free",
         "--format=csv,noheader,nounits"], text=True)
    d = {}
    for line in out.strip().splitlines():
        idx, name, free = [x.strip() for x in line.split(",")]
        d[idx] = (name, int(free))
    return d

def _flash_ok(name):
    return ("RTX 8000" not in name) and ("Tesla K" not in name) and ("Tesla P" not in name)

def claim_gpu(prefer=None):
    """Reserve a free, flash-attn-capable GPU (preferring `prefer`); None if all busy."""
    with _gpu_lock:
        try:
            g = _query_gpus()
        except Exception:
            return prefer or "0"
        def usable(idx):
            return (idx in g and idx not in _in_use
                    and g[idx][1] >= MIN_FREE_MIB and _flash_ok(g[idx][0]))
        if prefer and usable(prefer):
            _in_use.add(prefer); return prefer
        cands = sorted(((g[i][1], i) for i in g if usable(i)), reverse=True)
        if cands:
            _in_use.add(cands[0][1]); return cands[0][1]
        if prefer and prefer in g and _flash_ok(g[prefer][0]) and prefer not in _in_use:
            _in_use.add(prefer); return prefer   # last resort: use prefer even if tight
        return None

def release_gpu(idx):
    with _gpu_lock:
        _in_use.discard(idx)

def usable_gpu_count():
    try:
        g = _query_gpus()
        return max(1, sum(1 for i in g if g[i][1] >= MIN_FREE_MIB and _flash_ok(g[i][0])))
    except Exception:
        return 1

# How many generations run at once (capped by usable free GPUs). Override via env.
MAX_PARALLEL = min(int(os.environ.get("BOOGU_MAX_PARALLEL", "4")), usable_gpu_count())

def free_port(start=int(os.environ.get("BOOGU_PORT", "8771")), end=8870):
    for p in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", p)) != 0:   # nothing listening -> free
                return p
    raise RuntimeError(f"no free port in {start}-{end}")

# --- dimensions -----------------------------------------------------------------------
# The pipeline requires H/W to be multiples of 16 (vae_scale_factor 8 * patch 2).
DIM_MULT = 16

def snap_dim(x, mult=DIM_MULT, lo=256, hi=2048):
    x = int(round(float(x) / mult) * mult)
    return max(lo, min(hi, x))

def fit_to_ratio(img_path, target_px=1024 * 1024, mult=DIM_MULT, lo=512, hi=2048):
    """Output (W,H) matching the input image's aspect ratio at ~target_px, snapped to mult."""
    try:
        with PILImage.open(img_path) as im:
            iw, ih = im.size
        ar = iw / ih
        h = math.sqrt(target_px / ar)
        return snap_dim(h * ar, mult, lo, hi), snap_dim(h, mult, lo, hi)
    except Exception:
        return 1024, 1024

# --- generation -----------------------------------------------------------------------
def generate(model_label, prompt, image, steps, cfg, image_cfg, width, height, seed,
             offload, gpu_choice, match_ratio, progress=gr.Progress(track_tqdm=False)):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt / instruction.")
    # A supplied image must be used -> always route to the Edit model.
    if image and not MODELS[model_label]["edit"]:
        model_label = EDIT_LABEL
    m = MODELS[model_label]
    if m["edit"] and not image:
        raise gr.Error("Edit mode needs an input image — upload one.")

    explicit = None if gpu_choice == "auto" else gpu_choice.split(":")[0].strip()
    gpu = claim_gpu(prefer=explicit)   # reserves a free card so parallel jobs don't collide
    if gpu is None:
        raise gr.Error("All usable GPUs are busy right now — please retry in a moment.")
    try:
        # >=40 GB free -> whole 10B model resident (no offload); else offload (cheap, the
        # transformer stays resident during denoising — only MLLM/VAE swap once).
        use_offload = bool(offload) if gpu_free_mib(gpu) < 40000 else False
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUTDIR, f"{ts}.png")

        # Edit: the pipeline sets output size from the REFERENCE image (only ever downscales),
        # so resize the reference to the matched (x16) dims for a ~1 MP output at input ratio.
        input_path = image
        if m["edit"] and match_ratio and image:
            width, height = fit_to_ratio(image)
            try:
                with PILImage.open(image) as im:
                    im = im.convert("RGB").resize((width, height), PILImage.LANCZOS)
                input_path = os.path.join(OUTDIR, f"{ts}_in.png")
                im.save(input_path)
            except Exception:
                input_path = image
        width, height = snap_dim(width), snap_dim(height)

        env = dict(os.environ)
        env.update(
            PYTHONPATH=REPO, device="cuda:0", CUDA_VISIBLE_DEVICES=str(gpu),
            CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        args = [PY, m["script"],
                "--pretrained_pipeline_name_or_path", m["path"],
                "--instruction", prompt,
                "--num_inference_steps", str(int(steps)),
                "--height", str(int(height)), "--width", str(int(width)),
                "--text_guidance_scale", str(float(cfg)),
                "--seed", str(int(seed)),
                "--output_image_path", out_path,
                "--device", "cuda:0"]
        if use_offload:
            args += ["--enable_model_cpu_offload_flag", "True"]
        if m["edit"]:
            args += ["--input_image_paths", input_path, "--image_guidance_scale", str(float(image_cfg))]

        t0 = time.time()
        progress(0.05, desc=f"Loading {model_label} on GPU {gpu} …")
        proc = subprocess.run(args, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
        dt = time.time() - t0

        tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
        log = "\n".join(tail[-25:])
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise gr.Error(f"Generation failed on GPU {gpu} (exit {proc.returncode}).\n\n{log[-1500:]}")
        mode = "offload" if use_offload else "full-VRAM (no offload)"
        return out_path, f"✅ done in {dt:.0f}s on GPU {gpu} [{mode}]  ·  {width}x{height}  ·  {out_path}\n\n{log}"
    finally:
        release_gpu(gpu)

def on_model_change(model_label):
    m = MODELS[model_label]
    hint = ("Edit mode: upload an input image below."
            if m["edit"] else "Text-to-image: no input image needed.")
    return gr.update(value=m["steps"]), gr.update(value=m["cfg"]), gr.update(value=hint)

with gr.Blocks(title="Boogu-Image 0.1") as demo:
    gr.Markdown("# 🖼️ Boogu-Image-0.1  ·  Base / Turbo / Edit\nflash-attn + multi-GPU. Turbo ≈ 1 min, Base ≈ 1.5 min, Edit ≈ 2–3 min.")
    with gr.Row():
        with gr.Column(scale=1):
            model = gr.Dropdown(list(MODELS), value="Turbo (fast, 4 steps)", label="Model",
                                interactive=True)
            mode_hint = gr.Markdown("Text-to-image: no input image needed.")
            prompt = gr.Textbox(label="Prompt / instruction", lines=3,
                                placeholder="e.g. a fox astronaut on the moon, cinematic")
            image = gr.Image(label="Input image (used only by the Edit model)", type="filepath")
            with gr.Row():
                steps = gr.Slider(1, 60, value=4, step=1, label="Steps")
                cfg = gr.Slider(1.0, 8.0, value=1.0, step=0.5, label="Text CFG")
            image_cfg = gr.Slider(1.0, 5.0, value=2.5, step=0.5,
                label="Image CFG (Edit) — higher = tighter adherence to input (1.0 = off)")
            match_ratio = gr.Checkbox(value=True,
                label="Match output to input image ratio (Edit) — snaps to a valid ×16 size")
            with gr.Row():
                width = gr.Number(value=1024, precision=0, label="Width (×16)")
                height = gr.Number(value=1024, precision=0, label="Height (×16)")
            with gr.Row():
                seed = gr.Number(value=42, precision=0, label="Seed")
                offload = gr.Checkbox(value=True, label="CPU offload (auto: off when GPU has ≥40 GB free)")
            _gpu_choices = ["auto"] + list_gpus()
            _default_gpu = next((g for g in _gpu_choices if g.startswith(PINNED_GPU + ":")), "auto")
            gpu_choice = gr.Dropdown(_gpu_choices, value=_default_gpu, label="GPU (pinned; auto-offload by free VRAM)")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            out_img = gr.Image(label="Result", type="filepath")
            out_log = gr.Textbox(label="Log", lines=12)

    def on_image(image, match_ratio, model_label):
        # Uploading an image means the user wants to EDIT it -> force the Edit model,
        # otherwise Turbo/Base would silently ignore the image (text-to-image).
        if not image:
            return (gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update())
        switch = not MODELS[model_label]["edit"]
        new_model = EDIT_LABEL if switch else model_label
        m = MODELS[new_model]
        wU = hU = gr.update()
        if match_ratio:
            ww, hh = fit_to_ratio(image)
            wU, hU = gr.update(value=ww), gr.update(value=hh)
        return (gr.update(value=new_model) if switch else gr.update(),
                gr.update(value=m["steps"]) if switch else gr.update(),
                gr.update(value=m["cfg"]) if switch else gr.update(),
                gr.update(value="🖼️ Edit mode: editing the uploaded image."),
                wU, hU)

    model.change(on_model_change, model, [steps, cfg, mode_hint])
    image.change(on_image, [image, match_ratio, model],
                 [model, steps, cfg, mode_hint, width, height])
    match_ratio.change(on_image, [image, match_ratio, model],
                       [model, steps, cfg, mode_hint, width, height])
    go.click(generate,
             [model, prompt, image, steps, cfg, image_cfg, width, height, seed, offload, gpu_choice, match_ratio],
             [out_img, out_log], concurrency_limit=MAX_PARALLEL)

if __name__ == "__main__":
    if not os.path.exists(os.path.join(REPO, "inference.py")):
        sys.exit("ERROR: app.py must sit in the root of a boogu-project/Boogu-Image checkout "
                 "(inference.py not found next to it). Run scripts/install.sh, or copy app.py "
                 "into your Boogu-Image clone. See README.")
    port = free_port()
    print(f"\n*** Boogu UI on http://0.0.0.0:{port}  (LAN: http://{socket.gethostbyname(socket.gethostname())}:{port}) ***\n", flush=True)
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=port, show_error=True)
