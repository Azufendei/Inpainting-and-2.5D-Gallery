#!/usr/bin/env python3
"""
gallery_2_5d.py

- Walks a folder of images (and metadata .json saved by your scraper)
- Runs GLPN depth estimation
- Saves depth, visualization PNG, 3D mesh
- Launches a Tkinter gallery showing images, depth, mesh, and parallax
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageTk
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import open3d as o3d
from Scraper import process_site

# ---------- CONFIG ----------
GLPN_MODEL = "vinvino02/glpn-nyu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- UTILS ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_metadata_for_image(img_path: Path):
    j = img_path.with_suffix(img_path.suffix + ".json")
    if not j.exists():
        j2 = img_path.with_suffix(".json")
        if j2.exists():
            j = j2
    if j.exists():
        try:
            return json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

# ---------- DEPTH & MESH ----------
class DepthMeshBuilder:
    def __init__(self, model_name=GLPN_MODEL, device=device, max_width=512):
        print(f"[DEPTH] Loading model {model_name} -> {device}")
        self.processor = GLPNImageProcessor.from_pretrained(model_name)
        self.model = GLPNForDepthEstimation.from_pretrained(model_name).to(device)
        self.device = device
        self.max_width = int(max_width)

    def preprocess(self, img: Image.Image):
        w, h = img.size
        if w > self.max_width:
            new_w = self.max_width
            new_h = int(h * (new_w / w))
        else:
            new_w, new_h = w, h
        new_w -= new_w % 32
        new_h -= new_h % 32
        new_w = max(32, new_w)
        new_h = max(32, new_h)
        return img.resize((new_w, new_h), Image.LANCZOS)

    def predict_depth(self, img: Image.Image):
        img_small = self.preprocess(img)
        inputs = self.processor(images=img_small, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth.squeeze().cpu().numpy()
        return img_small, depth

    def save_depth_and_viz(self, stem: str, depth: np.ndarray, depth_dir: Path):
        ensure_dir(depth_dir)
        d = depth.copy()
        dmin, dmax = float(np.nanmin(d)), float(np.nanmax(d))
        vis = ((d - dmin) / (dmax - dmin) * 255).astype(np.uint8) if dmax - dmin > 1e-6 else (np.zeros_like(d)*255).astype(np.uint8)
        Image.fromarray(vis).save(depth_dir / f"{stem}_depth_vis.png")
        np.save(depth_dir / f"{stem}_depth.npy", d)
        return d

    def build_mesh(self, img: Image.Image, depth: np.ndarray, out_ply: Path, mesh_scale=0.8, downsample=2):
        ensure_dir(out_ply.parent)

        # Downsample depth and image if needed
        if downsample > 1:
            depth_small = depth[::downsample, ::downsample]
            img_small = img.resize((depth_small.shape[1], depth_small.shape[0]), Image.LANCZOS)
        else:
            depth_small, img_small = depth, img

        H, W = depth_small.shape

        # Normalize depth to [-0.5, 0.5] * mesh_scale
        z = ((depth_small - depth_small.min()) / (depth_small.max() - depth_small.min() + 1e-8) - 0.5) * mesh_scale

        xs, ys = np.linspace(0, 1, W), np.linspace(0, 1, H)
        xv, yv = np.meshgrid(xs, ys)

        # Original image coordinates without flipping Y
        verts = np.stack([xv.flatten(), yv.flatten(), z.flatten()], axis=1)

        # Build faces
        faces = []
        for r in range(H - 1):
            for c in range(W - 1):
                i = r * W + c
                faces.append([i, i + 1, i + W])
                faces.append([i + 1, i + W + 1, i + W])
        faces = np.asarray(faces, dtype=np.int32)

        # Vertex colors
        colors = np.array(img_small).astype(np.float32) / 255.0
        colors = colors.reshape(-1, 3)

        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        mesh.compute_vertex_normals()

        # Flip Y-axis after mesh is created for correct orientation
        mesh.transform([[1,0,0,0],
                        [0,-1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])

        o3d.io.write_triangle_mesh(str(out_ply), mesh)
        return out_ply


# ---------- GALLERY UI ----------
class GalleryUI:
    def __init__(self, root, items):
        self.root, self.items, self.idx = root, items, 0
        self.root.title("2.5D Gallery")
        self.build_ui()
        self.show_item(0)

    def build_ui(self):
        left = tk.Frame(self.root); left.pack(side=tk.LEFT, fill=tk.BOTH)
        self.thumb_label = tk.Label(left); self.thumb_label.pack()
        self.meta_text = ScrolledText(left, width=40, height=20); self.meta_text.pack(fill=tk.BOTH)
        btns = tk.Frame(left); btns.pack(fill=tk.X)
        for txt, cmd in [("Prev", self.prev), ("Next", self.next), ("Show Depth", self.show_depth), ("Open 3D", self.open_3d), ("Parallax", self.parallax)]:
            tk.Button(btns, text=txt, command=cmd).pack(side=tk.LEFT)
        right = tk.Frame(self.root); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(right, width=800, height=600, bg="black"); self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas_img = None

    def show_item(self, i):
        # 1. Handle Empty Folder Case
        if not self.items:
            self.idx = 0
            self.canvas.delete("all")
            self.canvas.create_text(400, 300, text="No images found in folder.", fill="white", font=("Arial", 16))
            self.thumb_label.config(image='', text="No Thumbnail")
            self.meta_text.delete("1.0", tk.END)
            self.meta_text.insert(tk.END, "Please check your image directory.")
            return

        # 2. Handle Index Bounds (Safe Clamping)
        self.idx = max(0, min(i, len(self.items) - 1))
        
        # 3. Load Item
        it = self.items[self.idx]
        
        try:
            img = Image.open(it["img_path"]).convert("RGB")
            
            # Update Thumbnail
            self.thumb = ImageTk.PhotoImage(ImageOps.fit(img, (320, 240), Image.LANCZOS))
            self.thumb_label.config(image=self.thumb)
            
            # Update Metadata
            self.meta_text.delete("1.0", tk.END)
            meta_content = json.dumps(it.get("meta"), indent=2) if it.get("meta") else "No metadata available."
            self.meta_text.insert(tk.END, f"File: {it['stem']}\n{meta_content}")
            
            # Update Main Canvas
            w, h = img.size
            scale = min(800/w, 600/h, 1.0) # Adjusted for 800x600 canvas
            disp = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            self.display_img = ImageTk.PhotoImage(disp)
            
            self.canvas.delete("all")
            # Center the image on the canvas
            self.canvas_img = self.canvas.create_image(400, 300, anchor="center", image=self.display_img)
            
        except Exception as e:
            self.meta_text.insert(tk.END, f"\n\nError loading image: {e}")

    def prev(self): self.show_item(self.idx-1)
    def next(self): self.show_item(self.idx+1)

    def show_depth(self):
        self.stop_parallax() 
        dp = Path(self.items[self.idx]["depth_vis"])
        if dp.exists():
            img = Image.open(dp).convert("L")
            w,h = img.size; scale=min(1000/w,800/h,1.0)
            disp=img.resize((int(w*scale),int(h*scale)),Image.LANCZOS)
            self.display_img = ImageTk.PhotoImage(disp.convert("RGB"))
            self.canvas.delete("all")
            self.canvas_img = self.canvas.create_image(0,0,anchor="nw",image=self.display_img)

    def open_3d(self):
        ply = Path(self.items[self.idx]["ply"])
        if not ply.exists(): tk.messagebox.showinfo("No mesh",f"Mesh not found: {ply}"); return
        mesh = o3d.io.read_triangle_mesh(str(ply))
        if not mesh.has_vertex_normals(): mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])

    def stop_parallax(self):
        if getattr(self, "parallax_running", False):
            self.parallax_running = False
            if getattr(self, "parallax_id", None):
                self.root.after_cancel(self.parallax_id)
                self.parallax_id = None


    def parallax(self, steps=20, shift_px=60, delay=40):
        """
        Fast real-time parallax animation:
        - No precomputed frames
        - Vectorized row-shift calculation
        - Smooth continuous back-and-forth
        """

        # --- STOP IF RUNNING ---
        if getattr(self, "parallax_running", False):
            self.parallax_running = False
            if getattr(self, "parallax_id", None):
                self.root.after_cancel(self.parallax_id)
            return

        # --- START ---
        self.parallax_running = True

        # load image + depth
        it = self.items[self.idx]
        img = Image.open(it["img_path"]).convert("RGB")
        depth = np.load(it["depth_npy"])

        # resize depth to match image if needed
        if depth.shape != img.size[::-1]:
            d = depth
            d = ((d - d.min())/(d.max()-d.min()+1e-8)*255).astype(np.uint8)
            d = Image.fromarray(d).resize(img.size, Image.LANCZOS)
            depth = np.array(d).astype(np.float32) / 255.0

        arr = np.array(img)
        h, w = depth.shape

        # precompute row-wise shift intensity
        # shape: [h]
        row_shifts = (0.5 - depth).mean(axis=1) * shift_px

        # generate oscillation values
        t_vals = list(np.linspace(-1, 1, steps)) + list(np.linspace(1, -1, steps)[1:-1])
        self._parallax_t_vals = t_vals
        self._parallax_index = 0
        self._parallax_arr = arr
        self._parallax_row_shifts = row_shifts

        def make_frame():
            """Create a single output frame without precomputation."""
            t = self._parallax_t_vals[self._parallax_index]
            src = self._parallax_arr
            out = np.zeros_like(src)

            rs = self._parallax_row_shifts * t
            rs = rs.astype(np.int32)

            for r in range(h):
                s = rs[r]
                if s > 0:
                    out[r, s:] = src[r, :-s]
                elif s < 0:
                    out[r, :w+s] = src[r, -s:]
                else:
                    out[r] = src[r]

            return Image.fromarray(out)

        def animate():
            if not self.parallax_running:
                return

            frame = make_frame()

            # resize to match the existing canvas scaling
            fw, fh = frame.size
            scale = min(1000/fw, 800/fh, 1.0)
            disp = frame.resize((int(fw*scale), int(fh*scale)), Image.LANCZOS)

            self.display_img = ImageTk.PhotoImage(disp)
            self.canvas.delete("all")
            self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=self.display_img)

            self._parallax_index = (self._parallax_index + 1) % len(self._parallax_t_vals)
            self.parallax_id = self.root.after(delay, animate)

        animate()



# ---------- MAIN ----------
def main(args):
    img_dir = Path(args.img_dir)
    assert img_dir.exists(), f"img_dir not found: {img_dir}"

    site_folder = Path(args.site.lower().replace(" ","_"))
    RESULTS_3D = ensure_dir(ROOT / "Results" / site_folder / "3D")
    DEPTH_DIR = ensure_dir(ROOT / "Results" / site_folder / "depth")
    THUMB_DIR = ensure_dir(ROOT / "Results" / site_folder / "thumbs")

    builder = DepthMeshBuilder(max_width=args.max_width)
    items=[]

    for p in sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]]):
        stem = p.stem
        depth_npy = DEPTH_DIR / f"{stem}_depth.npy"
        depth_vis = DEPTH_DIR / f"{stem}_depth_vis.png"
        ply_out = RESULTS_3D / f"{stem}.ply"

        if not depth_npy.exists():
            img = Image.open(p).convert("RGB")
            img_small, depth = builder.predict_depth(img)
            builder.save_depth_and_viz(stem, depth, DEPTH_DIR)
            builder.build_mesh(img_small, depth, ply_out, mesh_scale=args.mesh_scale, downsample=1)
        else:
            depth = np.load(depth_npy)

        meta = load_metadata_for_image(p)
        thumb_path = THUMB_DIR / f"{stem}_thumb.png"
        if not thumb_path.exists():
            thumb = ImageOps.fit(Image.open(p).convert("RGB"), (320,240), Image.LANCZOS)
            thumb.save(thumb_path)

        items.append({
            "img_path": str(p),
            "thumb": str(thumb_path),
            "meta": meta,
            "depth_npy": str(depth_npy),
            "depth_vis": str(depth_vis),
            "ply": str(ply_out),
            "stem": stem
        })

    root = tk.Tk()
    GalleryUI(root, items)
    root.mainloop()
ROOT = Path(__file__).resolve().parents[2]  # adjust if needed

# ---------- ENTRY ----------
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--site", type=str, default="Hagia Sophia")
    parser.add_argument("--run-scraper", action="store_true")
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--mesh-scale", type=float, default=0.6)
    args = parser.parse_args()

    site_folder = args.site.lower().replace(" ","_")
    if args.run_scraper:
        print(f"[SCRAPER] Running scraper for {args.site}")
        process_site(args.site, img_limit=20)

    args.img_dir = ROOT / "src" / "scraping" / "data" / "raw_images" / site_folder
    print(f"[GALLERY] Using img_dir: {args.img_dir}")
    main(args)


# only run gallery
#python gallery.py --site "Hagia Sophia"
# run scraping + gallery
#python gallery.py --site "Hagia Sophia" --run-scraper
