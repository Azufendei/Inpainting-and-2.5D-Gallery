import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data" / "inpaint_inputs"
MASK_DIR  = ROOT / "data" / "masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)


class MaskingApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Mask Creator")

        # Load image
        self.image = Image.open(image_path).convert("RGB")
        self.image_stem = image_path.stem
        self.w, self.h = self.image.size

        # Mask (white = mask)
        self.mask = Image.new("L", (self.w, self.h), 0)
        self.mask_draw = ImageDraw.Draw(self.mask)

        # Undo stack
        self.undo_stack = []

        # Drawing modes
        self.mode = "brush"   # brush | eraser | polygon
        self.brush_size = 20

        # Polygon points
        self.poly_points = []

        # Panning
        self.pan_start = None

        # Build UI
        self.build_toolbar()
        self.build_scrollable_canvas()

        # Initial display
        self.update_display()

    # -------------------- Toolbar --------------------
    def build_toolbar(self):
        toolbar = tk.Frame(self.root, bg="#ddd")
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="Brush", command=lambda: self.set_mode("brush")).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Eraser", command=lambda: self.set_mode("eraser")).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Polygon", command=lambda: self.set_mode("polygon")).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Finish Polygon", command=self.finish_polygon).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Clear", command=self.clear_mask).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Save Mask", command=self.save_mask).pack(side=tk.LEFT)

        tk.Label(toolbar, text="Brush size").pack(side=tk.LEFT, padx=5)
        self.brush_slider = tk.Scale(toolbar, from_=5, to=80, orient=tk.HORIZONTAL, command=self.change_brush_size)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side=tk.LEFT)

    # -------------------- Scrollable Canvas --------------------
    def build_scrollable_canvas(self):
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        self.v_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas
        self.canvas = tk.Canvas(frame, width=min(1200, self.w), height=min(800, self.h),
                                scrollregion=(0, 0, self.w, self.h),
                                yscrollcommand=self.v_scroll.set,
                                xscrollcommand=self.h_scroll.set)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)

        # Image handle
        self.tk_image = None
        self.canvas_image = self.canvas.create_image(0, 0, anchor="nw")

    # -------------------- Modes --------------------
    def set_mode(self, mode):
        self.mode = mode
        self.poly_points = []
        print(f"Mode → {mode}")
        self.update_display()

    def change_brush_size(self, v):
        self.brush_size = int(v)

    # -------------------- Events --------------------
    def canvas_coords(self, event):
        """Screen → canvas coords"""
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        return x, y

    def on_click(self, event):
        x, y = self.canvas_coords(event)
        if self.mode in ("brush", "eraser"):
            self.save_state()
            self.draw_brush(x, y)
        elif self.mode == "polygon":
            self.poly_points.append((x, y))
            self.update_display()

    def on_drag(self, event):
        x, y = self.canvas_coords(event)
        if self.mode in ("brush", "eraser"):
            self.draw_brush(x, y)

    def start_pan(self, event):
        self.pan_start = (event.x, event.y)

    def do_pan(self, event):
        if self.pan_start:
            dx = self.pan_start[0] - event.x
            dy = self.pan_start[1] - event.y
            self.canvas.xview_scroll(int(dx / 2), "units")
            self.canvas.yview_scroll(int(dy / 2), "units")
            self.pan_start = (event.x, event.y)

    # -------------------- Drawing --------------------
    def draw_brush(self, x, y):
        r = self.brush_size // 2
        fill_value = 255 if self.mode == "brush" else 0
        self.mask_draw.ellipse((x - r, y - r, x + r, y + r), fill=fill_value)
        self.update_display()

    def finish_polygon(self):
        if len(self.poly_points) >= 3:
            self.save_state()
            self.mask_draw.polygon(self.poly_points, fill=255)
        self.poly_points = []
        self.update_display()

    # -------------------- Undo --------------------
    def save_state(self):
        self.undo_stack.append(self.mask.copy())

    def undo(self):
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.mask_draw = ImageDraw.Draw(self.mask)
            self.update_display()

    # -------------------- Display --------------------
    def update_display(self):
        display = self.image.convert("RGBA")

        # Polygon preview
        if self.mode == "polygon" and self.poly_points:
            overlay = Image.new("RGBA", display.size, (0, 0, 0, 0))
            dr = ImageDraw.Draw(overlay)
            if len(self.poly_points) > 1:
                dr.line(self.poly_points, fill=(0, 255, 0, 255), width=2)
            for p in self.poly_points:
                dr.ellipse((p[0]-3, p[1]-3, p[0]+3, p[1]+3), fill=(0, 255, 0, 255))
            display = Image.alpha_composite(display, overlay)

        # Mask overlay (semi-transparent red)
        red = Image.new("RGBA", display.size, (255, 0, 0, 120))
        display = Image.composite(red, display, self.mask)

        self.tk_image = ImageTk.PhotoImage(display)
        self.canvas.itemconfig(self.canvas_image, image=self.tk_image)

    # -------------------- Save / Clear --------------------
    def save_mask(self):
        out = MASK_DIR / f"{self.image_stem}_mask.png"
        self.mask.save(out)
        print(f"Saved → {out}")

    def clear_mask(self):
        self.save_state()
        self.mask = Image.new("L", (self.w, self.h), 0)
        self.mask_draw = ImageDraw.Draw(self.mask)
        self.update_display()


def start_masking(image_path):
    root = tk.Tk()
    MaskingApp(root, image_path)
    root.mainloop()


if __name__ == "__main__":
    start_masking(IMAGE_DIR / "crumblingCherub.jpg")
