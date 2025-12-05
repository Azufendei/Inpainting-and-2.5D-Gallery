# Heritage-Reconstruction project

### src/inpainting/infer_sd.py Runs stability-ai's diffusion inpainting 1 model to reconstruct the user input image according to masked area and prompt.
### src/reconstruction/DepthCapture.py captures pixel depth using GLPN and reconstructs a 3D surface for the object ideal for small angle shift 2.5d viewing.
### (Mask for now needs to be generated manually by the user, tkinter ui for the same is implemented to provide ease for the same).
### Scraper.py scrapes wikimedia for manually input subject and gets photos, gallery.py creates a 2.5d viewing gallery using monocular 3d reconstruction with and the scraped images and displayes using a tkinter ui.
