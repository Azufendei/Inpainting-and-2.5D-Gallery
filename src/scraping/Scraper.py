import json
import time
import random
import re
from pathlib import Path
import requests
import wikipediaapi

# -----------------------
# config / constants
# -----------------------
API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "project name (contact: email)"}
wiki = wikipediaapi.Wikipedia(language="en", user_agent=HEADERS["User-Agent"])

BAD_EXTS = {".svg", ".webp", ".pdf", ".gif", ".tif", ".tiff", ".xcf"}
GOOD_EXTS = {".jpg", ".jpeg", ".png"}

# -----------------------
# helpers
# -----------------------
def norm(name: str) -> str:
    return name.strip().replace(" ", "_").lower()

def text_dir(site: str) -> Path:
    d = Path("data/raw_text") / norm(site)
    d.mkdir(parents=True, exist_ok=True)
    return d

def img_dir(site: str) -> Path:
    d = Path("data/raw_images") / norm(site)
    d.mkdir(parents=True, exist_ok=True)
    return d

def polite_get(url, params=None, stream=False):
    for _ in range(5):
        try:
            r = requests.get(url, params=params, headers=HEADERS, stream=stream, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 5))
                print(f"[WAIT] rate-limited — sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            print(f"[RETRY] {e}; sleeping")
            time.sleep(random.uniform(1.0, 2.5))
    raise RuntimeError(f"GET failed for: {url} (params={params})")

def is_photo(url: str) -> bool:
    ext = Path(url).suffix.lower()
    return ext in GOOD_EXTS and ext not in BAD_EXTS

# -----------------------
# wikipedia & architecture extraction
# -----------------------
ARCH_KEYS = [
    "architecture","architectural","structure","structural",
    "layout","plan","floor plan","design","style",
    "construction","built","materials","material","stone",
    "marble","brick","dimensions","measurements","height","width"
]

def fetch_wiki_text(site: str) -> dict:
    page = wiki.page(site)
    if not page.exists():
        print(f"[WARN] Wikipedia page not found: {site}")
        return None
    return {"title": site, "summary": page.summary or "", "text": page.text or ""}

def extract_architecture_sections(full_text: str) -> str:
    extracted = []
    sections = re.split(r"(?m)^==+\s*(.*?)\s*==+\s*$", full_text)
    if not sections:
        return ""
    intro = sections[0].strip()
    if intro and any(k in intro.lower() for k in ARCH_KEYS):
        extracted.append("[SECTION: INTRO]\n" + intro)
    for i in range(1, len(sections), 2):
        if i+1 >= len(sections): break
        header = sections[i].strip()
        body   = sections[i+1].strip()
        if any(k in header.lower() for k in ARCH_KEYS) or any(k in body.lower() for k in ARCH_KEYS):
            extracted.append(f"[SECTION: {header.upper()}]\n{body}")
    return "\n\n".join(extracted)

def save_text_data(site: str, w: dict):
    d = text_dir(site)
    (d / "summary.txt").write_text(w["summary"], encoding="utf-8")
    (d / "full.txt").write_text(w["text"], encoding="utf-8")
    arch = extract_architecture_sections(w["text"])
    (d / "architectural.txt").write_text(arch, encoding="utf-8")
    prompt = (
        f"Reconstruct a photorealistic and structurally accurate 3D model of {site}.\n"
        f"Use the architectural descriptions, materials, layout and structural details below:\n\n{arch}"
    )
    (d / "prompt.txt").write_text(prompt, encoding="utf-8")
    return prompt

# -----------------------
# get commons category via Wikipedia -> Wikidata P373
# -----------------------
def get_commons_category(site: str) -> str | None:
    print("[CAT] Resolving official Commons category via Wikipedia -> Wikidata (P373)...")
    params = {"action":"query","format":"json","prop":"pageprops","titles":site.replace(" ", "_")}
    data = polite_get("https://en.wikipedia.org/w/api.php", params=params).json()
    pages = data.get("query", {}).get("pages", {})
    wikidata_id = None
    for p in pages.values():
        wikidata_id = p.get("pageprops", {}).get("wikibase_item")
        if wikidata_id:
            break
    if not wikidata_id:
        print("[CAT] No wikidata item for page.")
        return None
    wd = polite_get("https://www.wikidata.org/w/api.php", params={
        "action":"wbgetentities","ids":wikidata_id,"format":"json","props":"claims"
    }).json()
    claims = wd.get("entities", {}).get(wikidata_id, {}).get("claims", {})
    if "P373" not in claims:
        print("[CAT] Wikidata item has no P373 Commons category.")
        return None
    category = claims["P373"][0]["mainsnak"]["datavalue"]["value"]
    print(f"[CAT] Resolved Commons category: {category}")
    return category

# -----------------------
# category traversal and file collection
# -----------------------
def category_members(category_title: str, cmtype: str=None, cmnamespace=None, cmlimit=50):
    """
    Wrapper returning list of category members (one page) with continuation handling.
    cmtype can be 'file' or 'subcat' or 'page' or None (all).
    cmnamespace can be set (6 for files, 14 for subcats).
    """
    members = []
    cont = {}
    while True:
        params = {"action":"query","format":"json","list":"categorymembers","cmtitle":category_title,"cmlimit":cmlimit}
        if cmtype:
            params["cmtype"] = cmtype
        if cmnamespace is not None:
            params["cmnamespace"] = cmnamespace
        params.update(cont)
        data = polite_get(API_URL, params=params).json()
        ms = data.get("query", {}).get("categorymembers", [])
        members.extend(ms)
        cont = data.get("continue", {})
        if not cont:
            break
        # map continue token keys (commons uses 'cmcontinue' generally)
        # cont already suitable to pass back into params
    return members

def collect_files_from_category_tree(cat_title: str, target_count: int=50, max_depth: int=2):
    """
    BFS over category -> subcategories to collect file members.
    Returns unique list of file titles (namespace 6).
    """
    files = []
    seen_cats = set()
    queue = [(cat_title, 0)]
    while queue and len(files) < target_count:
        cat, depth = queue.pop(0)
        if cat in seen_cats: 
            continue
        seen_cats.add(cat)
        # get files directly in category
        file_members = category_members(cat, cmtype="file", cmnamespace=6, cmlimit=100)
        for m in file_members:
            if m not in files:
                files.append(m)
                if len(files) >= target_count:
                    break
        if len(files) >= target_count:
            break
        # if we can go deeper, enqueue subcategories
        if depth < max_depth:
            subcats = category_members(cat, cmtype="subcat", cmnamespace=14, cmlimit=100)
            for sc in subcats:
                queue.append((sc["title"], depth+1))
    return files[:target_count]

# -----------------------
# image downloader
# -----------------------
def download_image_from_title(member, out_path: Path, idx:int):
    title = member["title"]
    # fetch imageinfo
    meta = polite_get(API_URL, params={
        "action":"query","format":"json","prop":"imageinfo","iiprop":"url|extmetadata|metadata|size","titles":title
    }).json()
    pages = meta.get("query", {}).get("pages", {})
    for p in pages.values():
        if "imageinfo" not in p:
            continue
        info = p["imageinfo"][0]
        url = info.get("url")
        if not url or not is_photo(url):
            return False
        ext = Path(url).suffix.lower()
        img_path = out_path / f"img_{idx:04}{ext}"
        meta_path = out_path / f"img_{idx:04}.json"
        if not img_path.exists():
            print(f"[DL] {img_path.name} -> {url}")
            r = polite_get(url, stream=True)
            with open(img_path, "wb") as fh:
                for chunk in r.iter_content(8192):
                    fh.write(chunk)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        return True
    return False

# -----------------------
# main scrape function (final)
# -----------------------
def scrape_images_for_site(site:str, img_limit:int=50, max_cat_depth:int=2):
    out = img_dir(site)
    category = get_commons_category(site)
    if not category:
        print("[IMG] failed to resolve commons category; aborting.")
        return
    cat_title = f"Category:{category.replace(' ', '_')}"
    print(f"[IMG] Gathering up to {img_limit} images from {cat_title} (depth={max_cat_depth})")
    members = collect_files_from_category_tree(cat_title, target_count=img_limit, max_depth=max_cat_depth)
    if not members:
        print("[IMG] No image members found in category tree.")
        return
    print(f"[IMG] Collected {len(members)} file members; starting download...")
    idx = 1
    for m in members:
        ok = download_image_from_title(m, out, idx)
        if ok:
            idx += 1
        else:
            print(f"[SKIP] {m.get('title')} (not a downloadable photo or failed)")
        time.sleep(random.uniform(0.5, 1.5))
    print(f"[IMG] Done — downloaded {idx-1} images to {out}")

# -----------------------
# high-level pipeline
# -----------------------
def process_site(site:str, img_limit:int=50):
    print(f"\n===== Processing: {site} =====")
    w = fetch_wiki_text(site)
    if w:
        save_text = save_text_data(site, w)  # saves summary, full, architectural, prompt
        print("[OK] Text & prompt saved.")
    scrape_images_for_site(site, img_limit)

# -----------------------
# run
# -----------------------
if __name__ == "__main__":
    process_site("Hagia Sophia", img_limit=20) # Alter according to the subject and total number if related images required
