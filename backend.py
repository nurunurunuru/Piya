import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import base64

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained YOLO model
candidate_paths = [
    os.path.join("runs", "detect", "dhaka_buildings-3", "weights", "best.pt"),
    os.path.join("runs", "detect", "dhaka_buildings-2", "weights", "best.pt"),
    os.path.join("runs", "detect", "dhaka_buildings",   "weights", "best.pt"),
    os.path.join("runs", "segment", "dhaka_buildings-3", "weights", "best.pt"),
    "yolov8n.pt",
    "yolov8n-seg.pt",
]

model = None
model_path = None
for path in candidate_paths:
    if os.path.exists(path):
        try:
            model = YOLO(path)
            model_path = path
            print(f"Model loaded successfully from {path}")
            break
        except Exception as e:
            print(f"Error loading model from {path}: {e}")

if not model:
    print("Warning: Could not load any trained model weights!")

# ─────────────────────────────────────────────
# Class definitions  (must match data.yaml)
#   0 = low     → Green
#   1 = medium  → Yellow/Amber
#   2 = high    → Red
# ─────────────────────────────────────────────
CLASS_INFO = {
    0: {"name": "Low-rise",    "short": "LOW",  "bgr": (30,  200,  30),  "hex": "#22c55e"},
    1: {"name": "Medium-rise", "short": "MED",  "bgr": (20,  200, 220),  "hex": "#eab308"},
    2: {"name": "High-rise",   "short": "HIGH", "bgr": (30,   30, 230),  "hex": "#ef4444"},
}


def img_to_base64(img_array: np.ndarray) -> str:
    """Convert OpenCV BGR image to a base64 data-URI."""
    _, buffer = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def draw_labeled_detections(base_img: np.ndarray, boxes_np, masks_np) -> np.ndarray:
    """
    Draw every building with:
      • colour-coded mask overlay  (if segmentation model)
      • colour-coded bounding box
      • pill-shaped label  (e.g.  HIGH 87%)
    """
    out = base_img.copy()
    h, w = out.shape[:2]

    # ── Mask overlays (drawn first, behind boxes) ─────────────────────────
    if masks_np is not None and len(masks_np):
        for mask_raw, box in zip(masks_np, boxes_np):
            cls_id  = int(box[5])
            color   = CLASS_INFO.get(cls_id, CLASS_INFO[0])["bgr"]
            binary  = cv2.resize(mask_raw, (w, h)) > 0.5

            overlay = out.copy()
            overlay[binary] = color
            cv2.addWeighted(overlay, 0.40, out, 0.60, 0, out)

            # Mask contour
            contour_src = (binary.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(contour_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 2)

    # ── Bounding boxes + labels ───────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box in boxes_np:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf   = float(box[4])
        cls_id = int(box[5])
        info   = CLASS_INFO.get(cls_id, CLASS_INFO[0])
        color  = info["bgr"]
        label  = f"{info['short']}  {conf:.0%}"

        # Box outline
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label pill background
        fscale = max(0.45, min(0.65, (x2 - x1) / 250))
        thick  = 1
        (tw, th), _ = cv2.getTextSize(label, font, fscale, thick)
        pad   = 5
        ly1   = max(0, y1 - th - pad * 2)
        ly2   = y1
        lx2   = x1 + tw + pad * 2
        cv2.rectangle(out, (x1, ly1), (lx2, ly2), color, -1)
        cv2.putText(out, label, (x1 + pad, ly2 - pad),
                    font, fscale, (255, 255, 255), thick, cv2.LINE_AA)

    return out


def draw_class_only(base_img: np.ndarray, boxes_np, masks_np, target_cls: int) -> np.ndarray:
    """Return a detection image showing ONLY buildings of target_cls."""
    filtered_boxes = [b for b in boxes_np if int(b[5]) == target_cls]
    filtered_masks = None
    if masks_np is not None and len(masks_np):
        filtered_masks = np.array(
            [m for b, m in zip(boxes_np, masks_np) if int(b[5]) == target_cls]
        ) if filtered_boxes else None

    # If no buildings of this class exist, just return a dim copy with a text overlay
    if not filtered_boxes:
        grey = cv2.addWeighted(base_img, 0.35, np.zeros_like(base_img), 0.65, 0)
        info = CLASS_INFO[target_cls]
        cv2.putText(grey, f"No {info['name']} buildings detected",
                    (30, grey.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (200, 200, 200), 2, cv2.LINE_AA)
        return grey

    return draw_labeled_detections(base_img, filtered_boxes, filtered_masks)


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not model:
        return {"error": "Model not loaded"}

    # ── Read image ────────────────────────────────────────────────────────
    contents     = await file.read()
    nparr        = np.frombuffer(contents, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if original_img is None:
        return {"error": "Could not decode image"}

    # ── Run YOLO inference ────────────────────────────────────────────────
    results  = model(original_img)
    result   = results[0]

    boxes_np = result.boxes.data.cpu().numpy() if result.boxes is not None else []
    masks_np = None
    if result.masks is not None:
        raw = result.masks.data.cpu().numpy()
        h, w = original_img.shape[:2]
        masks_np = np.array([cv2.resize(m, (w, h)) for m in raw])

    # ── 1. Full detection image (all classes, labelled) ───────────────────
    detected_img = draw_labeled_detections(original_img, boxes_np, masks_np)

    # ── 2. Classification colour-map (mask only, no box outline) ─────────
    classified_img = original_img.copy()
    if masks_np is not None and len(masks_np):
        for mask_raw, box in zip(masks_np, boxes_np):
            cls_id = int(box[5])
            color  = CLASS_INFO.get(cls_id, CLASS_INFO[0])["bgr"]
            binary = mask_raw > 0.5
            overlay = classified_img.copy()
            overlay[binary] = color
            cv2.addWeighted(overlay, 0.50, classified_img, 0.50, 0, classified_img)

    # ── 3. Per-class filtered views ───────────────────────────────────────
    high_img   = draw_class_only(original_img, boxes_np, masks_np, target_cls=2)
    medium_img = draw_class_only(original_img, boxes_np, masks_np, target_cls=1)
    low_img    = draw_class_only(original_img, boxes_np, masks_np, target_cls=0)

    # ── 4. Counts ─────────────────────────────────────────────────────────
    counts = {0: 0, 1: 0, 2: 0}
    for box in boxes_np:
        cls_id = int(box[5])
        if cls_id in counts:
            counts[cls_id] += 1

    return {
        "original":    img_to_base64(original_img),
        "detected":    img_to_base64(detected_img),
        "classified":  img_to_base64(classified_img),
        "high_only":   img_to_base64(high_img),
        "medium_only": img_to_base64(medium_img),
        "low_only":    img_to_base64(low_img),
        "counts": {
            "low":   counts[0],
            "mid":   counts[1],
            "high":  counts[2],
            "total": sum(counts.values()),
        },
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "model": model_path or "none"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
