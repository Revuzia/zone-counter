"""Shelf/Street Counter: RF-DETR detection + supervision zone analytics. No API key needed."""
import base64, io
import numpy as np, supervision as sv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

app = FastAPI()
model = RFDETRNano()  # downloads open weights on first run

def analyze(img: Image.Image, threshold=0.5, zone=None):
    img = img.convert("RGB")
    dets = model.predict(img, threshold=threshold)
    w, h = img.size
    # zone as normalized polygon [[x,y],...]; default = right half
    zone = zone or [[0.5,0],[1,0],[1,1],[0.5,1]]
    poly = (np.array(zone) * [w, h]).astype(int)
    pz = sv.PolygonZone(polygon=poly, triggering_anchors=(sv.Position.BOTTOM_CENTER,))
    in_zone = pz.trigger(dets) if len(dets) else np.array([], bool)
    names = [COCO_CLASSES[int(c)] for c in dets.class_id]
    labels = [f"{n} {c:.2f}" for n, c in zip(names, dets.confidence)]
    frame = np.array(img)
    frame = sv.BoxAnnotator().annotate(frame, dets)
    frame = sv.LabelAnnotator(text_scale=0.5).annotate(frame, dets, labels)
    frame = sv.PolygonZoneAnnotator(zone=pz, color=sv.Color.RED, thickness=3).annotate(frame)
    counts, zcounts = {}, {}
    for n, z in zip(names, in_zone):
        counts[n] = counts.get(n, 0) + 1
        if z: zcounts[n] = zcounts.get(n, 0) + 1
    return Image.fromarray(frame), {"total": len(dets), "counts": counts, "in_zone": zcounts}

@app.post("/api/detect")
async def detect(file: UploadFile = File(...), threshold: float = Form(0.5)):
    out, stats = analyze(Image.open(io.BytesIO(await file.read())), threshold)
    buf = io.BytesIO(); out.save(buf, "JPEG")
    return {"image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), **stats}

@app.get("/")
def index(): return FileResponse("static/index.html")

if __name__ == "__main__":
    import sys
    out, stats = analyze(Image.open(sys.argv[1])); out.save(sys.argv[2]); print(stats)
