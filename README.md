# Zone Counter — RF-DETR + supervision demo

Upload an image in a small React page. A FastAPI backend runs **RF-DETR Nano** (Roboflow's open-source, Apache-2.0 detector; COCO weights download automatically, **no API key**). Then **supervision** draws the boxes and labels, and a `PolygonZone` counts the objects whose bottom-center falls inside a zone (the right half of the image by default). The page shows the annotated image, the count per class, and the count inside the zone.

## Run
```bash
cd /workspace/roboflow_demo
python3 -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --port 8765        # open http://localhost:8765
# CLI: python app.py samples/street.jpg samples/street_annotated.jpg
```
It runs on CPU, at about 1–2 seconds per image after the model loads.

## Files
- `app.py`: the model, the supervision annotators and zone counting, the `/api/detect` endpoint, and the CLI
- `static/index.html`: the React UI (loaded from a CDN, no build step)
- `samples/`: the test image and the annotated output

## Next ideas
- Make the zone editable by drawing it on the image.
- Add video upload with ByteTrack plus line-crossing counts.
- Swap in a fine-tuned RF-DETR model.
