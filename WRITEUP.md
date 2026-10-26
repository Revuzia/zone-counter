# What I built with Roboflow's open-source tools

After applying for the Full Stack Engineer role, I wanted to learn Roboflow's stack by building something with it instead of only reading about it. The result is **Zone Counter**, a small full-stack web app. You upload an image, and it shows the detections, the count for each class, and how many objects fall inside a defined region.

Here is how it works. A FastAPI backend loads **RF-DETR Nano** from the `rfdetr` package. That gave me a real-time-class detector with open weights and no API key. **supervision** does everything after the model runs: it converts the predictions to `sv.Detections`, annotates them with `BoxAnnotator` and `LabelAnnotator`, and counts objects with `PolygonZone`, triggered on each box's bottom-center. The frontend is a lightweight React page that posts the image and renders the annotated result and the stats table.

What stood out to me was how little glue code I needed. Because supervision's `Detections` abstraction is model-agnostic, I could swap the detector without touching the analytics layer. Zone counting took only a few lines, and that's the part a real product (retail shelves, parking, safety zones) actually cares about.

I'm honest about the scope: this is a weekend-sized demo running on CPU with COCO weights. My next steps would be letting users draw the zone in the browser, adding video with ByteTrack and line-crossing counts, and trying a fine-tuned RF-DETR model from Roboflow Universe.

I've been building production software since 2024, mostly React/TypeScript and Node/Python. This project showed me why developer experience is such a big part of Roboflow's appeal.

— Steven Itskovich · github.com/Revuzia · linkedin.com/in/steven-itskovich
