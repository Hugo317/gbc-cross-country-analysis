---
title: Country conditions and company performance
emoji: 🩸
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
---

Dashboard for the GBC cross-country analysis. Self-contained: reads only `data/` (built by `build_extract.py`).

Local: `uv run --group dash python app/dash_app.py`
Deploy: create a Hugging Face Space (SDK: Docker) and push the contents of this `app/` folder.
