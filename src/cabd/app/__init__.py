"""Demo app (local web page): dimensions -> predicted behaviour (forward) and target behaviour -> dimensions (inverse).

  bundle     builds models/app_models_n1536.joblib from the converted 1536-geometry models
  engine     geometry, forward prediction, deformation frames, inverse design
  server     standard-library HTTP server + static/ (index.html, app.js, style.css)
  share_zip  self-contained zip for teammates (build/CABD_demo.zip)

usage: python -m cabd.app.server [--port 8765]   then open http://localhost:8765
"""
