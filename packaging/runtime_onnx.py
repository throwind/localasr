from __future__ import annotations

import os


os.environ.setdefault("LOCALASR_ENGINE", "sherpa-onnx")
os.environ.setdefault("LOCALASR_MAX_DIARIZATION_SECONDS", "30")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
