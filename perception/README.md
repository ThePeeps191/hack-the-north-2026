# perception

Real-time perception service for the robot car. It reads a video stream, finds the target described by the active `TaskSpec`, and publishes the target's position on every frame. New specs and new models are hot-swapped without restarting.

## What it does

1. Pulls the latest frame from `VIDEO_SOURCE` (car camera, phone, webcam, or a looping file).
2. Detects objects: YOLOE for open-vocabulary text prompts, or a fixed-vocab model (YOLO11, RF-DETR, TensorRT builds) from the registry.
3. Tracks with ByteTrack, filters by attribute (CLIP) and containment ("shoe inside person"), and selects one target.
4. Publishes a `TargetState` (target offset, size, and pipeline `phase`) for the controller, plus an annotated MJPEG for the UI.
5. Accepts new specs on `POST /spec`, prepares them in the background, and swaps them in between frames. Progress goes out as status events.

The controller drives only when `phase == "TRACKING"`.

## Endpoints (port 8001)

| Method | Path | Purpose |
|---|---|---|
| POST | `/spec` | Apply a new TaskSpec `{instruction_id, spec}` |
| DELETE | `/spec` | Stop tracking (IDLE) |
| GET | `/models` | Model registry and vocabularies |
| GET | `/health` | Phase, fps, active model and spec |
| WS | `/ws/target` | TargetState per frame |
| WS | `/ws/events` | Status events |
| GET | `/video` | Annotated MJPEG |

## Run

```bash
pip install -r requirements.txt

# GPU host (RTX 4070)
python scripts/build_engines.py          # once; builds TensorRT engines for fixed-vocab models
VIDEO_SOURCE=http://<car-ip>/stream uvicorn perception.main:app --host 0.0.0.0 --port 8001

# Mac (logic testing only)
PYTORCH_ENABLE_MPS_FALLBACK=1 IMGSZ=320 VIDEO_SOURCE=clips/pencil.mp4 \
  uvicorn perception.main:app --port 8001
```

Try a spec:

```bash
curl -X POST localhost:8001/spec -H 'content-type: application/json' -d '{
  "instruction_id": "test1",
  "spec": {"spec_id": "s1", "model": "yoloe", "mode": "center",
           "targets": [{"ref": "t1", "detect": ["pencil"], "select": "largest"}]}}'
```

## Config

`VIDEO_SOURCE`, `DEVICE` (auto/cuda/mps/cpu), `IMGSZ`, `PORT`, `MODELS_CONFIG`, `ACQUIRE_TIMEOUT_S`, `CONF_THRESHOLD`. Models are listed in `models.yaml`. TensorRT `.engine` files are built on the GPU host and never committed.
