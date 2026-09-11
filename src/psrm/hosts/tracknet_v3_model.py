from pathlib import Path
import sys,importlib,math
import cv2,numpy as np,torch
from PIL import Image
WIDTH,HEIGHT=512,288

def native_decode(raw: np.ndarray) -> tuple[bool, float, float]:
    mask = (raw > 0.5).astype(np.uint8) * 255
    if not mask.any():
        return False, math.nan, math.nan
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(contour) for contour in contours]
    x, y, w, h = max(rects, key=lambda rect: rect[2] * rect[3])
    return True, x + w / 2.0, y + h / 2.0

def read_video(path: Path, rgb: bool) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[:, :, ::-1] if rgb else frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No readable frames: {path}")
    return np.asarray(frames)

def prepare_official8(frames: np.ndarray, frame_ids: np.ndarray, median: np.ndarray) -> np.ndarray:
    median = np.asarray(Image.fromarray(median).resize((WIDTH, HEIGHT)))
    median = np.moveaxis(median, -1, 0)
    batches = []
    for frame_id in frame_ids:
        window = frames[frame_id - 7:frame_id + 1]
        resized = [np.moveaxis(np.asarray(Image.fromarray(f).resize((WIDTH, HEIGHT))), -1, 0)
                   for f in window]
        batches.append(np.concatenate([median, *resized], axis=0) / 255.0)
    return np.asarray(batches, dtype=np.float32)

def load_official8(tracknet_root: Path, checkpoint_path: Path, device: str):
    root = str(tracknet_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    general = importlib.import_module("utils.general")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = general.get_model("TrackNet", 8, "concat").to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model

def infer(model, inputs: np.ndarray, mode: str, device: str, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start:start + batch_size]).float().to(device)
            if mode == "official8":
                pred = model(batch)[:, 7]
            else:
                pred = model.forward(frames=batch)[0][:, 0]
            outputs.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)
