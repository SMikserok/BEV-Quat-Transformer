"""BEV transformation and panorama processing utilities.

This module provides functions for processing panoramic images and
converting them to bird's-eye view (BEV).
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from image_eller_quat import create_inverse_panorama


def quaternion_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to Euler angles.

    Returns yaw, pitch, roll in radians.

    """
    w, x, y, z = q

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return yaw, pitch, roll


def load_json(json_path: str) -> list[dict] | dict:
    """Load JSON file from path."""
    if not Path(json_path).exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with Path(json_path).open(encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def compute_coverage(alt: float, fov_deg: float) -> float:
    """Compute coverage area from altitude and field of view."""
    fov_rad = math.radians(fov_deg)
    return 2 * alt * math.tan(fov_rad / 2)


def resize_panorama(
    image: Image.Image,
    target_height: int,
    target_width: int,
    method: object = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Resize panorama image to target dimensions."""
    return image.resize((target_width, target_height), method)


def grid_sample(image: torch.Tensor, optical: torch.Tensor) -> tuple[torch.Tensor, None]:
    """Apply grid sampling to image using optical flow coordinates."""
    _, _, ih, iw = image.shape
    _, _, _, _ = optical.shape
    ix = optical[..., 0]
    iy = optical[..., 1]
    ix = 2 * ix / (iw - 1) - 1
    iy = 2 * iy / (ih - 1) - 1
    grid = torch.stack((ix, iy), dim=-1)
    return torch.nn.functional.grid_sample(image, grid, align_corners=True)


def bev_transform(
    batch_size: int,
    size: int,
    height: int,
    width: int,
    meter_per_pixel: float,
    camera_height: float,
    pitch_angle: float,
) -> torch.Tensor:
    """Create BEV transformation coordinates.

    Args:
        batch_size: Batch size for processing.
        size: BEV output size.
        height: Panorama height.
        width: Panorama width.
        meter_per_pixel: Conversion factor.
        camera_height: Height of camera.
        pitch_angle: Pitch angle in degrees.

    Returns:
        Transformation coordinates tensor.

    """
    ii, jj = torch.meshgrid(
        torch.arange(0, size, dtype=torch.float32),
        torch.arange(0, size, dtype=torch.float32),
        indexing="ij",
    )
    ii = ii.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    jj = jj.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    center_s = size / 2 - 0.5
    pitch_angle_rad = np.pi * pitch_angle / 180.0
    y_coord = ii - center_s
    x_coord = jj - center_s
    phi = torch.atan2(y_coord, x_coord)
    wrapped_phi = (phi + np.pi) % (2 * np.pi) - np.pi
    u_coord = (wrapped_phi + np.pi) / (2 * np.pi) * width
    radius_meters = torch.sqrt((ii - center_s) ** 2 + (jj - center_s) ** 2) * meter_per_pixel
    elevation_angle = torch.atan2(radius_meters, torch.tensor(camera_height))
    v_coord = (elevation_angle - pitch_angle_rad) / np.pi * height
    return torch.stack([u_coord, v_coord], dim=-1)


def process_panoramas_to_bev(
    input_folder: str,
    output_folder: str,
    panorama_height: int,
    panorama_width: int,
    bev_size: int,
    batch_size: int,
    camera_height: float,
    desired_coverage_meters: float,
    device: str = "cpu",
    pitch_angle: float = 0.0,
) -> None:
    """Convert panoramas to BEV (bird's eye view).

    Args:
        input_folder: Input panorama folder path.
        output_folder: Output BEV folder path.
        panorama_height: Height of input panoramas.
        panorama_width: Width of input panoramas.
        bev_size: Output BEV size.
        batch_size: Processing batch size.
        camera_height: Camera height in meters.
        desired_coverage_meters: Desired coverage area.
        device: Computation device (cpu/cuda).
        pitch_angle: Camera pitch angle in degrees.

    """
    if not Path(input_folder).exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        return

    ensure_dir(output_folder)
    meter_per_pixel = desired_coverage_meters / bev_size
    print(f"meter_per_pixel: {meter_per_pixel:.3f} (coverage {desired_coverage_meters}x{desired_coverage_meters}m)")
    transform = transforms.Compose([transforms.ToTensor()])
    files = sorted([f.name for f in Path(input_folder).iterdir() if f.name.lower().endswith((".jpg", ".png", ".jpeg"))])
    if not files:
        print(f"No images found for BEV conversion in {input_folder}")
        return
    uv_batch_template = bev_transform(
        batch_size,
        bev_size,
        panorama_height,
        panorama_width,
        meter_per_pixel,
        camera_height,
        pitch_angle,
    ).to(device)
    print(f"Found {len(files)} panoramas for BEV conversion...")
    
    i = 0
    pbar = tqdm(total=len(files), desc="Panorama to BEV conversion")
    while i < len(files):
        batch_files = files[i : i + batch_size]
        current_batch_size = len(batch_files)
        uv_batch = uv_batch_template[:current_batch_size] if current_batch_size < batch_size else uv_batch_template
        batch_images, batch_output_paths = [], []
        for file in batch_files:
            file_path = str(Path(input_folder) / file)
            output_path = str(Path(output_folder) / file)
            try:
                image_cv = cv2.imread(file_path)
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_cv)
                image_resized = resize_panorama(image, panorama_height, panorama_width)
                image_tensor = transform(image_resized).to(device)
                batch_images.append(image_tensor)
                batch_output_paths.append(output_path)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
        
        if not batch_images:
            i += batch_size
            pbar.update(current_batch_size)
            continue
            
        try:
            batch_tensor = torch.stack(batch_images)
            with torch.no_grad():
                bev_images = grid_sample(batch_tensor, uv_batch)
            for j, bev_image in enumerate(bev_images):
                save_image(bev_image, batch_output_paths[j])
                
            i += batch_size
            pbar.update(current_batch_size)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    print(f"\nCUDA Out Of Memory. Очистка кэша и уменьшение batch_size до {batch_size}...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    del batch_tensor
                    del batch_images
                else:
                    print("\nCUDA Out Of Memory: невозможно уменьшить batch_size (текущий размер 1).")
                    raise e
            else:
                raise e
                
    pbar.close()
    print(f"BEV conversion finished! Results in: {output_folder}")


def _process_single_panorama(
    entry: dict[str, float | str],
    images_dir: str,
    panorama_dir: str,
    panorama_shape: tuple[int, int],
    fov: float,
) -> None:
    """Process single panorama entry."""
    image_filename = str(entry["image_filename"])
    image_path = str(Path(images_dir) / image_filename)
    out_panorama_path = str(Path(panorama_dir) / image_filename)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Skip: image not found {image_path}")
        return

    wxyz = np.array([entry["w"], entry["x"], entry["y"], entry["z"]])
    yaw_rad, pitch_rad, roll_rad = quaternion_to_euler(wxyz)
    yaw, pitch, roll = math.degrees(yaw_rad), math.degrees(pitch_rad), math.degrees(roll_rad)
    print(f"\n--- Processing: {image_filename} ---")
    print(f"  Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°")

    fov_val = float(entry.get("fov", fov))
    panorama = create_inverse_panorama(image, panorama_shape[0], panorama_shape[1], wxyz, fov_val)
    cv2.imwrite(out_panorama_path, panorama)


def process_directory(
    json_path: str,
    images_dir: str,
    panorama_dir: str,
    bev_dir: str,
    panorama_shape: tuple[int, int],
    bev_size: int,
    fov: float,
    limit: int | str | None = None,
    batch_size: int = 8,
    device: str = "auto",
) -> None:
    """Process directory and create panoramas and BEV images.

    Args:
        json_path: Path to JSON metadata file.
        images_dir: Input images directory.
        panorama_dir: Output panorama directory.
        bev_dir: Output BEV directory.
        panorama_shape: Target panorama shape (width, height).
        bev_size: BEV output size.
        fov: Field of view in degrees.
        limit: Maximum number of images to process.
        batch_size: Batch size for BEV conversion.

    """
    if not Path(images_dir).exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
        
    try:
        data = load_json(json_path)
    except FileNotFoundError as e:
        print(f"Error loading JSON: {e}")
        return

    if not isinstance(data, list):
        print("Error: JSON data must be a list of entries.")
        return

    if limit is not None and str(limit).lower() != "none":
        limit_val = int(limit)
        if limit_val > 0:
            data = data[:limit_val]
            
    ensure_dir(panorama_dir)

    for entry in tqdm(data, desc="Panorama creation"):
        _process_single_panorama(entry, images_dir, panorama_dir, panorama_shape, fov)

    if data:
        alt = data[0]["alt"]
        fov_val = data[0].get("fov", fov)
        desired_coverage_meters = compute_coverage(alt, fov_val)
        
        if device == "auto" or device is None:
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            actual_device = device

        process_panoramas_to_bev(
            input_folder=panorama_dir,
            output_folder=bev_dir,
            panorama_height=panorama_shape[1],
            panorama_width=panorama_shape[0],
            bev_size=bev_size,
            batch_size=batch_size,
            camera_height=-alt,
            desired_coverage_meters=desired_coverage_meters,
            device=actual_device,
        )


def main() -> None:
    """Main entry point for BEV processing pipeline."""
    config_path = "config.yaml"
    if not Path(config_path).exists():
        print(f"Config file {config_path} not found!")
        return

    cfg = OmegaConf.load(config_path).pan_to_bev

    panorama_shape = tuple(cfg.panorama_shape) if 'panorama_shape' in cfg else (cfg.panorama_width, cfg.panorama_height)

    process_directory(
        json_path=cfg.json_path,
        images_dir=cfg.images_dir,
        panorama_dir=cfg.panorama_dir,
        bev_dir=cfg.bev_dir,
        panorama_shape=panorama_shape,
        bev_size=cfg.bev_size,
        fov=cfg.fov,
        limit=cfg.get("images_to_process_limit", None),
        batch_size=cfg.get("batch_size", 8),
        device=cfg.get("device", "auto"),
    )

if __name__ == "__main__":
    main()
