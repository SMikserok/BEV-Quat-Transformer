"""Convert panoramic images using quaternions.

This module provides functions for converting panoramic images based on
quaternion rotations and spherical coordinates.
"""

import math
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q

    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def create_uv_map(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Create UV map for panorama."""
    u = np.linspace(-np.pi, np.pi, width)
    v = np.linspace(-np.pi / 2, np.pi / 2, height)
    uu, vv = np.meshgrid(u, v)
    return uu, vv


def sphere_to_cartesian(phi: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical to Cartesian coordinates (NED system)."""
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return x, y, z


def create_inverse_panorama(
    image: np.ndarray,
    output_width: int,
    output_height: int,
    quaternion: np.ndarray,
    fov_deg: float = 90,
) -> np.ndarray:
    """Restore panorama from single image with known orientation.

    Center of panorama (horizontal) always corresponds to North.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image: must be a non-empty numpy array.")
        
    h, w = image.shape[:2]

    phi, theta = create_uv_map(output_width, output_height)

    world_x, world_y, world_z = sphere_to_cartesian(phi, theta)
    world_coords = np.stack([world_x.ravel(), world_y.ravel(), world_z.ravel()])

    r_matrix = quaternion_to_rotation_matrix(quaternion)

    camera_coords = r_matrix.T @ world_coords
    cam_x, cam_y, cam_z = camera_coords

    f = 0.5 * w / math.tan(math.radians(fov_deg) / 2)
    epsilon = 1e-6
    map_x = f * (cam_x / (cam_z + epsilon)) + w / 2
    map_y = f * (cam_y / (cam_z + epsilon)) + h / 2

    map_x = map_x.reshape(output_height, output_width).astype(np.float32)
    map_y = map_y.reshape(output_height, output_width).astype(np.float32)

    return cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

def main() -> None:
    """Main entry point for panorama restoration."""
    config_path = "config.yaml"
    if not Path(config_path).exists():
        print(f"Config file {config_path} not found!")
        return

    cfg = OmegaConf.load(config_path).image_eller_quat

    try:
        if not Path(cfg.image_path).exists():
            raise FileNotFoundError(f"File not found: {cfg.image_path}")
        original_image = cv2.imread(cfg.image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {cfg.image_path}")
    except Exception as e:
        print(f"Error: {e}")
        print("Creating test black image 512x512.")
        original_image = np.zeros((512, 512, 3), dtype=np.uint8)

    quaternion = np.array(cfg.quaternion)

    print("Restoring panorama from rotated image...")
    try:
        panorama = create_inverse_panorama(
            original_image, 
            cfg.output_width, 
            cfg.output_height, 
            quaternion, 
            cfg.horizontal_fov_deg
        )
        print("Panorama restored.")

        cv2.imwrite(cfg.output_path, panorama)
        print(f"Panorama saved as '{cfg.output_path}'")
    except Exception as e:
        print(f"Error creating panorama: {e}")

if __name__ == "__main__":
    main()
