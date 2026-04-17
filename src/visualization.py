from __future__ import annotations

import os
from pathlib import Path

# Use EGL for headless rendering on servers.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v * 0.0
    return v / n


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up))
    true_up = _normalize(np.cross(right, forward))

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def compute_bounds(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not arrays:
        raise ValueError("No arrays provided for bounds")

    bmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for arr in arrays:
        pts = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
        bmin = np.minimum(bmin, np.min(pts, axis=0))
        bmax = np.maximum(bmax, np.max(pts, axis=0))

    center = 0.5 * (bmin + bmax)
    extent = np.maximum(bmax - bmin, 1e-6)
    scale = float(np.linalg.norm(extent))
    return bmin, bmax, center, scale


def default_camera_pose(center: np.ndarray, scale: float) -> np.ndarray:
    dist = max(scale * 1.35, 1e-2)
    eye = center + dist * np.array([1.15, 0.65, 1.05], dtype=np.float64)
    return look_at(eye=eye, target=center, up=np.array([0.0, 1.0, 0.0], dtype=np.float64))


def _color_rgba(color_rgb: tuple[int, int, int], alpha: float) -> list[float]:
    r, g, b = color_rgb
    return [r / 255.0, g / 255.0, b / 255.0, float(alpha)]


def _new_scene(camera_pose: np.ndarray, yfov: float = np.pi / 4.0) -> tuple[pyrender.Scene, pyrender.Node]:
    scene = pyrender.Scene(bg_color=np.array([242, 244, 248, 255], dtype=np.uint8), ambient_light=[0.28, 0.28, 0.28])

    cam = pyrender.PerspectiveCamera(yfov=float(yfov))
    cam_node = scene.add(cam, pose=camera_pose)

    key = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    scene.add(key, pose=camera_pose)

    fill_pose = np.array(camera_pose, dtype=np.float64)
    fill_pose[:3, 3] += np.array([-0.7, 0.4, -0.6], dtype=np.float64)
    scene.add(fill, pose=fill_pose)
    return scene, cam_node


def render_mesh_frames(
    vertices_seq: np.ndarray,
    faces: np.ndarray,
    *,
    width: int,
    height: int,
    camera_pose: np.ndarray,
    mesh_color: tuple[int, int, int],
) -> list[np.ndarray]:
    vertices_seq = np.asarray(vertices_seq, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices_seq.ndim != 3 or vertices_seq.shape[-1] != 3:
        raise ValueError(f"Expected vertices sequence (T,N,3), got {vertices_seq.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces (F,3), got {faces.shape}")

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.85,
        baseColorFactor=_color_rgba(mesh_color, 1.0),
    )

    scene, _ = _new_scene(camera_pose=camera_pose)
    renderer = pyrender.OffscreenRenderer(viewport_width=int(width), viewport_height=int(height))
    frames: list[np.ndarray] = []

    try:
        for verts in vertices_seq:
            tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh = pyrender.Mesh.from_trimesh(tri, smooth=False, material=material)
            node = scene.add(mesh)
            rgb, _depth = renderer.render(scene)
            frames.append(np.asarray(rgb, dtype=np.uint8))
            scene.remove_node(node)
    finally:
        renderer.delete()

    return frames


def _joint_meshes(
    joints: np.ndarray,
    parents: np.ndarray,
    scale: float,
    joint_color: tuple[int, int, int],
    bone_color: tuple[int, int, int],
) -> list[pyrender.Mesh]:
    meshes: list[pyrender.Mesh] = []
    jr = max(scale * 0.018, 1e-4)
    br = max(scale * 0.006, 5e-5)

    joint_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.8,
        baseColorFactor=_color_rgba(joint_color, 1.0),
    )
    bone_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.8,
        baseColorFactor=_color_rgba(bone_color, 1.0),
    )

    for j in joints:
        sph = trimesh.creation.icosphere(subdivisions=1, radius=jr)
        sph.apply_translation(np.asarray(j, dtype=np.float64))
        meshes.append(pyrender.Mesh.from_trimesh(sph, smooth=True, material=joint_mat))

    for i, p in enumerate(parents.astype(int)):
        if p < 0:
            continue
        seg = np.vstack([joints[p], joints[i]])
        if np.linalg.norm(seg[1] - seg[0]) < 1e-8:
            continue
        cyl = trimesh.creation.cylinder(radius=br, sections=12, segment=seg)
        meshes.append(pyrender.Mesh.from_trimesh(cyl, smooth=True, material=bone_mat))

    return meshes


def render_rig_overlay(
    canonical_vertices: np.ndarray,
    canonical_faces: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    *,
    out_image: Path,
    width: int,
    height: int,
    turntable_out: Path | None,
    fps: int,
    turntable_frames: int,
) -> None:
    canonical_vertices = np.asarray(canonical_vertices, dtype=np.float32)
    canonical_faces = np.asarray(canonical_faces, dtype=np.int32)
    joints = np.asarray(joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int32)

    _bmin, _bmax, center, scale = compute_bounds([canonical_vertices, joints])
    camera_pose = default_camera_pose(center=center, scale=scale)

    scene, cam_node = _new_scene(camera_pose=camera_pose)
    renderer = pyrender.OffscreenRenderer(viewport_width=int(width), viewport_height=int(height))

    mesh_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.95,
        baseColorFactor=_color_rgba((188, 197, 210), 0.35),
    )

    tri = trimesh.Trimesh(vertices=canonical_vertices, faces=canonical_faces, process=False)
    scene.add(pyrender.Mesh.from_trimesh(tri, smooth=False, material=mesh_material))

    skel_meshes = _joint_meshes(
        joints=joints,
        parents=parents,
        scale=scale,
        joint_color=(225, 80, 70),
        bone_color=(50, 110, 220),
    )
    for m in skel_meshes:
        scene.add(m)

    out_image.parent.mkdir(parents=True, exist_ok=True)

    try:
        overlay_rgb, _ = renderer.render(scene)

        # Skeleton-only view helps readability when many bones are internal.
        skel_scene, _ = _new_scene(camera_pose=camera_pose)
        for m in skel_meshes:
            skel_scene.add(m)
        skel_rgb, _ = renderer.render(skel_scene)

        panel = np.concatenate([overlay_rgb, skel_rgb], axis=1)
        write_image(out_image, panel)

        if turntable_out is not None and turntable_frames > 0:
            tt_frames: list[np.ndarray] = []
            dist = max(scale * 1.9, 1e-2)
            for i in range(int(turntable_frames)):
                theta = 2.0 * np.pi * (i / max(1, turntable_frames))
                eye = center + dist * np.array([1.35 * np.cos(theta), 0.58, 1.35 * np.sin(theta)], dtype=np.float64)
                scene.set_pose(cam_node, look_at(eye=eye, target=center, up=np.array([0.0, 1.0, 0.0], dtype=np.float64)))
                frame, _ = renderer.render(scene)
                tt_frames.append(np.asarray(frame, dtype=np.uint8))
            save_video(turntable_out, tt_frames, fps=fps)
    finally:
        renderer.delete()


def save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError(f"No frames to save for video: {path}")

    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)

    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), float(fps), (w, h))
        if not writer.isOpened():
            continue
        try:
            for frame in frames:
                if frame.shape[:2] != (h, w):
                    raise ValueError("All frames must share one resolution")
                writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
            writer.release()
            return
        except Exception:
            writer.release()
            continue

    raise RuntimeError(f"Could not open video writer for: {path}")


def write_image(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def make_side_by_side(left_frames: list[np.ndarray], right_frames: list[np.ndarray]) -> list[np.ndarray]:
    t = min(len(left_frames), len(right_frames))
    if t <= 0:
        return []
    out = []
    for i in range(t):
        out.append(np.concatenate([left_frames[i], right_frames[i]], axis=1))
    return out


def keyframe_indices(n_frames: int) -> list[int]:
    if n_frames <= 0:
        return []
    idx = [0, n_frames // 2, n_frames - 1]
    out: list[int] = []
    for i in idx:
        if i not in out:
            out.append(i)
    return out
