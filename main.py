"""Main orchestration entry point for 2D Kinematic Gait Analysis.

This script integrates all thesis pipeline stages in real time:
1. Stage 1 (ingestion): frame acquisition from webcam/MP4.
2. Stage 2 (pose estimation): bilateral lower-limb landmark extraction.
3. Stage 3 (kinematics): filtering and knee-angle computation.
4. Stage 4 (classification): Robinson SI and binary gait diagnosis.
"""

from __future__ import annotations

import argparse
import time

import cv2
import mediapipe as mp

from src.classification import DiagnosticResult, GaitClassifier
from src.ingestion import IngestionConfig, VideoIngestion
from src.kinematics import KinematicAnalyzer
from src.pose_estimation import PoseEstimator, PoseResult

ANALYSIS_WINDOW_FRAMES: int = 90


def _parse_source(value: str) -> int | str:
	"""Parse CLI source value into webcam index or path string."""

	text = value.strip()
	if text.isdigit():
		return int(text)
	return text


def _build_cli_parser() -> argparse.ArgumentParser:
	"""Create CLI parser for the main real-time orchestrator."""

	parser = argparse.ArgumentParser(description="Run 2D Kinematic Gait Analysis pipeline.")
	parser.add_argument(
		"--source",
		type=str,
		default="0",
		help="Webcam index (e.g., 0) or path to an MP4 file.",
	)
	parser.add_argument(
		"--fps",
		type=float,
		default=30.0,
		help="Target ingestion frame rate.",
	)
	return parser


def _build_status_text(
	last_result: DiagnosticResult | None,
	buffer_size: int,
	window_size: int,
) -> tuple[str, tuple[int, int, int]]:
	"""Create HUD status text and display color.

	Args:
		last_result: Most recent diagnostic result if available.
		buffer_size: Current number of valid pose frames in analysis buffer.
		window_size: Required frame count for one analysis window.

	Returns:
		Tuple ``(text, bgr_color)`` for OpenCV overlay.
	"""

	if last_result is None:
		return (f"Status: Buffering... ({buffer_size}/{window_size})", (0, 255, 255))

	if last_result.is_abnormal:
		return (
			f"Status: Abnormal Gait (SI: {last_result.symmetry_index:.1f}%)",
			(0, 0, 255),
		)

	return (
		f"Status: Normal Gait (SI: {last_result.symmetry_index:.1f}%)",
		(0, 255, 0),
	)


def run_pipeline(source: int | str = 0, target_fps: float = 30.0) -> int:
	"""Run the complete 4-stage gait analysis loop.

	Args:
		source: Webcam index or MP4 path.
		target_fps: Target ingestion frame rate.

	Returns:
		Process exit code where ``0`` indicates successful completion.
	"""

	pose_buffer: list[PoseResult] = []
	last_result: DiagnosticResult | None = None

	kinematic_analyzer = KinematicAnalyzer(visibility_threshold=0.5)
	gait_classifier = GaitClassifier(abnormal_threshold=10.0)

	drawing_utils = mp.solutions.drawing_utils
	window_name = "2D Kinematic Gait Analysis - press q to quit"

	# Light runtime telemetry for monitoring loop timing performance.
	last_fps_time = time.perf_counter()
	frames_since_fps_update = 0
	display_fps = 0.0

	try:
		with VideoIngestion(IngestionConfig(source=source, target_fps=target_fps)) as stream:
			with PoseEstimator() as pose_estimator:
				for packet in stream.frames():
					pose_result = pose_estimator.process_frame(packet.frame_bgr)

					if pose_estimator.last_pose_landmarks is not None:
						drawing_utils.draw_landmarks(
							packet.frame_bgr,
							pose_estimator.last_pose_landmarks,
							mp.solutions.pose.POSE_CONNECTIONS,
						)

					if pose_result is not None:
						pose_buffer.append(pose_result)

					if len(pose_buffer) >= ANALYSIS_WINDOW_FRAMES:
						left_leg_signal = [sample.left_leg for sample in pose_buffer]
						right_leg_signal = [sample.right_leg for sample in pose_buffer]

						left_angles = kinematic_analyzer.process_leg_signal(left_leg_signal)
						right_angles = kinematic_analyzer.process_leg_signal(right_leg_signal)

						last_result = gait_classifier.evaluate_symmetry(
							left_angles=left_angles,
							right_angles=right_angles,
						)
						pose_buffer.clear()

					status_text, status_color = _build_status_text(
						last_result=last_result,
						buffer_size=len(pose_buffer),
						window_size=ANALYSIS_WINDOW_FRAMES,
					)

					frames_since_fps_update += 1
					now = time.perf_counter()
					elapsed = now - last_fps_time
					if elapsed >= 0.5:
						display_fps = frames_since_fps_update / elapsed
						frames_since_fps_update = 0
						last_fps_time = now

					cv2.putText(
						packet.frame_bgr,
						status_text,
						(20, 35),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.7,
						status_color,
						2,
						cv2.LINE_AA,
					)
					cv2.putText(
						packet.frame_bgr,
						f"FPS: {display_fps:.1f}",
						(20, 65),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.6,
						(255, 255, 255),
						2,
						cv2.LINE_AA,
					)

					cv2.imshow(window_name, packet.frame_bgr)
					if cv2.waitKey(1) & 0xFF == ord("q"):
						break
	except (RuntimeError, ValueError, FileNotFoundError) as exc:
		print(f"[main] pipeline error: {exc}")
		return 1
	except cv2.error as exc:
		print("[main] OpenCV display unavailable.")
		print(f"[main] detail: {exc}")
		return 1
	finally:
		cv2.destroyAllWindows()

	return 0


def _run_cli() -> int:
	"""Run the CLI entry point for the orchestration script."""

	args = _build_cli_parser().parse_args()
	return run_pipeline(source=_parse_source(args.source), target_fps=args.fps)


if __name__ == "__main__":
	raise SystemExit(_run_cli())
