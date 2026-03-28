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
import numpy as np

from src.classification import DiagnosticResult, GaitClassifier
from src.ingestion import IngestionConfig, VideoIngestion
from src.kinematics import KinematicAnalyzer
from src.pose_estimation import PoseEstimator, PoseResult
from ui.dashboard import GaitAnalysisDashboard

ANALYSIS_WINDOW_FRAMES: int = 90
COUNTDOWN_DURATION: float = 5.0

STATE_IDLE: str = "IDLE"
STATE_COUNTDOWN: str = "COUNTDOWN"
STATE_RECORDING: str = "RECORDING"


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


def run_pipeline(source: int | str = 0, target_fps: float = 30.0) -> int:
	"""Run the complete 4-stage gait analysis loop with state machine.

	Args:
		source: Webcam index or MP4 path.
		target_fps: Target ingestion frame rate.

	Returns:
		Process exit code where ``0`` indicates successful completion.
	"""

	pose_buffer: list[PoseResult] = []
	last_result: DiagnosticResult | None = None

	current_state: str = STATE_IDLE
	countdown_start_time: float = 0.0

	kinematic_analyzer = KinematicAnalyzer(visibility_threshold=0.5)
	gait_classifier = GaitClassifier(abnormal_threshold=10.0)
	dashboard = GaitAnalysisDashboard()

	drawing_utils = mp.solutions.drawing_utils

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

					if dashboard.start_requested and current_state == STATE_IDLE:
						current_state = STATE_COUNTDOWN
						countdown_start_time = time.perf_counter()
						dashboard.start_requested = False
						pose_buffer.clear()

					if current_state == STATE_COUNTDOWN:
						elapsed = time.perf_counter() - countdown_start_time
						countdown_time = max(0.0, COUNTDOWN_DURATION - elapsed)
						if countdown_time > 0:
							seconds_left = int(np.ceil(countdown_time))
							countdown_text = f"Starting in: {seconds_left}"
							text_size, baseline = cv2.getTextSize(
								countdown_text,
								cv2.FONT_HERSHEY_SIMPLEX,
								0.8,
								2,
							)
							text_width, text_height = text_size
							padding = 15
							badge_width = text_width + (2 * padding)
							badge_height = text_height + (2 * padding)

							frame_height, frame_width = packet.frame_bgr.shape[:2]
							badge_x1 = max((frame_width - badge_width) // 2, 0)
							badge_y1 = max(int(0.05 * frame_height), 0)
							badge_x2 = min(badge_x1 + badge_width, frame_width)
							badge_y2 = min(badge_y1 + badge_height, frame_height)

							cv2.rectangle(
								packet.frame_bgr,
								(badge_x1, badge_y1),
								(badge_x2, badge_y2),
								(40, 40, 40),
								thickness=-1,
							)

							text_x = badge_x1 + (badge_width - text_width) // 2
							text_y = badge_y1 + padding + text_height
							cv2.putText(
								packet.frame_bgr,
								countdown_text,
								(text_x, text_y),
								cv2.FONT_HERSHEY_SIMPLEX,
								0.8,
								(255, 255, 255),
								2,
								cv2.LINE_AA,
							)
						else:
							current_state = STATE_RECORDING

					if current_state == STATE_RECORDING:
						cv2.putText(
							packet.frame_bgr,
							"REC",
							(packet.frame_bgr.shape[1] - 120, 40),
							cv2.FONT_HERSHEY_SIMPLEX,
							1.2,
							(0, 0, 255),
							2,
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
							current_state = STATE_IDLE
							pose_buffer.clear()

					dashboard.update_display(
						frame_bgr=packet.frame_bgr,
						diagnostic_result=last_result,
						buffer_count=len(pose_buffer),
						recording_state=current_state,
					)
					if not dashboard._is_running:
						break
	except (RuntimeError, ValueError, FileNotFoundError) as exc:
		print(f"[main] pipeline error: {exc}")
		return 1

	return 0


def _run_cli() -> int:
	"""Run the CLI entry point for the orchestration script."""

	args = _build_cli_parser().parse_args()
	return run_pipeline(source=_parse_source(args.source), target_fps=args.fps)


if __name__ == "__main__":
	raise SystemExit(_run_cli())
