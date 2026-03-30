"""Main orchestration entry point for 2D Kinematic Gait Analysis.

This script integrates all thesis pipeline stages in real time:
1. Stage 1 (ingestion): frame acquisition from webcam/MP4.
2. Stage 2 (pose estimation): bilateral lower-limb landmark extraction.
3. Stage 3 (kinematics): filtering and knee-angle computation.
4. Stage 4 (classification): Robinson SI and binary gait diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import time
from pathlib import Path

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


def _log_to_csv(result: DiagnosticResult, log_path: Path, trial_num: int) -> None:
	"""Append a single diagnostic result row to the persistent CSV log.

	Args:
		result: Final bilateral gait diagnosis for the completed trial window.
		log_path: Destination CSV path used for longitudinal clinical audit.
		trial_num: Session-local trial number for the completed recording.
	"""

	file_exists = log_path.exists()
	with log_path.open("a", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file)
		if (not file_exists) or log_path.stat().st_size == 0:
			writer.writerow(
				[
					"Trial_Number",
					"Date",
					"Time",
					"Peak_Flexion_Left",
					"Peak_Flexion_Right",
					"Symmetry_Index",
					"Diagnosis",
				]
			)

		now = datetime.datetime.now()
		diagnosis = "Abnormal" if result.is_abnormal else "Normal"
		writer.writerow(
			[
				trial_num,
				now.strftime("%Y-%m-%d"),
				now.strftime("%H:%M:%S"),
				f"{result.peak_flexion_left:.2f}",
				f"{result.peak_flexion_right:.2f}",
				f"{result.symmetry_index:.2f}",
				diagnosis,
			]
		)


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
	video_writer: cv2.VideoWriter | None = None

	current_state: str = STATE_IDLE
	countdown_start_time: float = 0.0

	output_logs_dir = Path("data/output_logs")
	live_recordings_dir = Path("data/live_recordings")
	output_logs_dir.mkdir(parents=True, exist_ok=True)
	live_recordings_dir.mkdir(parents=True, exist_ok=True)
	session_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	log_file = Path(f"data/output_logs/session_{session_time}.csv")
	trial_counter = 1

	kinematic_analyzer = KinematicAnalyzer(visibility_threshold=0.5)
	gait_classifier = GaitClassifier(abnormal_threshold=10.0)
	dashboard = GaitAnalysisDashboard()
	while dashboard.selected_source is None and dashboard._is_running:
		dashboard.root.update()
		time.sleep(0.05)

	if not dashboard._is_running:
		return 0

	selected_source: int | str = dashboard.selected_source if dashboard.selected_source is not None else source
	is_live = isinstance(selected_source, int)
	if not is_live:
		dashboard.start_button.pack_forget()
		current_state = STATE_RECORDING

	drawing_utils = mp.solutions.drawing_utils
	last_frame_bgr: np.ndarray | None = None
	user_stopped: bool = False
	mp4_stop_requested: bool = False

	def _finalize_buffered_trial() -> DiagnosticResult:
		"""Run kinematic and symmetry analysis for the currently buffered pose window.

		Returns:
			Diagnostic result computed from the current pose buffer.

		Raises:
			ValueError: If no valid pose samples are available.
		"""

		nonlocal trial_counter
		if len(pose_buffer) == 0:
			raise ValueError("No valid pose frames available for analysis.")

		left_leg_signal = [sample.left_leg for sample in pose_buffer]
		right_leg_signal = [sample.right_leg for sample in pose_buffer]

		left_angles = kinematic_analyzer.process_leg_signal(left_leg_signal)
		right_angles = kinematic_analyzer.process_leg_signal(right_leg_signal)

		result = gait_classifier.evaluate_symmetry(
			left_angles=left_angles,
			right_angles=right_angles,
		)
		_log_to_csv(result=result, log_path=log_file, trial_num=trial_counter)
		trial_counter += 1
		return result

	try:
		with VideoIngestion(IngestionConfig(source=selected_source, target_fps=target_fps)) as stream:
			with PoseEstimator() as pose_estimator:
				for packet in stream.frames():
					last_frame_bgr = packet.frame_bgr.copy()
					pose_result = pose_estimator.process_frame(packet.frame_bgr)

					if pose_estimator.last_pose_landmarks is not None:
						drawing_utils.draw_landmarks(
							packet.frame_bgr,
							pose_estimator.last_pose_landmarks,
							mp.solutions.pose.POSE_CONNECTIONS,
						)

					if is_live and dashboard.start_requested and current_state == STATE_IDLE:
						current_state = STATE_COUNTDOWN
						countdown_start_time = time.perf_counter()
						dashboard.start_requested = False
						pose_buffer.clear()

					if is_live and current_state == STATE_COUNTDOWN:
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
							filename = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
							save_path = str(live_recordings_dir / filename)
							frame_size = (packet.frame_bgr.shape[1], packet.frame_bgr.shape[0])
							video_writer = cv2.VideoWriter(
								save_path,
								cv2.VideoWriter_fourcc(*"mp4v"),
								target_fps,
								frame_size,
							)
							if not video_writer.isOpened():
								raise RuntimeError(f"Failed to open video writer at: {save_path}")

					if current_state == STATE_RECORDING:
						if is_live and video_writer is not None:
							video_writer.write(packet.frame_bgr)

						if is_live:
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

						if dashboard.stop_requested:
							dashboard.stop_requested = False
							if is_live:
								if len(pose_buffer) > 0:
									last_result = _finalize_buffered_trial()
								current_state = STATE_IDLE
								pose_buffer.clear()
								if video_writer is not None:
									video_writer.release()
									video_writer = None
							else:
								mp4_stop_requested = True
								break

						if is_live and len(pose_buffer) >= ANALYSIS_WINDOW_FRAMES:
							last_result = _finalize_buffered_trial()
							current_state = STATE_IDLE
							pose_buffer.clear()
							if video_writer is not None:
								video_writer.release()
								video_writer = None

					dashboard.update_display(
						frame_bgr=packet.frame_bgr,
						diagnostic_result=last_result,
						buffer_count=len(pose_buffer),
						recording_state=current_state,
					)
					if not dashboard._is_running:
						user_stopped = True
						break

				if (not is_live) and (not user_stopped):
					if len(pose_buffer) == 0:
						if not mp4_stop_requested:
							raise ValueError("No valid pose frames were extracted from the selected MP4 video.")
					else:
						last_result = _finalize_buffered_trial()

					current_state = STATE_IDLE

					if last_frame_bgr is not None and dashboard._is_running:
						dashboard.update_display(
							frame_bgr=last_frame_bgr,
							diagnostic_result=last_result,
							buffer_count=len(pose_buffer),
							recording_state=current_state,
						)

					# Keep the final MP4 diagnosis visible until the user closes the window.
					while dashboard._is_running:
						dashboard.root.update()
						time.sleep(0.05)
	except (RuntimeError, ValueError, FileNotFoundError) as exc:
		print(f"[main] pipeline error: {exc}")
		return 1
	finally:
		if video_writer is not None:
			video_writer.release()

	return 0


def _run_cli() -> int:
	"""Run the CLI entry point for the orchestration script."""

	args = _build_cli_parser().parse_args()
	return run_pipeline(source=_parse_source(args.source), target_fps=args.fps)


if __name__ == "__main__":
	raise SystemExit(_run_cli())
