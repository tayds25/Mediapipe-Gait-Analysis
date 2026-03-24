"""
Stage 2: Pose Estimation Module

This module wraps MediaPipe BlazePose to extract only the lower-limb landmarks
required by the thesis pipeline. The design intentionally emits compact,
strongly-typed data structures so downstream kinematics and classification
stages remain decoupled from MediaPipe internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class JointData:
	"""Single 2D joint observation from MediaPipe Pose.

	Attributes:
		x: Normalized horizontal coordinate in image space [0, 1].
		y: Normalized vertical coordinate in image space [0, 1].
		visibility: MediaPipe landmark confidence score ``c`` in [0, 1].
	"""

	x: float
	y: float
	visibility: float


@dataclass(slots=True, frozen=True)
class LegKinematics:
	"""Lower-limb joint tuple for one leg side.

	Attributes:
		hip: Hip joint observation.
		knee: Knee joint observation.
		ankle: Ankle joint observation.
	"""

	hip: JointData
	knee: JointData
	ankle: JointData


@dataclass(slots=True, frozen=True)
class PoseResult:
	"""Bilateral lower-limb pose container used by downstream modules.

	Attributes:
		left_leg: Extracted left hip-knee-ankle observations.
		right_leg: Extracted right hip-knee-ankle observations.
	"""

	left_leg: LegKinematics
	right_leg: LegKinematics


class PoseEstimator:
	"""MediaPipe BlazePose wrapper specialized for gait lower-limb extraction.

	Landmark mapping is fixed to BlazePose indices:
	- Left hip/knee/ankle: 23, 25, 27
	- Right hip/knee/ankle: 24, 26, 28
	"""

	_LEFT_HIP_IDX: int = 23
	_LEFT_KNEE_IDX: int = 25
	_LEFT_ANKLE_IDX: int = 27
	_RIGHT_HIP_IDX: int = 24
	_RIGHT_KNEE_IDX: int = 26
	_RIGHT_ANKLE_IDX: int = 28

	def __init__(
		self,
		static_image_mode: bool = False,
		model_complexity: int = 1,
		smooth_landmarks: bool = True,
		min_detection_confidence: float = 0.5,
		min_tracking_confidence: float = 0.5,
	) -> None:
		"""Initialize the MediaPipe Pose estimator.

		Args:
			static_image_mode: Whether to treat each frame independently.
			model_complexity: BlazePose complexity level (0, 1, or 2).
			smooth_landmarks: Enables internal temporal smoothing.
			min_detection_confidence: Minimum person-detection confidence.
			min_tracking_confidence: Minimum pose-tracking confidence.

		Raises:
			ValueError: If confidence thresholds are outside [0, 1].
		"""

		if not 0.0 <= min_detection_confidence <= 1.0:
			raise ValueError("min_detection_confidence must be within [0, 1].")
		if not 0.0 <= min_tracking_confidence <= 1.0:
			raise ValueError("min_tracking_confidence must be within [0, 1].")

		self._mp_pose: Any = mp.solutions.pose
		self._pose: Any = self._mp_pose.Pose(
			static_image_mode=static_image_mode,
			model_complexity=model_complexity,
			smooth_landmarks=smooth_landmarks,
			min_detection_confidence=min_detection_confidence,
			min_tracking_confidence=min_tracking_confidence,
		)
		self._last_mp_result: Any | None = None

	@property
	def last_pose_landmarks(self) -> Any | None:
		"""Return the latest raw MediaPipe landmark list for visualization only."""

		if self._last_mp_result is None:
			return None
		return self._last_mp_result.pose_landmarks

	def close(self) -> None:
		"""Release MediaPipe native resources explicitly."""

		self._pose.close()

	def __enter__(self) -> "PoseEstimator":
		"""Support context-manager usage for deterministic cleanup."""

		return self

	def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
		"""Ensure resource cleanup on context exit."""

		_ = (exc_type, exc, tb)
		self.close()

	def process_frame(self, frame_bgr: NDArray[np.uint8]) -> PoseResult | None:
		"""Estimate pose from a BGR frame and extract bilateral lower-limb joints.

		Academic rationale:
			The ingestion module emits BGR frames (OpenCV native format), while
			MediaPipe expects RGB ordering. Explicit conversion is required to avoid
			color-channel mismatch that can degrade landmark stability.

		Args:
			frame_bgr: Input frame in OpenCV BGR format.

		Returns:
			A ``PoseResult`` with left and right leg joints if pose landmarks are
			available, otherwise ``None``.

		Raises:
			ValueError: If the frame is empty or does not have 3 color channels.
		"""

		if frame_bgr.size == 0:
			raise ValueError("frame_bgr cannot be empty.")
		if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
			raise ValueError("frame_bgr must have shape (H, W, 3).")

		frame_rgb: NDArray[np.uint8] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		mp_result: Any = self._pose.process(frame_rgb)
		self._last_mp_result = mp_result

		if mp_result.pose_landmarks is None:
			return None

		landmarks: Any = mp_result.pose_landmarks.landmark

		left_leg = LegKinematics(
			hip=self._landmark_to_joint(landmarks[self._LEFT_HIP_IDX]),
			knee=self._landmark_to_joint(landmarks[self._LEFT_KNEE_IDX]),
			ankle=self._landmark_to_joint(landmarks[self._LEFT_ANKLE_IDX]),
		)
		right_leg = LegKinematics(
			hip=self._landmark_to_joint(landmarks[self._RIGHT_HIP_IDX]),
			knee=self._landmark_to_joint(landmarks[self._RIGHT_KNEE_IDX]),
			ankle=self._landmark_to_joint(landmarks[self._RIGHT_ANKLE_IDX]),
		)

		return PoseResult(left_leg=left_leg, right_leg=right_leg)

	@staticmethod
	def _landmark_to_joint(landmark: Any) -> JointData:
		"""Convert a MediaPipe landmark proto into ``JointData``."""

		return JointData(
			x=float(landmark.x),
			y=float(landmark.y),
			visibility=float(landmark.visibility),
		)


def _run_webcam_pose_smoke_test(camera_index: int = 0) -> int:
	"""Run a webcam smoke test for Stage 2 pose extraction.

	The function streams webcam frames from the ingestion module, estimates
	pose, overlays MediaPipe skeleton visualization, and prints left-knee
	visibility as proof of targeted landmark extraction.

	Args:
		camera_index: Webcam index (typically ``0``).

	Returns:
		Process exit code where ``0`` indicates success.
	"""

	# Local import keeps module boundaries explicit and avoids cyclic imports.
	from ingestion import IngestionConfig, VideoIngestion

	drawing_utils: Any = mp.solutions.drawing_utils
	window_name: str = "Pose Estimation Smoke Test - press q to quit"

	try:
		with VideoIngestion(IngestionConfig(source=camera_index, target_fps=30.0)) as stream:
			with PoseEstimator() as estimator:
				for packet in stream.frames():
					result = estimator.process_frame(packet.frame_bgr)

					if estimator.last_pose_landmarks is not None:
						drawing_utils.draw_landmarks(
							packet.frame_bgr,
							estimator.last_pose_landmarks,
							mp.solutions.pose.POSE_CONNECTIONS,
						)

					if result is not None and packet.index % 15 == 0:
						print(
							"[pose_estimation] Left knee visibility: "
							f"{result.left_leg.knee.visibility:.3f}"
						)

					cv2.imshow(window_name, packet.frame_bgr)
					if cv2.waitKey(1) & 0xFF == ord("q"):
						break
	except (RuntimeError, ValueError, FileNotFoundError) as exc:
		print(f"[pose_estimation] error: {exc}")
		return 1
	except cv2.error as exc:
		print("[pose_estimation] OpenCV display unavailable.")
		print(f"[pose_estimation] detail: {exc}")
		return 1
	finally:
		cv2.destroyAllWindows()

	return 0


if __name__ == "__main__":
	raise SystemExit(_run_webcam_pose_smoke_test())
