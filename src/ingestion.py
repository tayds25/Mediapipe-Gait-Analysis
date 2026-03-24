"""
Stage 1: Data Ingestion Module
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import time
from typing import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class IngestionConfig:
	"""Configuration for OpenCV-based frame ingestion.

	Attributes:
		source: Video source as either webcam index (for example, ``0``) or
			a path to an MP4 file.
		target_fps: Target frame emission rate in frames per second. The
			generator applies pacing to approximate this rate.
		resize_to: Optional ``(width, height)`` resize target. When ``None``,
			native source resolution is preserved.
	"""

	source: int | str | Path
	target_fps: float = 30.0
	resize_to: tuple[int, int] | None = None


@dataclass(slots=True, frozen=True)
class FramePacket:
	"""Discrete frame unit emitted by the ingestion stage.

	Attributes:
		index: Zero-based sequential frame index.
		timestamp_s: Monotonic acquisition timestamp in seconds from stream
			start, useful for consistent downstream temporal analysis.
		frame_bgr: Frame array in OpenCV BGR channel order.
	"""

	index: int
	timestamp_s: float
	frame_bgr: NDArray[np.uint8]


class VideoIngestion:
	"""OpenCV stream wrapper for real-time and offline frame extraction.

	Design rationale:
		The ingestion stage is intentionally isolated from pose estimation and
		kinematic computation to preserve architectural decoupling and support
		reproducible benchmarking across webcam and MP4 sources.
	"""

	def __init__(self, config: IngestionConfig) -> None:
		"""Initialize the ingestion object.

		Args:
			config: Static capture and pacing configuration.

		Raises:
			ValueError: If ``target_fps`` is not strictly positive.
			FileNotFoundError: If a file source path does not exist.
		"""

		if config.target_fps <= 0.0:
			raise ValueError("target_fps must be > 0.")

		self._config: IngestionConfig = config
		self._source: int | str = self._normalize_source(config.source)
		self._capture: cv2.VideoCapture | None = None
		self._active_backend: int | None = None

	@staticmethod
	def _normalize_source(source: int | str | Path) -> int | str:
		"""Normalize and validate a capture source.

		Args:
			source: Webcam index or path-like video source.

		Returns:
			Normalized source value accepted by ``cv2.VideoCapture``.

		Raises:
			FileNotFoundError: If a path-like source does not exist.
			ValueError: If a path-like source is empty.
		"""

		if isinstance(source, int):
			return source

		text_source: str = str(source).strip()
		if text_source == "":
			raise ValueError("Video source cannot be empty.")

		if text_source.isdigit():
			return int(text_source)

		source_path = Path(text_source)
		if not source_path.exists():
			raise FileNotFoundError(f"Video file not found: {source_path}")
		return str(source_path)

	def open(self) -> None:
		"""Open the underlying OpenCV capture stream.

		Raises:
			RuntimeError: If OpenCV fails to open the source.
		"""

		if self._capture is not None and self._capture.isOpened():
			return

		self._capture, self._active_backend = self._create_capture()
		if not self._capture.isOpened():
			if isinstance(self._source, int):
				available_indices: list[int] = self.probe_camera_indices(max_index=5)
				if len(available_indices) == 0:
					raise RuntimeError(
						"Failed to open webcam index "
						f"{self._source}. No camera indices in range [0, 5] responded. "
						"Verify webcam permissions, camera drivers, and that no other app is locking the device."
					)
				raise RuntimeError(
					"Failed to open webcam index "
					f"{self._source}. Available indices detected: {available_indices}."
				)

			raise RuntimeError(f"Failed to open video source: {self._source}")

		if isinstance(self._source, int):
			self._capture.set(cv2.CAP_PROP_FPS, self._config.target_fps)

		self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

	def _create_capture(self) -> tuple[cv2.VideoCapture, int | None]:
		"""Create an OpenCV capture with backend fallback for Windows webcams.

		For camera indices on Windows, DirectShow and Media Foundation are tested
		first because they often reduce startup and acquisition latency compared
		to backend-auto selection.

		Returns:
			Tuple of ``(capture, backend_id_or_none)``.
		"""

		if isinstance(self._source, str):
			return cv2.VideoCapture(self._source), None

		preferred_backends: list[int] = self._preferred_windows_backends()

		for backend in preferred_backends:
			capture = cv2.VideoCapture(self._source, backend)
			if capture.isOpened():
				return capture, backend
			capture.release()

		return cv2.VideoCapture(self._source), None

	@staticmethod
	def _preferred_windows_backends() -> list[int]:
		"""Return preferred OpenCV backend IDs for Windows webcam capture."""

		if os.name != "nt":
			return []

		backends: list[int] = []
		if hasattr(cv2, "CAP_DSHOW"):
			backends.append(int(cv2.CAP_DSHOW))
		if hasattr(cv2, "CAP_MSMF"):
			backends.append(int(cv2.CAP_MSMF))
		return backends

	@staticmethod
	def probe_camera_indices(max_index: int = 5) -> list[int]:
		"""Probe webcam indices and return responsive camera IDs.

		Args:
			max_index: Maximum index (inclusive) to probe.

		Returns:
			List of camera indices that can be opened and read successfully.

		Raises:
			ValueError: If ``max_index`` is negative.
		"""

		if max_index < 0:
			raise ValueError("max_index must be >= 0.")

		available: list[int] = []
		preferred_backends: list[int] = VideoIngestion._preferred_windows_backends()
		for index in range(max_index + 1):
			found_for_index: bool = False
			for backend in preferred_backends:
				capture = cv2.VideoCapture(index, backend)
				if not capture.isOpened():
					capture.release()
					continue

				ok, _ = capture.read()
				capture.release()
				if ok:
					available.append(index)
					found_for_index = True
					break

			if found_for_index:
				continue

			# Fall back to backend auto-selection when preferred backends do not work.
			capture = cv2.VideoCapture(index)
			if not capture.isOpened():
				capture.release()
				continue

			ok, _ = capture.read()
			capture.release()
			if ok:
				available.append(index)

		return available

	def close(self) -> None:
		"""Release capture resources safely."""

		if self._capture is not None:
			self._capture.release()
			self._capture = None

	def __enter__(self) -> "VideoIngestion":
		"""Open the stream for context manager usage."""

		self.open()
		return self

	def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
		"""Guarantee resource release on context manager exit."""

		_ = (exc_type, exc, tb)
		self.close()

	def frames(self) -> Iterator[FramePacket]:
		"""Yield discrete frames paced to the configured target FPS.

		Yields:
			``FramePacket`` instances in source order.

		Raises:
			RuntimeError: If the stream is not open.
		"""

		if self._capture is None or not self._capture.isOpened():
			raise RuntimeError("Stream is not open. Call open() first.")

		frame_interval_s: float = 1.0 / self._config.target_fps
		next_emit_time_s: float = time.perf_counter()
		stream_start_s: float = next_emit_time_s

		frame_index: int = 0
		while True:
			success, frame = self._capture.read()
			if not success:
				break

			if self._config.resize_to is not None:
				frame = cv2.resize(frame, self._config.resize_to, interpolation=cv2.INTER_LINEAR)

			current_time_s: float = time.perf_counter()
			sleep_s: float = next_emit_time_s - current_time_s
			if sleep_s > 0.0:
				time.sleep(sleep_s)
				current_time_s = time.perf_counter()

			yield FramePacket(
				index=frame_index,
				timestamp_s=current_time_s - stream_start_s,
				frame_bgr=frame,
			)

			frame_index += 1
			next_emit_time_s = max(next_emit_time_s + frame_interval_s, current_time_s)


def _run_webcam_smoke_test(camera_index: int = 0) -> int:
	"""Run a safe webcam smoke test for manual ingestion validation.

	Args:
		camera_index: OpenCV camera index, usually ``0`` for default webcam.

	Returns:
		Process exit code where ``0`` indicates success.
	"""

	config = IngestionConfig(source=camera_index, target_fps=30.0)
	window_name = "Ingestion Webcam Test - press q to quit"

	try:
		with VideoIngestion(config) as stream:
			for packet in stream.frames():
				cv2.imshow(window_name, packet.frame_bgr)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
	except RuntimeError as exc:
		print(f"[ingestion] stream error: {exc}")
		return 1
	except cv2.error as exc:
		print("[ingestion] OpenCV display unavailable. Verify WSLg/GUI support.")
		print(f"[ingestion] detail: {exc}")
		return 1
	finally:
		cv2.destroyAllWindows()

	return 0


def _build_cli_parser() -> argparse.ArgumentParser:
	"""Create CLI parser for basic ingestion testing."""

	parser = argparse.ArgumentParser(description="Run ingestion module smoke test.")
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
		help="Target output frame rate.",
	)
	parser.add_argument(
		"--probe-cameras",
		action="store_true",
		help="Probe camera indices [0..5], print results, and exit.",
	)
	return parser


def _run_cli_smoke_test() -> int:
	"""Run the CLI-configured ingestion smoke test."""

	parser = _build_cli_parser()
	args = parser.parse_args()

	if args.probe_cameras:
		indices = VideoIngestion.probe_camera_indices(max_index=5)
		if len(indices) == 0:
			print("[ingestion] no responsive webcam indices found in range [0, 5].")
			return 1

		print(f"[ingestion] responsive webcam indices: {indices}")
		return 0

	source: int | str = int(args.source) if args.source.isdigit() else args.source

	config = IngestionConfig(source=source, target_fps=args.fps)
	window_name = "Ingestion Stream Test - press q to quit"

	try:
		with VideoIngestion(config) as stream:
			for packet in stream.frames():
				cv2.imshow(window_name, packet.frame_bgr)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
	except (RuntimeError, ValueError, FileNotFoundError) as exc:
		print(f"[ingestion] error: {exc}")
		return 1
	except cv2.error as exc:
		print("[ingestion] OpenCV display unavailable. Verify WSLg/GUI support.")
		print(f"[ingestion] detail: {exc}")
		return 1
	finally:
		cv2.destroyAllWindows()

	return 0


if __name__ == "__main__":
	raise SystemExit(_run_cli_smoke_test())
