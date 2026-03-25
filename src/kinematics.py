"""Stage 3: Filter and Kinematic Computation Module.

This module transforms lower-limb landmark time series into continuous knee
flexion angle trajectories. The implementation follows thesis constraints:
visibility thresholding, linear interpolation for missing data, Savitzky-
Golay smoothing, and vectorized interior-angle computation using arctan2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import savgol_filter

JointName = Literal["hip", "knee", "ankle"]


class JointLike(Protocol):
	"""Structural type for joint inputs from Stage 2 and synthetic tests."""

	x: float
	y: float
	visibility: float


class LegKinematicsLike(Protocol):
	"""Structural type for leg inputs containing hip, knee, and ankle joints."""

	hip: JointLike
	knee: JointLike
	ankle: JointLike


class KinematicAnalyzer:
	"""Process lower-limb landmark time series into knee angle trajectories.

	Pipeline steps:
	1. Extract joint coordinate arrays over time.
	2. Apply visibility thresholding (c < 0.5 -> NaN).
	3. Interpolate NaN gaps linearly with pandas.
	4. Smooth coordinate signals with Savitzky-Golay.
	5. Compute interior knee flexion angle using vectorized arctan2 math.
	"""

	def __init__(self, visibility_threshold: float = 0.5) -> None:
		"""Initialize the kinematic analyzer.

		Args:
			visibility_threshold: Confidence cutoff ``c`` used to mark unreliable
				landmark samples as missing prior to interpolation.

		Raises:
			ValueError: If visibility_threshold is outside [0, 1].
		"""

		if not 0.0 <= visibility_threshold <= 1.0:
			raise ValueError("visibility_threshold must be within [0, 1].")

		self._visibility_threshold: float = visibility_threshold
		self._savgol_window_length: int = 11
		self._savgol_polyorder: int = 3

	def extract_joint_signal(
		self,
		leg_data: Sequence[LegKinematicsLike],
		joint_name: JointName,
	) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
		"""Extract thresholded x/y arrays for one joint across time.

		Academic rationale:
			The thesis requires dropping low-confidence detections before filtering.
			For any sample where visibility ``c < 0.5``, x and y are replaced with
			``np.nan``. This explicitly marks missing observations so they can be
			reconstructed by interpolation.

		Args:
			leg_data: Sequential leg observations over frames.
			joint_name: Joint key to extract ("hip", "knee", or "ankle").

		Returns:
			Tuple ``(x_signal, y_signal)`` as float64 arrays.

		Raises:
			ValueError: If leg_data is empty.
		"""

		if len(leg_data) == 0:
			raise ValueError("leg_data must contain at least one frame.")

		x_values: list[float] = []
		y_values: list[float] = []

		for sample in leg_data:
			joint = getattr(sample, joint_name)
			if float(joint.visibility) < self._visibility_threshold:
				x_values.append(np.nan)
				y_values.append(np.nan)
				continue

			x_values.append(float(joint.x))
			y_values.append(float(joint.y))

		return (
			np.asarray(x_values, dtype=np.float64),
			np.asarray(y_values, dtype=np.float64),
		)

	@staticmethod
	def interpolate_nan_signal(
		x_signal: NDArray[np.float64],
		y_signal: NDArray[np.float64],
	) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
		"""Linearly interpolate missing coordinate values.

		Academic rationale:
			Savitzky-Golay filtering cannot process NaN samples. Missing values
			from visibility thresholding are filled using linear interpolation with
			pandas to restore continuous signals for filtering.

		Args:
			x_signal: 1D x-coordinate signal that may contain NaN values.
			y_signal: 1D y-coordinate signal that may contain NaN values.

		Returns:
			Tuple ``(x_interp, y_interp)`` with NaN gaps filled.

		Raises:
			ValueError: If x and y lengths do not match.
		"""

		if x_signal.shape[0] != y_signal.shape[0]:
			raise ValueError("x_signal and y_signal must have the same length.")

		x_interp = (
			pd.Series(x_signal)
			.interpolate(method="linear", limit_direction="both")
			.to_numpy(dtype=np.float64)
		)
		y_interp = (
			pd.Series(y_signal)
			.interpolate(method="linear", limit_direction="both")
			.to_numpy(dtype=np.float64)
		)

		return x_interp, y_interp

	def smooth_signal(self, signal_1d: NDArray[np.float64]) -> NDArray[np.float64]:
		"""Smooth a 1D coordinate signal using Savitzky-Golay filtering.

		The thesis-optimized filter parameters are fixed for 30 FPS gait data:
		``window_length=11`` and ``polyorder=3``.

		Args:
			signal_1d: Input 1D coordinate signal.

		Returns:
			Smoothed signal as float64 array.

		Raises:
			ValueError: If the signal is empty.
		"""

		if signal_1d.size == 0:
			raise ValueError("signal_1d must not be empty.")

		if signal_1d.shape[0] < self._savgol_window_length:
			# Keep pipeline robust for short trials where 11-sample windows are not feasible.
			return signal_1d.copy()

		return savgol_filter(
			signal_1d,
			window_length=self._savgol_window_length,
			polyorder=self._savgol_polyorder,
			mode="interp",
		).astype(np.float64)

	@staticmethod
	def compute_knee_flexion_angles(
		x_hip: NDArray[np.float64],
		y_hip: NDArray[np.float64],
		x_knee: NDArray[np.float64],
		y_knee: NDArray[np.float64],
		x_ankle: NDArray[np.float64],
		y_ankle: NDArray[np.float64],
	) -> NDArray[np.float64]:
		"""Compute vectorized interior knee flexion angles across time.

		Thesis formula (exact form):
			theta_flexion = abs((atan2(y_a - y_k, x_a - x_k) -
				atan2(y_h - y_k, x_h - x_k)) * 180 / pi)

		Biological normalization:
			If angle > 180, convert to 360 - angle.

		Args:
			x_hip: Hip x-coordinate trajectory.
			y_hip: Hip y-coordinate trajectory.
			x_knee: Knee x-coordinate trajectory.
			y_knee: Knee y-coordinate trajectory.
			x_ankle: Ankle x-coordinate trajectory.
			y_ankle: Ankle y-coordinate trajectory.

		Returns:
			1D float64 array of interior knee flexion angles in degrees.

		Raises:
			ValueError: If input arrays are not all the same length.
		"""

		lengths = {
			x_hip.shape[0],
			y_hip.shape[0],
			x_knee.shape[0],
			y_knee.shape[0],
			x_ankle.shape[0],
			y_ankle.shape[0],
		}
		if len(lengths) != 1:
			raise ValueError("All coordinate arrays must have the same length.")

		theta_flexion = np.abs(
			(
				np.arctan2(y_ankle - y_knee, x_ankle - x_knee)
				- np.arctan2(y_hip - y_knee, x_hip - x_knee)
			)
			* 180.0
			/ np.pi
		)

		normalized = np.where(theta_flexion > 180.0, 360.0 - theta_flexion, theta_flexion)
		return normalized.astype(np.float64)

	def process_leg_signal(self, leg_data: Sequence[LegKinematicsLike]) -> NDArray[np.float64]:
		"""Run the full kinematic pipeline for one leg time series.

		Pipeline:
			extract -> threshold -> interpolate -> smooth -> compute angle.

		Args:
			leg_data: Time-ordered leg observations from Stage 2.

		Returns:
			1D array of knee flexion angles (degrees) over time.
		"""

		hip_x, hip_y = self.extract_joint_signal(leg_data, "hip")
		knee_x, knee_y = self.extract_joint_signal(leg_data, "knee")
		ankle_x, ankle_y = self.extract_joint_signal(leg_data, "ankle")

		hip_x, hip_y = self.interpolate_nan_signal(hip_x, hip_y)
		knee_x, knee_y = self.interpolate_nan_signal(knee_x, knee_y)
		ankle_x, ankle_y = self.interpolate_nan_signal(ankle_x, ankle_y)

		hip_x = self.smooth_signal(hip_x)
		hip_y = self.smooth_signal(hip_y)
		knee_x = self.smooth_signal(knee_x)
		knee_y = self.smooth_signal(knee_y)
		ankle_x = self.smooth_signal(ankle_x)
		ankle_y = self.smooth_signal(ankle_y)

		return self.compute_knee_flexion_angles(
			x_hip=hip_x,
			y_hip=hip_y,
			x_knee=knee_x,
			y_knee=knee_y,
			x_ankle=ankle_x,
			y_ankle=ankle_y,
		)


@dataclass(slots=True, frozen=True)
class _SyntheticJointData:
	"""Internal joint model used only for module smoke testing."""

	x: float
	y: float
	visibility: float


@dataclass(slots=True, frozen=True)
class _SyntheticLegKinematics:
	"""Internal leg model used only for module smoke testing."""

	hip: _SyntheticJointData
	knee: _SyntheticJointData
	ankle: _SyntheticJointData


def _build_synthetic_leg_signal(num_frames: int = 300, fps: float = 30.0) -> list[_SyntheticLegKinematics]:
	"""Create synthetic gait-like leg data for isolated Stage 3 validation.

	The signal approximates periodic sagittal-plane movement and injects
	periodic low-visibility frames to exercise thresholding and interpolation.

	Args:
		num_frames: Number of synthetic samples to generate.
		fps: Sampling frequency in frames per second.

	Returns:
		List of synthetic ``LegKinematics``-compatible observations.
	"""

	t = np.arange(num_frames, dtype=np.float64) / fps
	phase = 2.0 * np.pi * 1.2 * t

	hip_x = 0.50 + 0.02 * np.sin(phase)
	hip_y = 0.40 + 0.01 * np.cos(phase)
	knee_x = 0.53 + 0.03 * np.sin(phase + 0.35)
	knee_y = 0.62 + 0.04 * np.cos(phase + 0.35)
	ankle_x = 0.56 + 0.05 * np.sin(phase + 0.90)
	ankle_y = 0.86 + 0.06 * np.cos(phase + 0.90)

	# Periodically drop confidence below the thesis threshold to test NaN handling.
	visibility = np.where((np.arange(num_frames) % 37) < 3, 0.35, 0.95)

	signal: list[_SyntheticLegKinematics] = []
	for i in range(num_frames):
		signal.append(
			_SyntheticLegKinematics(
				hip=_SyntheticJointData(float(hip_x[i]), float(hip_y[i]), float(visibility[i])),
				knee=_SyntheticJointData(float(knee_x[i]), float(knee_y[i]), float(visibility[i])),
				ankle=_SyntheticJointData(float(ankle_x[i]), float(ankle_y[i]), float(visibility[i])),
			)
		)

	return signal


def _run_kinematics_smoke_test() -> int:
	"""Run an isolated synthetic smoke test for Stage 3 computations."""

	analyzer = KinematicAnalyzer(visibility_threshold=0.5)
	synthetic_leg_data = _build_synthetic_leg_signal(num_frames=300, fps=30.0)
	angles = analyzer.process_leg_signal(synthetic_leg_data)

	print(f"[kinematics] samples: {angles.size}")
	print(f"[kinematics] min angle: {float(np.min(angles)):.3f} deg")
	print(f"[kinematics] max angle: {float(np.max(angles)):.3f} deg")
	return 0


if __name__ == "__main__":
	raise SystemExit(_run_kinematics_smoke_test())
