"""Stage 4: Symmetry Classification Module.

This module converts bilateral knee-angle trajectories into a clinical
diagnostic decision using Robinson's Symmetry Index (SI). The implementation
follows the thesis constraints exactly, including absolute-magnitude handling
and a 10% abnormality threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class DiagnosticResult:
	"""Final bilateral gait asymmetry result.

	Attributes:
		peak_flexion_left: Left-leg peak flexion magnitude in degrees.
		peak_flexion_right: Right-leg peak flexion magnitude in degrees.
		symmetry_index: Robinson's Symmetry Index as percentage.
		is_abnormal: True when SI is at or above the 10% clinical cutoff.
	"""

	peak_flexion_left: float
	peak_flexion_right: float
	symmetry_index: float
	is_abnormal: bool


class GaitClassifier:
	"""Evaluate bilateral knee-angle symmetry using Robinson's SI."""

	def __init__(self, abnormal_threshold: float = 10.0) -> None:
		"""Initialize classifier with thesis clinical threshold.

		Args:
			abnormal_threshold: SI threshold (%) for abnormal classification.
		"""

		self._abnormal_threshold: float = abnormal_threshold

	@staticmethod
	def _validate_angle_array(name: str, angles: NDArray[np.float64]) -> None:
		"""Validate shape and finiteness of a knee-angle trajectory.

		Args:
			name: Logical signal name for error reporting.
			angles: 1D knee-angle array.

		Raises:
			ValueError: If the array is empty, non-1D, or contains non-finite values.
		"""

		if angles.ndim != 1:
			raise ValueError(f"{name} must be a 1D array.")
		if angles.size == 0:
			raise ValueError(f"{name} must not be empty.")
		if not np.isfinite(angles).all():
			raise ValueError(f"{name} must contain only finite values.")

	@staticmethod
	def _extract_peak_flexion(angles: NDArray[np.float64]) -> float:
		"""Extract peak flexion magnitude from interior knee-angle trajectory.

		Thesis definition:
			X = 180.0 - min(angle_array)

		Because interior angle decreases with flexion, the minimum interior angle
		corresponds to maximum bend magnitude from a straight leg.

		Args:
			angles: 1D interior knee-angle trajectory in degrees.

		Returns:
			Peak flexion magnitude in degrees.
		"""

		return float(180.0 - np.min(angles))

	@staticmethod
	def _compute_symmetry_index(x_right: float, x_left: float) -> float:
		"""Compute Robinson's Symmetry Index (SI) with zero-division safety.

		Thesis formula (strict):
			SI = (abs(X_R - X_L) / (0.5 * (abs(X_R) + abs(X_L)))) * 100

		Safety rule:
			If both peak flexions are exactly zero, SI is defined as 0.0.

		Args:
			x_right: Right-leg peak flexion magnitude.
			x_left: Left-leg peak flexion magnitude.

		Returns:
			Symmetry Index percentage.
		"""

		denominator = 0.5 * (np.abs(x_right) + np.abs(x_left))
		if denominator == 0.0:
			return 0.0

		si = (np.abs(x_right - x_left) / denominator) * 100.0
		return float(si)

	def evaluate_symmetry(
		self,
		left_angles: NDArray[np.float64],
		right_angles: NDArray[np.float64],
	) -> DiagnosticResult:
		"""Compute bilateral asymmetry and return final diagnostic decision.

		Args:
			left_angles: Left-leg 1D interior knee-angle trajectory.
			right_angles: Right-leg 1D interior knee-angle trajectory.

		Returns:
			``DiagnosticResult`` containing peak flexions, SI, and diagnosis.

		Raises:
			ValueError: If either input array is invalid.
		"""

		self._validate_angle_array("left_angles", left_angles)
		self._validate_angle_array("right_angles", right_angles)

		x_left = self._extract_peak_flexion(left_angles)
		x_right = self._extract_peak_flexion(right_angles)
		si = self._compute_symmetry_index(x_right=x_right, x_left=x_left)
		is_abnormal = si >= self._abnormal_threshold

		return DiagnosticResult(
			peak_flexion_left=x_left,
			peak_flexion_right=x_right,
			symmetry_index=si,
			is_abnormal=is_abnormal,
		)


def _run_classification_smoke_test() -> int:
	"""Run a synthetic severe-limp smoke test for Stage 4.

	Left leg bends to 120 degrees (peak flexion 60), while right leg only bends
	to 150 degrees (peak flexion 30). This should exceed the 10% SI threshold
	and be classified as abnormal.
	"""

	left_angles = np.array([180.0, 170.0, 150.0, 130.0, 120.0, 130.0, 150.0, 170.0], dtype=np.float64)
	right_angles = np.array([180.0, 176.0, 165.0, 156.0, 150.0, 156.0, 165.0, 176.0], dtype=np.float64)

	classifier = GaitClassifier(abnormal_threshold=10.0)
	result = classifier.evaluate_symmetry(left_angles=left_angles, right_angles=right_angles)

	print(f"[classification] Symmetry Index: {result.symmetry_index:.3f}%")
	print(f"[classification] Is Abnormal: {result.is_abnormal}")
	return 0


if __name__ == "__main__":
	raise SystemExit(_run_classification_smoke_test())
