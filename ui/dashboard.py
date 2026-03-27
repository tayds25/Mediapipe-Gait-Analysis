"""Tkinter dashboard for displaying gait-analysis outputs.

This module is intentionally UI-only and does not perform pose estimation,
kinematics, or classification computations. It receives rendered frames and
diagnostic values from the orchestration layer and presents them in a clinical
dashboard layout.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageTk

if TYPE_CHECKING:
	from src.classification import DiagnosticResult


class GaitAnalysisDashboard:
	"""Display-only dashboard for live gait analysis monitoring."""

	def __init__(self) -> None:
		"""Initialize the Tkinter root window and UI widgets."""

		self.root = tk.Tk()
		self.root.title("Kinematic Gait Analysis System")
		self.root.geometry("1024x768")
		self.root.resizable(True, True)
		self.root.minsize(1024, 768)
		self.root.protocol("WM_DELETE_WINDOW", self._on_close)

		self._is_running: bool = True
		self._photo_ref: ImageTk.PhotoImage | None = None
		self.start_requested: bool = False

		self._build_layout()

	def _build_layout(self) -> None:
		"""Construct and style the dashboard layout."""

		self.root.configure(bg="#f4f6f8")
		style = ttk.Style(self.root)
		style.theme_use("clam")

		style.configure("Dashboard.TFrame", background="#f4f6f8")
		style.configure("Panel.TFrame", background="#ffffff", relief="flat")
		style.configure(
			"Heading.TLabel",
			background="#ffffff",
			foreground="#1f2937",
			font=("Segoe UI", 16, "bold"),
		)
		style.configure(
			"MetricName.TLabel",
			background="#ffffff",
			foreground="#4b5563",
			font=("Segoe UI", 11, "normal"),
		)
		style.configure(
			"MetricValue.TLabel",
			background="#ffffff",
			foreground="#111827",
			font=("Segoe UI", 13, "bold"),
		)

		container = ttk.Frame(self.root, style="Dashboard.TFrame", padding=12)
		container.pack(fill=tk.BOTH, expand=True)

		left_frame = ttk.Frame(container, style="Panel.TFrame", padding=10)
		left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		left_frame.pack_propagate(False)

		right_frame = ttk.Frame(container, style="Panel.TFrame", width=300, padding=16)
		right_frame.pack(side=tk.RIGHT, fill=tk.Y)
		right_frame.pack_propagate(False)

		self.video_label = tk.Label(
			left_frame,
			bg="#111827",
			fg="#e5e7eb",
			text="Video stream will appear here",
			font=("Segoe UI", 14, "normal"),
		)
		self.video_label.pack(fill=tk.BOTH, expand=True)

		ttk.Label(right_frame, text="Clinical Metrics", style="Heading.TLabel").pack(anchor="w", pady=(0, 18))

		self.status_value_label = self._add_metric_row(right_frame, "System Status", "Buffering...")
		self.symmetry_value_label = self._add_metric_row(right_frame, "Symmetry Index", "N/A")
		self.left_peak_value_label = self._add_metric_row(right_frame, "Left Knee Peak Flexion", "N/A")
		self.right_peak_value_label = self._add_metric_row(right_frame, "Right Knee Peak Flexion", "N/A")

		self.buffer_label = ttk.Label(right_frame, text="Buffer: 0 frames", style="MetricName.TLabel")
		self.buffer_label.pack(anchor="w", pady=(20, 0))

		self.recording_indicator = ttk.Label(right_frame, text="Waiting...", style="MetricName.TLabel")
		self.recording_indicator.pack(anchor="w", pady=(16, 0))

		self.start_button = ttk.Button(
			right_frame,
			text="Start Trial",
			command=self._on_start_clicked,
		)
		self.start_button.pack(fill=tk.X, pady=(16, 0))

	def _on_start_clicked(self) -> None:
		"""Handle the Start Trial button click."""
		self.start_requested = True
		self.start_button.config(state="disabled")

	def _add_metric_row(self, parent: ttk.Frame, name: str, default_value: str) -> ttk.Label:
		"""Create a named metric row and return its value label."""

		row = ttk.Frame(parent, style="Panel.TFrame")
		row.pack(fill=tk.X, pady=(0, 12))

		ttk.Label(row, text=name, style="MetricName.TLabel").pack(anchor="w")
		value_label = ttk.Label(row, text=default_value, style="MetricValue.TLabel")
		value_label.pack(anchor="w", pady=(2, 0))
		return value_label

	def _frame_to_photoimage(self, frame_bgr: np.ndarray) -> ImageTk.PhotoImage:
		"""Convert BGR OpenCV frame into Tk-compatible PhotoImage."""

		if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
			raise ValueError("frame_bgr must have shape (H, W, 3).")

		target_w = max(self.video_label.master.winfo_width(), 1)
		target_h = max(self.video_label.master.winfo_height(), 1)

		frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
		pil_image = Image.fromarray(resized)
		return ImageTk.PhotoImage(image=pil_image)

	def update_display(
		self,
		frame_bgr: np.ndarray,
		diagnostic_result: DiagnosticResult | None,
		buffer_count: int,
		recording_state: str = "IDLE",
	) -> None:
		"""Update video pane and clinical metric labels.

		Args:
			frame_bgr: Current OpenCV frame in BGR format.
			diagnostic_result: Latest diagnostic output, or None while buffering.
			buffer_count: Number of valid pose frames currently buffered.
			recording_state: Current state string ("IDLE", "COUNTDOWN", "RECORDING").

		Raises:
			ValueError: If frame_bgr is invalid.
		"""

		if frame_bgr.size == 0:
			raise ValueError("frame_bgr cannot be empty.")

		self.root.update_idletasks()
		photo = self._frame_to_photoimage(frame_bgr)
		self._photo_ref = photo
		self.video_label.configure(image=photo, text="")

		if diagnostic_result is None:
			self.status_value_label.configure(text="Buffering...", foreground="#b45309")
			self.symmetry_value_label.configure(text="N/A")
			self.left_peak_value_label.configure(text="N/A")
			self.right_peak_value_label.configure(text="N/A")
		else:
			if diagnostic_result.is_abnormal:
				self.status_value_label.configure(text="Abnormal", foreground="#b91c1c")
			else:
				self.status_value_label.configure(text="Normal", foreground="#15803d")

			self.symmetry_value_label.configure(text=f"{diagnostic_result.symmetry_index:.2f}%")
			self.left_peak_value_label.configure(text=f"{diagnostic_result.peak_flexion_left:.2f} deg")
			self.right_peak_value_label.configure(text=f"{diagnostic_result.peak_flexion_right:.2f} deg")

		self.buffer_label.configure(text=f"Buffer: {buffer_count} frames")

		if recording_state == "IDLE":
			self.recording_indicator.configure(text="Waiting...", foreground="#6b7280")
			self.start_button.config(state="normal")
		elif recording_state == "COUNTDOWN":
			self.recording_indicator.configure(text="Starting...", foreground="#b45309")
		elif recording_state == "RECORDING":
			self.recording_indicator.configure(text="🔴 RECORDING", foreground="#b91c1c")

		if self._is_running:
			self.root.update_idletasks()
			self.root.update()

	def _on_close(self) -> None:
		"""Safely terminate the dashboard window."""

		if not self._is_running:
			return

		self._is_running = False
		self.root.quit()
		self.root.destroy()

	def start(self) -> None:
		"""Start the Tkinter event loop."""

		if not self._is_running:
			return
		self.root.mainloop()


def _run_dashboard_smoke_test() -> int:
	"""Run a basic dashboard smoke test with a synthetic black frame."""

	dashboard = GaitAnalysisDashboard()
	blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
	dashboard.update_display(frame_bgr=blank_frame, diagnostic_result=None, buffer_count=0)
	dashboard.start()
	return 0


if __name__ == "__main__":
	raise SystemExit(_run_dashboard_smoke_test())
