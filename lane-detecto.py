"""
Road Lane Violation Detection System
=====================================
Uses Canny edge detection, Hough Transform, and classical vision techniques
to detect road lanes and identify lane violations from dashcam footage.

Usage:
    python lane-detecto.py [--input VIDEO_PATH] [--output OUTPUT_PATH]
"""

import cv2
import numpy as np
import argparse
import os
import sys
from collections import deque


# ─────────────────────── Configuration ───────────────────────
class Config:
    """All tunable parameters in one place."""

    # Canny edge detection thresholds
    CANNY_LOW = 50
    CANNY_HIGH = 150

    # Gaussian blur kernel size (must be odd)
    BLUR_KERNEL = (5, 5)

    # Hough Transform parameters
    HOUGH_RHO = 2               # Distance resolution in pixels
    HOUGH_THETA = np.pi / 180   # Angular resolution in radians
    HOUGH_THRESHOLD = 50        # Min votes to consider a line
    HOUGH_MIN_LINE_LEN = 40     # Min length of line segment (px)
    HOUGH_MAX_LINE_GAP = 150    # Max gap between segments to join (px)

    # Lane line slope filters (absolute value)
    MIN_SLOPE = 0.5
    MAX_SLOPE = 2.0

    # Smoothing: average lane positions over last N frames
    SMOOTHING_FRAMES = 25

    # Lane violation tolerance (fraction of lane width)
    # Vehicle center must be within TOLERANCE * lane_width of lane center
    VIOLATION_TOLERANCE_PX = 30  # pixels from lane center

    # ROI polygon as fractions of (width, height)
    # Bottom-left, Top-left, Top-right, Bottom-right
    ROI_VERTICES_FRAC = [
        (0.02, 1.0),   # bottom-left
        (0.42, 0.40),  # top-left
        (0.58, 0.40),  # top-right
        (0.98, 1.0),   # bottom-right
    ]

    # How far up the frame to draw lane lines (fraction of height)
    LANE_TOP_FRAC = 0.55

    # Slow-down factor: output FPS = input FPS / SLOW_FACTOR
    SLOW_FACTOR = 2

    # Road frame detection thresholds
    ROAD_GRAY_MIN = 60       # Min gray value for "asphalt" pixels
    ROAD_GRAY_MAX = 200      # Max gray value for "asphalt" pixels
    ROAD_SAT_MAX = 60        # Max saturation for "road-like" pixels
    ROAD_GRAY_FRAC = 0.20    # Fraction of bottom-half pixels that must be gray/road
    ROAD_MIN_EDGES = 200     # Min edge pixels in ROI to qualify as road frame
    ROAD_MIN_LINES = 3       # Min Hough lines in ROI to qualify

    # Colors (BGR)
    COLOR_LANE = (0, 255, 0)         # Green lane lines
    COLOR_LANE_FILL = (0, 255, 0)    # Lane area fill
    COLOR_LANE_OUTLINE = (0, 200, 0) # Lane shape outline
    COLOR_VEHICLE_CENTER = (255, 0, 0)  # Blue
    COLOR_LANE_CENTER = (0, 255, 255)   # Yellow
    COLOR_SAFE = (0, 255, 0)            # Green text
    COLOR_VIOLATION = (0, 0, 255)       # Red text
    COLOR_ROI = (255, 255, 0)           # Cyan ROI outline

    # Font
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.9
    FONT_THICKNESS = 2


# ─────────────────────── Helper Functions ───────────────────────

def get_roi_mask(image, vertices):
    """Create a binary mask for the Region of Interest."""
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        fill_color = (255,) * image.shape[2]
    else:
        fill_color = 255
    cv2.fillPoly(mask, vertices, fill_color)
    return cv2.bitwise_and(image, mask)


def get_roi_vertices(width, height, fracs):
    """Convert fractional ROI coordinates to pixel coordinates."""
    vertices = np.array(
        [(int(fx * width), int(fy * height)) for fx, fy in fracs],
        dtype=np.int32,
    )
    return vertices.reshape(1, -1, 2)


def canny_edge_detection(frame, config):
    """Apply Canny edge detection with preprocessing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, config.BLUR_KERNEL, 0)
    edges = cv2.Canny(blur, config.CANNY_LOW, config.CANNY_HIGH)
    return edges


def hough_lines(edges, config):
    """Detect line segments using Probabilistic Hough Transform."""
    lines = cv2.HoughLinesP(
        edges,
        rho=config.HOUGH_RHO,
        theta=config.HOUGH_THETA,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LINE_LEN,
        maxLineGap=config.HOUGH_MAX_LINE_GAP,
    )
    return lines


def is_road_frame(frame, config, roi_vertices):
    """
    Heuristic classifier to detect whether a frame shows road/highway footage.
    Checks:
      1. Bottom half has enough gray/asphalt-like pixels (low sat, medium brightness)
      2. ROI region has enough Canny edges (lane markings)
      3. ROI region has enough Hough lines with road-like slopes
    Returns True if the frame is likely a road scene.
    """
    h, w = frame.shape[:2]

    # ── Check 1: Gray/asphalt dominance in bottom half ──
    bottom_half = frame[h // 2:, :]
    hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
    gray_bottom = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)

    # Road pixels: low saturation AND medium brightness
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    road_mask = (sat < config.ROAD_SAT_MAX) & \
                (gray_bottom >= config.ROAD_GRAY_MIN) & \
                (gray_bottom <= config.ROAD_GRAY_MAX)
    road_frac = np.count_nonzero(road_mask) / road_mask.size

    if road_frac < config.ROAD_GRAY_FRAC:
        return False

    # ── Check 2: Enough Canny edges in the ROI ──
    edges = canny_edge_detection(frame, config)
    masked_edges = get_roi_mask(edges, roi_vertices)
    edge_count = np.count_nonzero(masked_edges)

    if edge_count < config.ROAD_MIN_EDGES:
        return False

    # ── Check 3: Enough Hough lines with road-like slopes in ROI ──
    lines = hough_lines(masked_edges, config)
    if lines is None:
        return False

    road_line_count = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = abs((y2 - y1) / (x2 - x1))
        if config.MIN_SLOPE <= slope <= config.MAX_SLOPE:
            road_line_count += 1

    if road_line_count < config.ROAD_MIN_LINES:
        return False

    return True


def separate_lanes(lines, width, config):
    """
    Separate detected lines into left and right lanes based on slope.
    Left lane: negative slope (top-left to bottom-right in image coords)
    Right lane: positive slope
    Returns averaged (slope, intercept) for each lane, or None.
    """
    left_fits = []
    right_fits = []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Filter by slope magnitude
        if abs(slope) < config.MIN_SLOPE or abs(slope) > config.MAX_SLOPE:
            continue

        # Classify by slope sign AND position
        mid_x = (x1 + x2) / 2
        if slope < 0 and mid_x < width * 0.5:
            left_fits.append((slope, intercept))
        elif slope > 0 and mid_x > width * 0.5:
            right_fits.append((slope, intercept))

    left_avg = np.mean(left_fits, axis=0) if left_fits else None
    right_avg = np.mean(right_fits, axis=0) if right_fits else None

    return left_avg, right_avg


def lane_endpoints(slope_intercept, y_bottom, y_top):
    """
    Given (slope, intercept), compute (x1, y1, x2, y2)
    from y_bottom (near car) to y_top (horizon).
    """
    if slope_intercept is None:
        return None
    slope, intercept = slope_intercept
    if abs(slope) < 1e-6:
        return None
    x_bottom = int((y_bottom - intercept) / slope)
    x_top = int((y_top - intercept) / slope)
    return (x_bottom, int(y_bottom), x_top, int(y_top))


def compute_lane_center(left_line, right_line):
    """
    Compute the lane center x position at the bottom of the frame
    using the bottom x-coordinates of the left and right lane lines.
    """
    if left_line is None or right_line is None:
        return None
    left_x_bottom = left_line[0]
    right_x_bottom = right_line[0]
    return (left_x_bottom + right_x_bottom) // 2


def detect_violation(vehicle_center_x, lane_center_x, tolerance):
    """
    Detect if vehicle has crossed lane boundaries.
    Returns (is_violation, offset_pixels).
    """
    if lane_center_x is None:
        return False, 0
    offset = vehicle_center_x - lane_center_x
    is_violation = abs(offset) > tolerance
    return is_violation, offset


# ─────────────────────── Drawing Functions ───────────────────────

def _line_intersection(p1, p2, p3, p4):
    """
    Find intersection point of line segment (p1->p2) and (p3->p4).
    Returns (x, y) or None if lines are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (int(ix), int(iy))


def draw_lane_lines(frame, left_line, right_line, config):
    """Draw lane boundary lines and filled lane polygon on the frame."""
    overlay = frame.copy()
    h = frame.shape[0]

    if left_line is not None and right_line is not None:
        # Find where left and right lane lines intersect (vanishing point)
        left_bot = (left_line[0], left_line[1])
        left_top = (left_line[2], left_line[3])
        right_bot = (right_line[0], right_line[1])
        right_top = (right_line[2], right_line[3])

        vanish = _line_intersection(left_bot, left_top, right_bot, right_top)

        # Determine the top of the polygon:
        # If lines intersect within the frame, clip the polygon at the
        # intersection so we don't get an inverted triangle above it.
        if vanish is not None and 0 < vanish[1] < h:
            # Use vanishing point as the apex — triangle from bottom to apex
            clip_y = vanish[1]
            # Recompute x at a y slightly below the vanishing point so polygon
            # stays below the VP
            margin = 10  # pixels below vanishing point
            poly_top_y = clip_y + margin

            # Left x at poly_top_y
            l_slope = (left_top[1] - left_bot[1])
            if abs(left_top[0] - left_bot[0]) > 0:
                l_m = l_slope / (left_top[0] - left_bot[0])
            else:
                l_m = 1e6
            if abs(l_m) > 1e-6:
                l_x_at_top = left_bot[0] + (poly_top_y - left_bot[1]) * (left_top[0] - left_bot[0]) / l_slope
            else:
                l_x_at_top = left_top[0]

            r_slope = (right_top[1] - right_bot[1])
            if abs(right_top[0] - right_bot[0]) > 0:
                r_m = r_slope / (right_top[0] - right_bot[0])
            else:
                r_m = 1e6
            if abs(r_m) > 1e-6:
                r_x_at_top = right_bot[0] + (poly_top_y - right_bot[1]) * (right_top[0] - right_bot[0]) / r_slope
            else:
                r_x_at_top = right_top[0]

            pts = np.array([
                [left_bot[0], left_bot[1]],         # left bottom
                [int(l_x_at_top), int(poly_top_y)], # left near vanishing pt
                [int(r_x_at_top), int(poly_top_y)], # right near vanishing pt
                [right_bot[0], right_bot[1]],        # right bottom
            ], dtype=np.int32)
        else:
            # Lines don't cross in frame — standard trapezoid
            pts = np.array([
                [left_bot[0], left_bot[1]],
                [left_top[0], left_top[1]],
                [right_top[0], right_top[1]],
                [right_bot[0], right_bot[1]],
            ], dtype=np.int32)

        # Semi-transparent green fill — only between the lane lines
        cv2.fillPoly(overlay, [pts], config.COLOR_LANE_FILL)

        # Draw the outline of the lane polygon
        cv2.polylines(overlay, [pts], isClosed=True,
                      color=config.COLOR_LANE_OUTLINE, thickness=3)

        # ── Draw thick lane boundary lines (clipped to polygon) ──
        left_top_clipped = (int(pts[1][0]), int(pts[1][1]))
        right_top_clipped = (int(pts[2][0]), int(pts[2][1]))
        cv2.line(overlay, left_bot, left_top_clipped, (0, 255, 255), 5)   # Yellow left
        cv2.line(overlay, right_bot, right_top_clipped, (255, 255, 255), 5)  # White right

        # ── Draw dashed center lane marker ──
        center_x_bottom = (left_bot[0] + right_bot[0]) // 2
        center_x_top = (pts[1][0] + pts[2][0]) // 2
        y_bot_d = left_bot[1]
        y_tp_d = int(pts[1][1])
        num_dashes = 12
        for i in range(num_dashes):
            if i % 2 == 0:
                t1 = i / num_dashes
                t2 = (i + 1) / num_dashes
                x1 = int(center_x_bottom + (center_x_top - center_x_bottom) * t1)
                y1 = int(y_bot_d + (y_tp_d - y_bot_d) * t1)
                x2 = int(center_x_bottom + (center_x_top - center_x_bottom) * t2)
                y2 = int(y_bot_d + (y_tp_d - y_bot_d) * t2)
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)

    elif left_line is not None:
        cv2.line(overlay, (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]), (0, 255, 255), 5)

    elif right_line is not None:
        cv2.line(overlay, (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]), (255, 255, 255), 5)

    # Blend with original (semi-transparent)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    return frame


def draw_center_markers(frame, vehicle_cx, lane_cx, height, config):
    """Draw vehicle center and lane center markers."""
    # Vehicle center (vertical line at bottom)
    cv2.line(frame, (vehicle_cx, height), (vehicle_cx, height - 50),
             config.COLOR_VEHICLE_CENTER, 3)
    cv2.putText(frame, "Vehicle", (vehicle_cx - 30, height - 55),
                config.FONT, 0.5, config.COLOR_VEHICLE_CENTER, 1)

    if lane_cx is not None:
        cv2.line(frame, (lane_cx, height), (lane_cx, height - 50),
                 config.COLOR_LANE_CENTER, 3)
        cv2.putText(frame, "Lane Center", (lane_cx - 40, height - 55),
                    config.FONT, 0.5, config.COLOR_LANE_CENTER, 1)

    return frame


def draw_status(frame, is_violation, offset, frame_num, fps, config):
    """Draw violation status and info on the frame."""
    if is_violation:
        status_text = "LANE VIOLATION!"
        color = config.COLOR_VIOLATION
        direction = "LEFT" if offset < 0 else "RIGHT"
        detail = f"Drifting {direction} by {abs(offset):.0f}px"
    else:
        status_text = "IN LANE"
        color = config.COLOR_SAFE
        detail = f"Offset: {abs(offset):.0f}px"

    # Background rectangle for better readability
    cv2.rectangle(frame, (10, 10), (400, 110), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (400, 110), color, 2)

    cv2.putText(frame, status_text, (20, 45),
                config.FONT, config.FONT_SCALE, color, config.FONT_THICKNESS)
    cv2.putText(frame, detail, (20, 75),
                config.FONT, 0.6, (255, 255, 255), 1)

    timestamp = frame_num / fps if fps > 0 else 0
    time_text = f"Time: {int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
    cv2.putText(frame, time_text, (20, 100),
                config.FONT, 0.5, (200, 200, 200), 1)

    return frame


def draw_edges_debug(frame, edges, roi_vertices):
    """Draw a small debug view of edges in the corner."""
    h, w = frame.shape[:2]
    small_h, small_w = h // 4, w // 4

    # Convert edges to BGR so we can overlay
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_small = cv2.resize(edges_bgr, (small_w, small_h))

    # Place in top-right corner
    frame[10:10 + small_h, w - small_w - 10:w - 10] = edges_small
    cv2.rectangle(frame, (w - small_w - 10, 10),
                  (w - 10, 10 + small_h), (255, 255, 255), 1)
    cv2.putText(frame, "Edge Detection", (w - small_w - 5, 10 + small_h + 18),
                Config.FONT, 0.45, (255, 255, 255), 1)
    return frame


# ─────────────────────── Smoothing Class ───────────────────────

class LaneSmoother:
    """Smooth lane line parameters over multiple frames to reduce jitter."""

    def __init__(self, max_frames=10):
        self.left_history = deque(maxlen=max_frames)
        self.right_history = deque(maxlen=max_frames)

    def update(self, left_fit, right_fit):
        if left_fit is not None:
            self.left_history.append(left_fit)
        if right_fit is not None:
            self.right_history.append(right_fit)

    def get_smoothed(self):
        left_avg = None
        right_avg = None
        if self.left_history:
            left_avg = np.mean(self.left_history, axis=0)
        if self.right_history:
            right_avg = np.mean(self.right_history, axis=0)
        return left_avg, right_avg


# ─────────────────────── Main Pipeline ───────────────────────

def process_video(input_path, output_path, config, save_frames=False):
    """Main video processing pipeline."""

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        sys.exit(1)

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Slowed-down output FPS
    output_fps = fps / config.SLOW_FACTOR

    print(f"Input video: {input_path}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames / fps:.1f} seconds")
    print(f"Output FPS: {output_fps:.1f} (slow factor: {config.SLOW_FACTOR}x)")
    print(f"Output: {output_path}")
    print("-" * 50)

    # Create output frames directory if needed
    frames_dir = "output_frames"
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving individual frames to: {frames_dir}/")

    # Video writer with slowed-down FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    # ROI vertices
    roi_vertices = get_roi_vertices(width, height, config.ROI_VERTICES_FRAC)

    # Lane smoother
    smoother = LaneSmoother(max_frames=config.SMOOTHING_FRAMES)

    # Vehicle center (camera assumed at center)
    vehicle_center_x = width // 2

    # Stats
    frame_num = 0
    processed_frames = 0
    violation_count = 0
    violation_frames = 0
    no_lane_frames = 0
    skipped_frames = 0
    non_road_frames = 0

    # Y coordinates for lane line drawing
    y_bottom = height
    y_top = int(height * config.LANE_TOP_FRAC)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # ── Valid frame filtering ──
        if frame is None or frame.size == 0:
            skipped_frames += 1
            continue
        if frame.shape[0] != height or frame.shape[1] != width:
            skipped_frames += 1
            continue

        # ── Road frame classifier ──
        if not is_road_frame(frame, config, roi_vertices):
            non_road_frames += 1
            continue

        processed_frames += 1

        # ── Step 1: Canny Edge Detection ──
        edges = canny_edge_detection(frame, config)

        # ── Step 2: Apply ROI Mask ──
        masked_edges = get_roi_mask(edges, roi_vertices)

        # ── Step 3: Hough Transform ──
        lines = hough_lines(masked_edges, config)

        # ── Step 4: Separate left and right lanes ──
        left_fit, right_fit = separate_lanes(lines, width, config)

        # ── Step 5: Smooth lane positions ──
        smoother.update(left_fit, right_fit)
        left_smooth, right_smooth = smoother.get_smoothed()

        # ── Step 6: Compute lane endpoints ──
        left_line = lane_endpoints(left_smooth, y_bottom, y_top)
        right_line = lane_endpoints(right_smooth, y_bottom, y_top)

        # ── Step 7: Compute lane center ──
        lane_center_x = compute_lane_center(left_line, right_line)

        # ── Step 8: Detect violation ──
        is_violation = False
        offset = 0
        if lane_center_x is not None:
            is_violation, offset = detect_violation(
                vehicle_center_x, lane_center_x, config.VIOLATION_TOLERANCE_PX
            )
            if is_violation:
                violation_frames += 1
        else:
            no_lane_frames += 1

        # ── Step 9: Draw annotations ──
        annotated = frame.copy()

        # Draw lane lines and filled lane area
        annotated = draw_lane_lines(annotated, left_line, right_line, config)

        # Draw center markers
        annotated = draw_center_markers(
            annotated, vehicle_center_x, lane_center_x, height, config
        )

        # Draw status overlay
        annotated = draw_status(
            annotated, is_violation, offset, frame_num, fps, config
        )

        # Draw edge detection debug view
        annotated = draw_edges_debug(annotated, masked_edges, roi_vertices)

        # Write frame
        out.write(annotated)

        # Save individual frame image
        if save_frames:
            frame_path = os.path.join(frames_dir, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(frame_path, annotated)

        # Progress reporting
        if frame_num % 100 == 0 or frame_num == total_frames:
            pct = (frame_num / total_frames) * 100 if total_frames > 0 else 0
            print(f"  Frame {frame_num}/{total_frames} ({pct:.1f}%) "
                  f"| Road: {processed_frames} | Non-road: {non_road_frames} "
                  f"| Violations: {violation_frames}")

    # Cleanup
    cap.release()
    out.release()

    # ── Summary Report ──
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Total frames read: {frame_num}")
    print(f"Road frames processed: {processed_frames}")
    print(f"Non-road frames skipped: {non_road_frames}")
    print(f"Invalid frames skipped: {skipped_frames}")
    if processed_frames > 0:
        print(f"Frames with lane violation: {violation_frames} "
              f"({violation_frames / processed_frames * 100:.1f}%)")
        print(f"Frames with no lane detected: {no_lane_frames} "
              f"({no_lane_frames / processed_frames * 100:.1f}%)")
        estimated_duration = processed_frames / output_fps
        print(f"Output video duration: {estimated_duration:.1f} seconds "
              f"({config.SLOW_FACTOR}x slower)")
    print(f"Output video saved to: {output_path}")
    if os.path.exists(output_path):
        print(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.1f} MB")
    if save_frames:
        print(f"Individual frames saved to: {frames_dir}/ ({processed_frames} images)")

    return {
        "total_frames": frame_num,
        "road_frames": processed_frames,
        "non_road_frames": non_road_frames,
        "violation_frames": violation_frames,
        "no_lane_frames": no_lane_frames,
        "fps": fps,
        "output_fps": output_fps,
        "duration": frame_num / fps if fps > 0 else 0,
    }


# ─────────────────────── Entry Point ───────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Road Lane Violation Detection System"
    )
    parser.add_argument(
        "--input", "-i",
        default="video.mp4",
        help="Path to input video file (default: video.mp4)",
    )
    parser.add_argument(
        "--output", "-o",
        default="output_video.mp4",
        help="Path to output video file (default: output_video.mp4)",
    )
    parser.add_argument(
        "--save-frames", "-s",
        action="store_true",
        default=False,
        help="Save each annotated frame as a JPEG image in output_frames/",
    )
    parser.add_argument(
        "--slow", type=int, default=2,
        help="Slow-down factor for output video (default: 2 = half speed)",
    )
    args = parser.parse_args()

    config = Config()
    config.SLOW_FACTOR = args.slow
    results = process_video(args.input, args.output, config, save_frames=args.save_frames)
