from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class GestureResult:
    label: str
    confidence: float
    kind: str = "letter"
    pose: str = ""
    debug: str = ""


class RuleBasedASLRecognizer:
    """Starter static ASL fingerspelling recognizer from MediaPipe landmarks.

    This is useful for prototyping the sign-to-visual bridge without a trained
    model. It recognizes a practical subset of static ASL handshapes and a few
    two-hand commands. Full sign language recognition should replace this with
    a trained temporal model.
    """

    def recognize_single(self, landmarks: Iterable[object], handedness: str = "Right") -> GestureResult | None:
        pts = [as_point(point) for point in landmarks]
        if len(pts) < 21:
            return None

        features = HandFeatures(pts, handedness)
        label, confidence, debug = self._classify_letter(features)
        pose = self._classify_pose(features)
        if label is None:
            return GestureResult("UNKNOWN", 0.0, pose=pose, debug=debug)
        return GestureResult(label, confidence, pose=pose, debug=debug)

    def recognize_command(self, results: Sequence[GestureResult]) -> GestureResult | None:
        confident = [result for result in results if result.confidence >= 0.55]
        if len(confident) < 2:
            return None

        poses = {result.pose for result in confident}
        if poses == {"OPEN_PALM"}:
            return GestureResult("SEND", 0.92, kind="command", pose="TWO_OPEN_PALMS", debug="two open palms")
        if poses == {"CLOSED_FIST"}:
            return GestureResult("CLEAR", 0.9, kind="command", pose="TWO_CLOSED_FISTS", debug="two closed fists")
        if "OPEN_PALM" in poses and "CLOSED_FIST" in poses:
            return GestureResult("SPACE", 0.86, kind="command", pose="OPEN_AND_FIST", debug="open palm plus fist")
        return None

    def _classify_letter(self, f: "HandFeatures") -> tuple[str | None, float, str]:
        index = f.extended["index"]
        middle = f.extended["middle"]
        ring = f.extended["ring"]
        pinky = f.extended["pinky"]
        thumb = f.extended["thumb"]
        four = [index, middle, ring, pinky]
        extended_count = sum(1 for item in four if item)

        thumb_index = f.distance(4, 8)
        index_middle = f.distance(8, 12)

        if thumb and index and not middle and not ring and not pinky:
            if thumb_index > 0.85:
                return "L", 0.86, "thumb and index extended"

        if index and not middle and not ring and not pinky:
            return "D", 0.82, "index extended"

        if pinky and not index and not middle and not ring:
            if thumb:
                return "Y", 0.84, "thumb and pinky extended"
            return "I", 0.8, "pinky extended"

        if index and middle and not ring and not pinky:
            if index_middle < 0.35:
                return "U", 0.78, "index and middle together"
            return "V", 0.82, "index and middle spread"

        if index and middle and ring and not pinky:
            return "W", 0.82, "three fingers extended"

        if middle and ring and pinky and not index:
            if thumb_index < 0.55:
                return "F", 0.82, "index touches thumb, other fingers extended"

        if all(four):
            return "B", 0.78 if thumb else 0.84, "four fingers extended"

        if extended_count == 0:
            if thumb_index < 0.5 and f.finger_angle["index"] > 75:
                return "O", 0.78, "rounded fingertips"
            if 0.5 <= thumb_index <= 1.2 and f.finger_angle["index"] > 90:
                return "C", 0.72, "curved hand"
            if thumb:
                return "A", 0.76, "fist with thumb visible"
            return "E", 0.7, "closed hand"

        return None, 0.0, "no static rule matched"

    def _classify_pose(self, f: "HandFeatures") -> str:
        four = [f.extended["index"], f.extended["middle"], f.extended["ring"], f.extended["pinky"]]
        if all(four):
            return "OPEN_PALM"
        if not any(four) and not f.extended["thumb"]:
            return "CLOSED_FIST"
        return "PARTIAL"


class HandFeatures:
    def __init__(self, pts: Sequence[tuple[float, float, float]], handedness: str):
        self.pts = pts
        self.handedness = handedness
        self.palm_size = max(distance(pts[0], pts[9]), 1e-6)
        self.finger_angle = {
            "thumb": angle(pts[2], pts[3], pts[4]),
            "index": angle(pts[5], pts[6], pts[8]),
            "middle": angle(pts[9], pts[10], pts[12]),
            "ring": angle(pts[13], pts[14], pts[16]),
            "pinky": angle(pts[17], pts[18], pts[20]),
        }
        self.extended = {
            "thumb": self._thumb_extended(),
            "index": self._finger_extended(5, 6, 8, "index"),
            "middle": self._finger_extended(9, 10, 12, "middle"),
            "ring": self._finger_extended(13, 14, 16, "ring"),
            "pinky": self._finger_extended(17, 18, 20, "pinky"),
        }

    def distance(self, first: int, second: int) -> float:
        return distance(self.pts[first], self.pts[second]) / self.palm_size

    def _finger_extended(self, mcp: int, pip: int, tip: int, name: str) -> bool:
        wrist_to_tip = distance(self.pts[0], self.pts[tip])
        wrist_to_pip = distance(self.pts[0], self.pts[pip])
        return self.finger_angle[name] > 145 and wrist_to_tip > wrist_to_pip * 1.05

    def _thumb_extended(self) -> bool:
        wrist_to_tip = distance(self.pts[0], self.pts[4])
        wrist_to_ip = distance(self.pts[0], self.pts[3])
        horizontal_spread = abs(self.pts[4][0] - self.pts[2][0])
        return self.finger_angle["thumb"] > 135 and wrist_to_tip > wrist_to_ip * 1.02 and horizontal_spread > self.palm_size * 0.22


def as_point(point: object) -> tuple[float, float, float]:
    if isinstance(point, tuple) or isinstance(point, list):
        if len(point) == 2:
            return float(point[0]), float(point[1]), 0.0
        return float(point[0]), float(point[1]), float(point[2])
    return float(point.x), float(point.y), float(getattr(point, "z", 0.0))


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def angle(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> float:
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    ba_len = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    bc_len = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)
    if ba_len == 0 or bc_len == 0:
        return 0.0
    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    cosine = max(-1.0, min(1.0, dot / (ba_len * bc_len)))
    return math.degrees(math.acos(cosine))
