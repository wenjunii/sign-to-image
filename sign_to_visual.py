from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

from gesture_rules import GestureResult, RuleBasedASLRecognizer
from prompting import (
    AGE_MODES,
    GENDER_MODES,
    VISUAL_MODES,
    GestureCommitter,
    PromptBuilder,
    PromptState,
    SignedTextBuffer,
)

cv2 = None
mp = None
udp_client = None
ALLOW_LOCAL_VENV_REEXEC = False


def load_runtime_dependencies() -> None:
    global cv2, mp, udp_client

    missing = []
    try:
        import cv2 as cv2_module
    except ImportError:
        missing.append("opencv-python")
    else:
        cv2 = cv2_module

    try:
        import mediapipe as mp_module
    except ImportError:
        missing.append("mediapipe")
    else:
        if not hasattr(mp_module, "solutions") or not hasattr(mp_module.solutions, "hands"):
            missing.append("mediapipe with legacy solutions.hands support")
        else:
            mp = mp_module

    try:
        from pythonosc import udp_client as udp_client_module
    except ImportError:
        missing.append("pythonosc")
    else:
        udp_client = udp_client_module

    if missing:
        venv_python = local_venv_python()
        if ALLOW_LOCAL_VENV_REEXEC and venv_python:
            print(f"Missing dependencies in {sys.executable}; retrying with project venv.", flush=True)
            os.execv(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]])

        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Install them with: pip install -r requirements.txt"
        )


def local_venv_python() -> str | None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    if not os.path.exists(candidate):
        return None
    if os.path.abspath(candidate).lower() == os.path.abspath(sys.executable).lower():
        return None
    return candidate


def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live sign-language-to-visual prompt bridge for StreamDiffusionTD.")
    parser.add_argument("--camera", type=int, help="Camera index. Overrides CAMERA_INDEX from .env.")
    parser.add_argument("--osc-ip", help="OSC target IP. Overrides OSC_IP from .env.")
    parser.add_argument("--osc-port", type=int, help="OSC target port. Overrides OSC_PORT from .env.")
    parser.add_argument(
        "--commit-mode",
        choices=["auto", "manual"],
        help="auto commits held signs; manual commits current sign with Space.",
    )
    parser.add_argument("--no-preview", action="store_true", help="Disable the OpenCV preview window.")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the camera image.")
    return parser.parse_args()


@dataclass
class PipelineConfig:
    camera_index: int
    mirror_camera: bool
    show_preview: bool
    osc_ip: str
    osc_port: int
    osc_prompt_address: str
    osc_partial_text_address: str
    osc_gesture_address: str
    min_confidence: float
    commit_mode: str
    auto_send_prompt: bool
    hold_seconds: float
    release_seconds: float
    repeat_cooldown_seconds: float
    max_text_chars: int


class SignToVisualPipeline:
    def __init__(self, config: PipelineConfig):
        load_runtime_dependencies()

        self.config = config
        self.recognizer = RuleBasedASLRecognizer()
        self.committer = GestureCommitter(
            hold_seconds=config.hold_seconds,
            release_seconds=config.release_seconds,
            repeat_cooldown_seconds=config.repeat_cooldown_seconds,
        )
        self.text_buffer = SignedTextBuffer(max_chars=config.max_text_chars)
        self.prompt_builder = PromptBuilder(max_text_chars=config.max_text_chars)
        self.prompt_state = PromptState(
            gender=os.environ.get("DEFAULT_GENDER", "neutral"),
            age=os.environ.get("DEFAULT_AGE", "adult"),
            visual_mode=os.environ.get("DEFAULT_VISUAL_MODE", "asian_american"),
        )
        self.osc_client = udp_client.SimpleUDPClient(config.osc_ip, config.osc_port)
        self.last_prompt = ""
        self.last_detected: GestureResult | None = None
        self.is_running = True

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.6,
        )

    def start(self) -> None:
        capture = cv2.VideoCapture(self.config.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera index {self.config.camera_index}.")

        print("\n" + "=" * 58)
        print("SIGN TO VISUAL STREAMDIFFUSION BRIDGE")
        print(f"OSC target: {self.config.osc_ip}:{self.config.osc_port}")
        print(f"Commit mode: {self.config.commit_mode}")
        print("Keys: Space commit/space | Enter send | Backspace delete | c clear | q/Esc exit")
        print("Identity: m man | w woman | n neutral | 1 young | 2 adult | 3 elder")
        print("Visual: d Asian American | b Black and Brown | x Asian + Black and Brown")
        print("=" * 58 + "\n")

        try:
            while self.is_running:
                ok, frame = capture.read()
                if not ok:
                    print("Camera frame could not be read.")
                    break

                if self.config.mirror_camera:
                    frame = cv2.flip(frame, 1)

                active = self.process_frame(frame)
                self.handle_auto_commit(active)

                if self.config.show_preview:
                    self.draw_overlay(frame, active)
                    cv2.imshow("Sign to Visual", frame)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF

                self.handle_key(key, active)
        finally:
            self.hands.close()
            capture.release()
            if self.config.show_preview:
                cv2.destroyAllWindows()

    def process_frame(self, frame) -> GestureResult | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        detections: list[GestureResult] = []

        hand_landmarks = results.multi_hand_landmarks or []
        handedness = results.multi_handedness or []
        for index, landmarks in enumerate(hand_landmarks):
            hand_label = "Right"
            if index < len(handedness):
                hand_label = handedness[index].classification[0].label

            result = self.recognizer.recognize_single(landmarks.landmark, hand_label)
            if result is not None:
                detections.append(result)

            if self.config.show_preview:
                self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

        command = self.recognizer.recognize_command(detections)
        if command is not None:
            self.last_detected = command
            return command

        confident = [item for item in detections if item.confidence >= self.config.min_confidence]
        if not confident:
            self.last_detected = None
            return None

        active = max(confident, key=lambda item: item.confidence)
        self.last_detected = active
        return active

    def handle_auto_commit(self, active: GestureResult | None) -> None:
        if self.config.commit_mode != "auto":
            return

        now = time.time()
        token = active.label if active and active.confidence >= self.config.min_confidence else None
        committed = self.committer.update(token, now)
        if committed:
            self.apply_token(committed, source="auto")

    def handle_key(self, key: int, active: GestureResult | None) -> None:
        if key in {ord("q"), 27}:
            self.is_running = False
        elif key in {13, 10}:
            self.send_prompt(reason="enter")
        elif key in {8, 127}:
            self.apply_token("BACKSPACE", source="keyboard")
        elif key == ord("c"):
            self.apply_token("CLEAR", source="keyboard")
        elif key == ord(" "):
            if self.config.commit_mode == "manual" and active and active.kind == "letter":
                self.apply_token(active.label, source="manual")
            else:
                self.apply_token("SPACE", source="keyboard")
        elif key == ord("m"):
            self.set_gender("man")
        elif key == ord("w"):
            self.set_gender("woman")
        elif key == ord("n"):
            self.set_gender("neutral")
        elif key == ord("1"):
            self.set_age("young")
        elif key == ord("2"):
            self.set_age("adult")
        elif key == ord("3"):
            self.set_age("elder")
        elif key == ord("d"):
            self.set_visual_mode("asian_american")
        elif key == ord("b"):
            self.set_visual_mode("black_brown")
        elif key == ord("x"):
            self.set_visual_mode("asian_black_brown")

    def apply_token(self, token: str, source: str) -> None:
        if token == "SEND":
            self.send_prompt(reason=source)
            return

        changed = self.text_buffer.apply(token)
        if changed:
            text = self.text_buffer.normalized()
            self.osc_client.send_message(self.config.osc_partial_text_address, text)
            self.osc_client.send_message(self.config.osc_gesture_address, token)
            print(f"[TEXT:{source}] {text}")
            if self.config.auto_send_prompt:
                self.send_prompt(reason=source)

    def send_prompt(self, reason: str) -> None:
        text = self.text_buffer.normalized()
        prompt = self.prompt_builder.build(text, self.prompt_state)
        if not prompt:
            return
        if prompt == self.last_prompt:
            return
        self.osc_client.send_message(self.config.osc_prompt_address, prompt)
        self.osc_client.send_message(self.config.osc_partial_text_address, text)
        self.last_prompt = prompt
        print(f"[PROMPT:{reason}] {prompt}")

    def set_gender(self, value: str) -> None:
        if value in GENDER_MODES:
            self.prompt_state.gender = value
            print(f"[MODE] GENDER -> {value.upper()}")

    def set_age(self, value: str) -> None:
        if value in AGE_MODES:
            self.prompt_state.age = value
            print(f"[MODE] AGE -> {value.upper()}")

    def set_visual_mode(self, value: str) -> None:
        if value in VISUAL_MODES:
            self.prompt_state.visual_mode = value
            print(f"[MODE] VISUAL -> {VISUAL_MODES[value]['label']}")

    def draw_overlay(self, frame, active: GestureResult | None) -> None:
        height, width = frame.shape[:2]
        detected = "None"
        if active:
            detected = f"{active.label} ({active.confidence:.2f})"

        lines = [
            f"Gesture: {detected}",
            f"Text: {self.text_buffer.text[-64:]}",
            (
                f"Mode: {self.config.commit_mode} | "
                f"{self.prompt_state.gender}/{self.prompt_state.age}/{self.prompt_state.visual_mode}"
            ),
        ]

        x = 14
        y = 28
        for line in lines:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
            y += 30

        footer = "Enter send | Space commit/space | Backspace delete | q exit"
        cv2.rectangle(frame, (0, height - 34), (width, height), (0, 0, 0), -1)
        cv2.putText(frame, footer, (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)


def make_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        camera_index=args.camera if args.camera is not None else env_int("CAMERA_INDEX", 0),
        mirror_camera=False if args.no_mirror else env_bool("MIRROR_CAMERA", True),
        show_preview=False if args.no_preview else env_bool("SHOW_PREVIEW", True),
        osc_ip=args.osc_ip or os.environ.get("OSC_IP", "127.0.0.1"),
        osc_port=args.osc_port or env_int("OSC_PORT", 7001),
        osc_prompt_address=os.environ.get("OSC_PROMPT_ADDRESS", "/prompt"),
        osc_partial_text_address=os.environ.get("OSC_PARTIAL_TEXT_ADDRESS", "/partial_text"),
        osc_gesture_address=os.environ.get("OSC_GESTURE_ADDRESS", "/gesture"),
        min_confidence=env_float("SIGN_MIN_CONFIDENCE", 0.64),
        commit_mode=args.commit_mode or os.environ.get("SIGN_COMMIT_MODE", "auto").strip().lower(),
        auto_send_prompt=env_bool("AUTO_SEND_PROMPT", True),
        hold_seconds=env_float("SIGN_HOLD_SECONDS", 0.75),
        release_seconds=env_float("SIGN_RELEASE_SECONDS", 0.25),
        repeat_cooldown_seconds=env_float("SIGN_REPEAT_COOLDOWN_SECONDS", 1.8),
        max_text_chars=env_int("MAX_TEXT_CHARS", 160),
    )


def main() -> int:
    global ALLOW_LOCAL_VENV_REEXEC
    ALLOW_LOCAL_VENV_REEXEC = True

    load_env_file()
    args = parse_args()
    config = make_config(args)

    if config.commit_mode not in {"auto", "manual"}:
        print(f"Unknown SIGN_COMMIT_MODE '{config.commit_mode}', falling back to auto.")
        config.commit_mode = "auto"

    backend = os.environ.get("SIGN_RECOGNITION_BACKEND", "rule_based").strip().lower()
    if backend != "rule_based":
        print(f"SIGN_RECOGNITION_BACKEND={backend} is not implemented yet; using rule_based.")

    try:
        SignToVisualPipeline(config).start()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
