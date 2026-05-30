from __future__ import annotations

from dataclasses import dataclass


GENDER_MODES = {
    "man": "man",
    "woman": "woman",
    "neutral": "person",
}

AGE_MODES = {
    "young": "young",
    "adult": "adult",
    "elder": "elderly",
}

VISUAL_MODES = {
    "asian_american": {
        "label": "ASIAN AMERICAN",
        "subject_prefix": "Asian-American",
        "context": (
            "capturing a diverse Asian-American identity, blending modern US urban "
            "settings with subtle traditional Asian cultural motifs and textures"
        ),
    },
    "black_brown": {
        "label": "BLACK AND BROWN PEOPLE",
        "subject_prefix": "Black or Brown",
        "context": (
            "centering Black and Brown people, contemporary US urban life, rich "
            "diasporic cultural textures, warm natural skin tones, dignified and "
            "vibrant representation"
        ),
    },
    "asian_black_brown": {
        "label": "ASIAN + BLACK AND BROWN PEOPLE",
        "subject_prefix": "Asian, Black, or Brown",
        "context": (
            "centering Asian, Black, and Brown people together, diverse contemporary "
            "US community life, layered diasporic cultural textures, warm natural "
            "skin tones, dignified and vibrant representation"
        ),
    },
}

FIXED_PROMPT_TEMPLATE = (
    "A hyper-realistic photorealistic cinematic shot of {text} featuring a "
    "prominent {age_desc} {subject_focus}, {visual_context}, 8k UHD, highly "
    "detailed, masterfully lit, RAW photo, shot on 35mm lens, f/1.8, natural "
    "colors, masterpiece"
)


@dataclass
class PromptState:
    gender: str = "neutral"
    age: str = "adult"
    visual_mode: str = "asian_american"


class PromptBuilder:
    def __init__(self, max_text_chars: int = 160):
        self.max_text_chars = max_text_chars

    def build(self, text: str, state: PromptState) -> str:
        clean_text = normalize_signed_text(text)
        if not clean_text:
            return ""

        clean_text = clean_text[-self.max_text_chars :].strip()
        visual_mode = VISUAL_MODES.get(state.visual_mode, VISUAL_MODES["asian_american"])
        gender_focus = GENDER_MODES.get(state.gender, GENDER_MODES["neutral"])
        age_desc = AGE_MODES.get(state.age, AGE_MODES["adult"])
        subject_focus = f"{visual_mode['subject_prefix']} {gender_focus}"

        return FIXED_PROMPT_TEMPLATE.format(
            text=clean_text,
            age_desc=age_desc,
            subject_focus=subject_focus,
            visual_context=visual_mode["context"],
        )


class SignedTextBuffer:
    def __init__(self, max_chars: int = 160):
        self.max_chars = max_chars
        self.text = ""

    def apply(self, token: str) -> bool:
        token = token.strip()
        if not token:
            return False

        upper_token = token.upper()
        before = self.text

        if upper_token == "SPACE":
            if self.text and not self.text.endswith(" "):
                self.text += " "
        elif upper_token == "BACKSPACE":
            self.text = self.text[:-1]
        elif upper_token == "CLEAR":
            self.text = ""
        elif upper_token.startswith("WORD:"):
            word = token.split(":", 1)[1].strip()
            if word:
                if self.text and not self.text.endswith(" "):
                    self.text += " "
                self.text += word
        elif len(upper_token) == 1 and upper_token.isalpha():
            self.text += upper_token.lower()
        else:
            return False

        self.text = self.text[-self.max_chars :]
        return self.text != before

    def clear(self) -> None:
        self.text = ""

    def normalized(self) -> str:
        return normalize_signed_text(self.text)


class GestureCommitter:
    def __init__(
        self,
        hold_seconds: float = 0.75,
        release_seconds: float = 0.25,
        repeat_cooldown_seconds: float = 1.8,
    ):
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds
        self.repeat_cooldown_seconds = repeat_cooldown_seconds
        self.candidate = None
        self.candidate_since = None
        self.last_seen_at = None
        self.last_committed = None
        self.last_commit_at = None

    def update(self, token: str | None, now: float) -> str | None:
        if not token:
            if self.last_seen_at is not None and now - self.last_seen_at >= self.release_seconds:
                self.candidate = None
                self.candidate_since = None
                self.last_committed = None
            return None

        self.last_seen_at = now
        if token != self.candidate:
            self.candidate = token
            self.candidate_since = now
            return None

        if self.candidate_since is None or now - self.candidate_since < self.hold_seconds:
            return None

        if token != self.last_committed:
            self.last_committed = token
            self.last_commit_at = now
            return token

        if self.last_commit_at is not None and now - self.last_commit_at >= self.repeat_cooldown_seconds:
            self.last_commit_at = now
            return token

        return None


def normalize_signed_text(text: str) -> str:
    return " ".join(text.strip().split())
