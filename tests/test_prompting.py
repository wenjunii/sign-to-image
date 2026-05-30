import unittest

from prompting import GestureCommitter, PromptBuilder, PromptState, SignedTextBuffer


class SignedTextBufferTests(unittest.TestCase):
    def test_applies_letters_space_backspace_and_clear(self):
        buffer = SignedTextBuffer(max_chars=20)

        self.assertTrue(buffer.apply("C"))
        self.assertTrue(buffer.apply("A"))
        self.assertTrue(buffer.apply("T"))
        self.assertEqual(buffer.normalized(), "cat")

        self.assertTrue(buffer.apply("SPACE"))
        self.assertTrue(buffer.apply("D"))
        self.assertEqual(buffer.normalized(), "cat d")

        self.assertTrue(buffer.apply("BACKSPACE"))
        self.assertEqual(buffer.normalized(), "cat")

        self.assertTrue(buffer.apply("CLEAR"))
        self.assertEqual(buffer.normalized(), "")


class PromptBuilderTests(unittest.TestCase):
    def test_builds_reference_style_prompt(self):
        prompt = PromptBuilder().build(
            "a glowing city at night",
            PromptState(gender="woman", age="young", visual_mode="black_brown"),
        )

        self.assertIn("a glowing city at night", prompt)
        self.assertIn("young Black or Brown woman", prompt)
        self.assertIn("RAW photo", prompt)


class GestureCommitterTests(unittest.TestCase):
    def test_commits_after_stable_hold(self):
        committer = GestureCommitter(hold_seconds=0.5, release_seconds=0.2, repeat_cooldown_seconds=1.0)

        self.assertIsNone(committer.update("A", now=10.0))
        self.assertIsNone(committer.update("A", now=10.3))
        self.assertEqual(committer.update("A", now=10.6), "A")
        self.assertIsNone(committer.update("A", now=10.7))

        self.assertIsNone(committer.update(None, now=11.0))
        self.assertIsNone(committer.update("A", now=11.1))
        self.assertEqual(committer.update("A", now=11.7), "A")


if __name__ == "__main__":
    unittest.main()
