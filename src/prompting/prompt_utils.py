import re
from typing import List, Tuple, Dict, Optional

from prompting.constants import (
    MIN_WORD_COUNT,
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    PROMPT_SETTINGS,
)

# === Flexible Rules ===
FLEXIBLE_PREFIX_PATTERNS = [r"it's very important that"]

FLEXIBLE_SUFFIX_PATTERNS = [
    r"take all the time you need(?:ed)?(?:.*?)?achieve the best possible result",
    r"achieve the best possible result",
]


class PromptValidator:
    """
    Validates and cleans prompts according to specified rules.
    """

    def __init__(self, settings: Dict = PROMPT_SETTINGS, mode: str = "lenient"):
        """
        Initialize the prompt validator.

        Args:
            settings (Dict): Validation settings
            mode (str): Validation mode ("strict" or "lenient")
        """
        self.settings = settings
        self.strict_mode = mode == "strict"
        print(f"\n🎯 [DEBUG] Initialized PromptValidator in {mode} mode")

    def is_valid(self, prompt: str, allow_manual: bool = False) -> bool:
        """
        Check if a prompt is valid according to the rules.

        Args:
            prompt (str): The prompt to validate
            allow_manual (bool): Whether to allow manual overrides

        Returns:
            bool: Whether the prompt is valid
        """
        prompt = prompt.strip()
        words = prompt.split()

        if len(words) <= MIN_WORD_COUNT:
            print("⛔ [VALIDATOR] Too few words")
            return False if not allow_manual else True

        if self.settings.get(
            "enforce_single_sentence"
        ) and not self._is_single_sentence(prompt):
            print("⛔ [VALIDATOR] More than one sentence")
            return False if not allow_manual else True

        if self.settings.get("strip_quotes") and prompt.startswith(("'", '"')):
            print("⛔ [VALIDATOR] Starts with quote")
            return False if not allow_manual else True

        if self.settings.get("suppress_numbering") and prompt.lstrip()[0].isdigit():
            print("⛔ [VALIDATOR] Starts with number")
            return False if not allow_manual else True

        if self.settings.get("require_realistic_scale") and not self._has_valid_suffix(
            prompt, words
        ):
            print("⛔ [VALIDATOR] Suffix not found")
            return False if not allow_manual else True

        if not self._has_valid_prefix(prompt):
            print("⛔ [VALIDATOR] Prefix not found")
            return False if not allow_manual else True

        print("✅ [VALIDATOR] Prompt is valid")
        return True

    def _is_single_sentence(self, prompt: str) -> bool:
        """
        Check if the prompt is a single sentence.

        Args:
            prompt (str): The prompt to check

        Returns:
            bool: Whether the prompt is a single sentence
        """
        if any(re.search(pat, prompt.lower()) for pat in FLEXIBLE_SUFFIX_PATTERNS):
            return True
        return len([s for s in re.split(r"[.!?]+", prompt.strip()) if s.strip()]) <= 1

    def _has_valid_prefix(self, prompt: str) -> bool:
        """
        Check if the prompt has a valid prefix.

        Args:
            prompt (str): The prompt to check

        Returns:
            bool: Whether the prompt has a valid prefix
        """
        text_start = " ".join(prompt.strip().lower().split()[:30])
        return any(re.search(pat, text_start) for pat in FLEXIBLE_PREFIX_PATTERNS)

    def _has_valid_suffix(self, prompt: str, words: List[str]) -> bool:
        """
        Check if the prompt has a valid suffix.

        Args:
            prompt (str): The prompt to check
            words (List[str]): List of words in the prompt

        Returns:
            bool: Whether the prompt has a valid suffix
        """
        tail = " ".join(words[-30:]).lower()
        return any(re.search(pat, tail) for pat in FLEXIBLE_SUFFIX_PATTERNS)

    def clean(self, prompt: str) -> str:
        """
        Clean a prompt according to the settings.

        Args:
            prompt (str): The prompt to clean

        Returns:
            str: The cleaned prompt
        """
        cleaned = prompt.strip()
        if self.settings.get("strip_quotes"):
            cleaned = cleaned.strip('"').strip("'")
        if self.settings.get("suppress_numbering"):
            cleaned = cleaned.lstrip("0123456789. )").strip()
        return cleaned


def extract_valid_prompts(
    raw_output: str,
    num_prompts: int,
    fallback_prompt: str,
    mode: str = "lenient",
    allow_manual_override: bool = False,
    return_metadata: bool = False,
) -> List[str] | List[Tuple[str, bool]]:
    """
    Extract and clean valid prompts from raw GPT output.

    Args:
        raw_output (str): Raw output from GPT
        num_prompts (int): Number of prompts to extract
        fallback_prompt (str): Fallback prompt if not enough valid prompts
        mode (str): Validation mode ("strict" or "lenient")
        allow_manual_override (bool): Whether to allow manual overrides
        return_metadata (bool): Whether to return metadata with prompts

    Returns:
        List[str] | List[Tuple[str, bool]]: List of valid prompts, optionally with metadata
    """
    print(f"\n🎯 [DEBUG] Extracting {num_prompts} valid prompts")
    print(f"📌 [DEBUG] Mode: {mode}")
    print(f"📌 [DEBUG] Allow manual override: {allow_manual_override}")

    validator = PromptValidator(mode=mode)
    raw_lines = raw_output.split("\n")
    cleaned = [validator.clean(line) for line in raw_lines if line.strip()]

    results = []
    for line in cleaned:
        is_valid = validator.is_valid(line, allow_manual=allow_manual_override)
        if is_valid:
            results.append((line, False)) if return_metadata else results.append(line)
        else:
            print(f"⚠️ [DEBUG] Rejected: {line}")

    while len(results) < num_prompts:
        result = (fallback_prompt, True) if return_metadata else fallback_prompt
        results.append(result)
        print(f"📝 [DEBUG] Added fallback prompt: {fallback_prompt}")

    if len(results) > num_prompts:
        results = results[:num_prompts]
        print(f"📝 [DEBUG] Truncated to {num_prompts} prompts")

    print(f"✅ [DEBUG] Extracted {len(results)} valid prompts")
    return results
