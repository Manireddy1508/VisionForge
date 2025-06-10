"""
Utility functions for prompt validation and processing.
Provides dynamic template-based validation and field extraction.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Set

from prompting.constants import (
    TEMPLATE_FIELDS,
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    REQUIRED_ELEMENTS,
    OPTIONAL_ELEMENTS,
    PromptConfig,
    DEFAULT_PROMPT_CONFIG,
)

# === Logging Setup ===
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Structured result of prompt validation."""
    is_valid: bool
    missing_fields: Set[str]
    invalid_fields: Dict[str, str]
    warnings: List[str]
    cleaned_prompt: str

@dataclass
class TemplateFieldMatch:
    """Represents a matched template field in a prompt."""
    field_name: str
    field_type: str
    value: str
    start_pos: int
    end_pos: int

class PromptValidator:
    """
    Validates and cleans prompts according to template fields and configuration.
    Uses dynamic template structure from constants.py.
    """

    def __init__(self, config: PromptConfig = DEFAULT_PROMPT_CONFIG):
        """
        Initialize the prompt validator.

        Args:
            config: Validation configuration
        """
        self.config = config
        logger.info(f"Initialized PromptValidator with config: {config}")

    def validate(self, prompt: str, allow_manual: bool = False) -> ValidationResult:
        """
        Validate a prompt against template fields and configuration.

        Args:
            prompt (str): The prompt to validate
            allow_manual (bool): Whether to allow manual overrides

        Returns:
            ValidationResult: Structured validation result
        """
        # Initialize result
        result = ValidationResult(
            is_valid=True,
            missing_fields=set(),
            invalid_fields={},
            warnings=[],
            cleaned_prompt=prompt.strip()
        )

        # Clean the prompt
        result.cleaned_prompt = self.clean(result.cleaned_prompt)
        if not result.cleaned_prompt:
            result.is_valid = False
            result.warnings.append("Empty prompt after cleaning")
            return result

        # Check word count
        words = result.cleaned_prompt.split()
        if len(words) < self.config.min_word_count:
            result.warnings.append(f"Too few words (minimum: {self.config.min_word_count})")
            if not allow_manual:
                result.is_valid = False
                return result

        # Check single sentence
        if self.config.enforce_single_sentence and not self._is_single_sentence(result.cleaned_prompt):
            result.warnings.append("Multiple sentences detected")
            if not allow_manual:
                result.is_valid = False
                return result

        # Extract and validate template fields
        field_matches = self._extract_template_fields(result.cleaned_prompt)
        logger.debug(f"Found {len(field_matches)} template fields in prompt")
        
        # Check required fields
        for field_name in REQUIRED_ELEMENTS:
            if not any(m.field_name == field_name for m in field_matches):
                result.missing_fields.add(field_name)
                result.warnings.append(f"Missing required field: {field_name}")
                if not allow_manual:
                    result.is_valid = False

        # Validate field constraints
        for match in field_matches:
            field = TEMPLATE_FIELDS.get(match.field_name)
            if field and field.constraints:
                for constraint in field.constraints:
                    if not self._validate_constraint(match.value, constraint):
                        result.invalid_fields[match.field_name] = constraint
                        result.warnings.append(f"Invalid {match.field_name}: {constraint}")
                        if not allow_manual:
                            result.is_valid = False

        # Check prefix/suffix if required
        if self.config.require_realistic_scale:
            if not result.cleaned_prompt.lower().startswith(PROMPT_PREFIX.lower()):
                result.warnings.append(f"Missing required prefix: {PROMPT_PREFIX}")
                if not allow_manual:
                    result.is_valid = False
            if not result.cleaned_prompt.lower().endswith(PROMPT_SUFFIX.lower()):
                result.warnings.append(f"Missing required suffix: {PROMPT_SUFFIX}")
                if not allow_manual:
                    result.is_valid = False

        logger.debug(f"Validation result: {result}")
        return result

    def _is_single_sentence(self, prompt: str) -> bool:
        """Check if the prompt is a single sentence."""
        # Split on sentence boundaries
        sentences = [s.strip() for s in re.split(r"[.!?]+", prompt) if s.strip()]
        return len(sentences) <= 1

    def _extract_template_fields(self, prompt: str) -> List[TemplateFieldMatch]:
        """Extract template fields from the prompt."""
        matches = []
        for field_name, field in TEMPLATE_FIELDS.items():
            # Look for field in square brackets
            pattern = rf"\[{field.name}\]"
            for match in re.finditer(pattern, prompt, re.IGNORECASE):
                # Extract the value between brackets
                start = match.start()
                end = match.end()
                value = prompt[start:end].strip("[]")
                
                matches.append(TemplateFieldMatch(
                    field_name=field_name,
                    field_type=field.field_type.name,
                    value=value,
                    start_pos=start,
                    end_pos=end
                ))
        return matches

    def _validate_constraint(self, value: str, constraint: str) -> bool:
        """Validate a field value against its constraint."""
        # Basic constraint validation - can be extended for specific constraints
        if "must be" in constraint.lower():
            return bool(value.strip())
            return True

    def clean(self, prompt: str) -> str:
        """
        Clean a prompt according to configuration.

        Args:
            prompt (str): The prompt to clean

        Returns:
            str: The cleaned prompt
        """
        # First strip any leading/trailing whitespace
        cleaned = prompt.strip()
        
        # Clean quotes - handle both single and double quotes
        if self.config.strip_quotes:
            # Remove any combination of quotes at the start and end
            cleaned = cleaned.strip('"\'')
            # Also remove any escaped quotes
            cleaned = cleaned.replace('\\"', '').replace("\\'", '')
            
        # Clean numbering
        if self.config.suppress_numbering:
            cleaned = cleaned.lstrip("0123456789. )").strip()
            
        return cleaned

def extract_valid_prompts(
    raw_output: str,
    num_prompts: int,
    fallback_prompt: str,
    config: PromptConfig = DEFAULT_PROMPT_CONFIG,
    allow_manual_override: bool = True,
    return_metadata: bool = False,
) -> List[str] | List[Tuple[str, bool]]:
    """
    Extract and clean valid prompts from raw GPT output.

    Args:
        raw_output (str): Raw output from GPT
        num_prompts (int): Number of prompts to extract
        fallback_prompt (str): Fallback prompt if not enough valid prompts
        config (PromptConfig): Validation configuration
        allow_manual_override (bool): Whether to allow manual overrides
        return_metadata (bool): Whether to return metadata with prompts

    Returns:
        List[str] | List[Tuple[str, bool]]: List of valid prompts, optionally with metadata
    """
    logger.info(f"Extracting {num_prompts} valid prompts from raw output")
    logger.debug(f"Raw output: {raw_output}")
    logger.debug(f"Mode: {'manual override' if allow_manual_override else 'strict'}")

    validator = PromptValidator(config=config)
    
    # Split and clean the raw output
    raw_lines = raw_output.split("\n")
    cleaned = []
    for line in raw_lines:
        if line.strip():
            # First clean the line
            cleaned_line = validator.clean(line)
            # Remove any remaining quotes at the start/end
            cleaned_line = cleaned_line.strip('"\'')
            # Remove any escaped quotes
            cleaned_line = cleaned_line.replace('\\"', '').replace("\\'", '')
            if cleaned_line:
                cleaned.append(cleaned_line)
    
    logger.debug(f"Cleaned lines: {cleaned}")

    results = []
    for line in cleaned:
        validation = validator.validate(line, allow_manual=allow_manual_override)
        if validation.is_valid:
            results.append((line, False)) if return_metadata else results.append(line)
            logger.debug(f"Accepted prompt: {line}")
        else:
            logger.warning(f"Rejected prompt: {line}")
            logger.debug(f"Validation issues: {validation.warnings}")

    # Pad with fallback prompts if needed
    while len(results) < num_prompts:
        result = (fallback_prompt, True) if return_metadata else fallback_prompt
        results.append(result)
        logger.debug(f"Added fallback prompt: {fallback_prompt}")

    # Truncate if too many
    if len(results) > num_prompts:
        results = results[:num_prompts]
        logger.debug(f"Truncated to {num_prompts} prompts")

    logger.info(f"Extracted {len(results)} valid prompts: {results}")
    return results

def extract_template_fields(prompt: str) -> Dict[str, str]:
    """
    Extract template fields from a prompt into a structured format.

    Args:
        prompt (str): The prompt to parse

    Returns:
        Dict[str, str]: Dictionary of field names to values
    """
    validator = PromptValidator()
    matches = validator._extract_template_fields(prompt)
    return {match.field_name: match.value for match in matches}
