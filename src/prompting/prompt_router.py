"""
Intent classification and prompt routing for the image generation system.
Provides dynamic intent classification using structured system roles and configuration.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import openai
from openai import OpenAI

from prompting.constants import (
    DEFAULT_SYSTEM_CONFIG,
    PromptConfig,
    DEFAULT_PROMPT_CONFIG,
)

# === Logging Setup ===
logger = logging.getLogger(__name__)

# === OpenAI Client ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class IntentClassification:
    """Structured result of intent classification."""
    intent: str
    confidence: float
    category: str
    context: Dict[str, Any]
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

@dataclass
class ClassificationContext:
    """Context for intent classification."""
    prompt: str
    config: PromptConfig = field(default_factory=lambda: DEFAULT_PROMPT_CONFIG)
    strict_mode: bool = False

class IntentClassifier:
    """
    Classifies prompt intent using dynamic system roles and configuration.
    Uses template fields and context for inference.
    """

    def __init__(self, config: PromptConfig = DEFAULT_PROMPT_CONFIG):
        """
        Initialize the intent classifier.

        Args:
            config: Configuration for classification
        """
        self.config = config
        self.system_role = DEFAULT_SYSTEM_CONFIG["roles"]["intent_classifier"]
        logger.info(f"Initialized IntentClassifier with config: {config}")

    def _build_system_message(self, context: ClassificationContext) -> str:
        """
        Build a dynamic system message for intent classification.

        Args:
            context: ClassificationContext containing all relevant information

        Returns:
            str: The complete system message
        """
        # Build base message using system role
        message = f"""
{self.system_role}

Your task is to classify the underlying intent of a user's prompt for AI image generation.

Consider these aspects when classifying:
1. Primary purpose (commercial, artistic, social, educational)
2. Target audience and context
3. Desired emotional impact
4. Visual style and technique preferences
5. Brand or message requirements

Return only a single lowercase label (one word or hyphenated phrase) that best describes the intent.
Do not include examples, punctuation, or explanations.

Classification guidelines:
- Focus on the core purpose and desired outcome
- Consider the visual style and technique preferences
- Infer intent from context and field values
- Be specific but concise in labeling
"""

        # Add strict mode instructions if enabled
        if context.strict_mode:
            message += "\n\nStrict mode enabled: Only use predefined intent labels."
            known_intents = list(DEFAULT_SYSTEM_CONFIG["intent_instructions"].keys())
            message += f"\nKnown intents: {', '.join(known_intents)}"

        return message.strip()

    def classify(self, context: ClassificationContext) -> IntentClassification:
        """
        Classify the intent of a prompt.

         Args:
            context: ClassificationContext containing the prompt and configuration

    Returns:
            IntentClassification: The classification result
        """
        try:
            # Build system message
            system_message = self._build_system_message(context)

            # Get GPT classification
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": context.prompt},
                ],
                temperature=0.3,
                max_tokens=50,
            )

            # Extract and clean intent
            intent = response.choices[0].message.content.strip().lower()
            logger.info(f"Classified intent: {intent}")

            # Validate intent if strict mode is enabled
            warnings = []
            if context.strict_mode:
                known_intents = set(DEFAULT_SYSTEM_CONFIG["intent_instructions"].keys())
                if intent not in known_intents:
                    warnings.append(f"Intent '{intent}' not in known intents")
                    # Use most similar known intent
                    intent = self._find_similar_intent(intent, known_intents)

            # Determine category based on intent
            category = self._get_intent_category(intent)

            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence(response)

            return IntentClassification(
                intent=intent,
                confidence=confidence,
                category=category,
                context={"response": response},
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return IntentClassification(
                intent="unknown",
                confidence=0.0,
                category="other",
                context={"error": str(e)},
                warnings=[f"Classification failed: {str(e)}"]
            )

    def _find_similar_intent(self, intent: str, known_intents: set) -> str:
        """Find the most similar known intent."""
        # Simple similarity check - can be enhanced with better matching
        for known in known_intents:
            if known in intent or intent in known:
                return known
        return "artistic-expression"  # Default fallback

    def _get_intent_category(self, intent: str) -> str:
        """Determine the category for a given intent."""
        categories = {
            "commercial": ["product-ad", "service-promotion", "brand-storytelling", "campaign-launch"],
            "artistic": ["artistic-expression", "experimental-style"],
            "social": ["public-awareness", "social-trend"],
            "educational": ["educational-content"]
        }
        
        for category, intents in categories.items():
            if intent in intents:
                return category
        return "other"

    def _calculate_confidence(self, response: Any) -> float:
        """Calculate confidence score for the classification."""
        # Base confidence on response characteristics
        confidence = 0.7  # Base confidence
        
        # Adjust based on response length (shorter is more confident)
        content = response.choices[0].message.content
        if len(content.split()) <= 2:
            confidence += 0.2
        
        # Adjust based on response format (cleaner is more confident)
        if content.islower() and "-" in content:
            confidence += 0.1
            
        return min(confidence, 1.0)

class PromptRouter:
    """Routes prompts to appropriate handlers based on intent and context."""
    
    def __init__(self, config: PromptConfig = DEFAULT_PROMPT_CONFIG):
        """Initialize the prompt router."""
        self.classifier = IntentClassifier(config)
        logger.info("Initialized PromptRouter")
    
    def route_prompt(self, prompt: str, strict_mode: bool = False) -> str:
        """
        Route a prompt through the appropriate processing pipeline.

    Args:
            prompt (str): The input prompt
            strict_mode (bool): Whether to enforce known intents

    Returns:
            str: The processed prompt
        """
        try:
            # Create classification context
            context = ClassificationContext(
                prompt=prompt,
                config=self.classifier.config,
                strict_mode=strict_mode
            )
            
            # Classify intent
            classification = self.classifier.classify(context)
            logger.info(f"Classified intent: {classification.intent} (confidence: {classification.confidence:.2f})")
            
            # Log any warnings
            for warning in classification.warnings:
                logger.warning(warning)
            
            # Apply category-specific processing
            if classification.confidence < 0.5:
                logger.warning("Low confidence classification, using minimal processing")
                return prompt
                
            return self._process_by_category(prompt, classification)
                
        except Exception as e:
            logger.error(f"Error routing prompt: {e}")
            return prompt
    
    def _process_by_category(self, prompt: str, classification: IntentClassification) -> str:
        """Process prompt based on its category."""
        try:
            if classification.category == "commercial":
                return f"{prompt}, professional commercial photography, high-end product visualization"
            elif classification.category == "artistic":
                return f"{prompt}, artistic composition, creative expression"
            elif classification.category == "social":
                return f"{prompt}, social media optimized, engaging composition"
            elif classification.category == "educational":
                return f"{prompt}, clear educational visualization, informative composition"
            else:
                return prompt
        except Exception as e:
            logger.error(f"Error processing category {classification.category}: {e}")
            return prompt

def classify_prompt_intent(prompt: str, strict_mode: bool = False) -> str:
    """
    Classify the intent of a prompt using the intent classifier.

    Args:
        prompt (str): The input prompt
        strict_mode (bool): Whether to enforce known intents

    Returns:
        str: The classified intent label
    """
    try:
        router = PromptRouter()
        classification = router.classifier.classify(
            ClassificationContext(prompt=prompt, strict_mode=strict_mode)
        )
        return classification.intent
    except Exception as e:
        logger.error(f"Error classifying prompt intent: {e}")
        return "unknown"

def analyze_prompt_context(prompt: str) -> Dict[str, Any]:
    """
    Analyze the context of a prompt to provide additional insights.

    Args:
        prompt (str): The input prompt

    Returns:
        Dict[str, Any]: Dictionary containing context analysis
    """
    try:
        router = PromptRouter()
        classification = router.classifier.classify(
            ClassificationContext(prompt=prompt)
        )

        return {
            "intent": classification.intent,
            "category": classification.category,
            "confidence": classification.confidence,
            "warnings": classification.warnings,
            "context": classification.context
        }
    except Exception as e:
        logger.error(f"Error analyzing prompt context: {e}")
        return {
            "intent": "unknown",
            "category": "other",
            "confidence": 0.0,
            "warnings": [f"Analysis failed: {str(e)}"],
            "context": {}
        }
