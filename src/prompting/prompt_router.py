import os
import re
import openai
from typing import List, Optional, Dict, Tuple

# === OpenAI Client ===
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Constants ===
DEFAULT_MODEL = "gpt-4"
DEFAULT_MAX_TOKENS = 10
DEFAULT_TEMPERATURE = 0.0

# === Intent Vocabulary ===
INTENT_LABELS = [
    "product-ad",
    "service-promotion",
    "public-awareness",
    "brand-storytelling",
    "artistic-expression",
    "social-trend",
    "educational-content",
    "campaign-launch",
    "experimental-style",
]

# === Intent Categories ===
INTENT_CATEGORIES = {
    "commercial": [
        "product-ad",
        "service-promotion",
        "brand-storytelling",
        "campaign-launch",
    ],
    "artistic": ["artistic-expression", "experimental-style"],
    "social": ["public-awareness", "social-trend"],
    "educational": ["educational-content"],
}

# === Intent Keywords ===
INTENT_KEYWORDS = {
    "product-ad": [
        "product",
        "advertisement",
        "commercial",
        "marketing",
        "brand",
        "promotion",
    ],
    "service-promotion": ["service", "offer", "promotion", "business", "professional"],
    "public-awareness": [
        "awareness",
        "campaign",
        "cause",
        "social",
        "public",
        "community",
    ],
    "brand-storytelling": [
        "brand",
        "story",
        "narrative",
        "identity",
        "values",
        "mission",
    ],
    "artistic-expression": [
        "art",
        "creative",
        "artistic",
        "expression",
        "style",
        "medium",
    ],
    "social-trend": ["trend", "viral", "social", "media", "popular", "current"],
    "educational-content": [
        "education",
        "learn",
        "teach",
        "inform",
        "explain",
        "demonstrate",
    ],
    "campaign-launch": [
        "launch",
        "campaign",
        "announcement",
        "release",
        "introduction",
    ],
    "experimental-style": [
        "experimental",
        "innovative",
        "unique",
        "creative",
        "artistic",
    ],
}

# === Base system message for GPT ===
DEFAULT_INTENT_SYSTEM_MESSAGE = f"""
You are an intent classification engine for Flux Pro's AI prompt enhancement system.

Given a short user prompt, your job is to classify the *underlying marketing or creative intent* using a single lowercase label (one word or hyphenated phrase).

Consider the following aspects when classifying:
1. Primary purpose (commercial, artistic, social, educational)
2. Target audience and context
3. Desired emotional impact
4. Visual style and technique preferences
5. Brand or message requirements

Only return the intent label. Do not include examples, punctuation, or explanations.

Recognized labels include:
{', '.join(INTENT_LABELS)}

You may infer a new label if appropriate, but return only the inferred label.
""".strip()


def extract_keywords(text: str) -> List[str]:
    """
    Extract relevant keywords from the text.

    Args:
        text (str): The input text

    Returns:
        List[str]: List of extracted keywords
    """
    # Convert to lowercase and split into words
    words = text.lower().split()

    # Remove common stop words and short words
    stop_words = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
    }
    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    return keywords


def calculate_intent_scores(text: str) -> Dict[str, float]:
    """
    Calculate intent scores based on keyword matching.

    Args:
        text (str): The input text

    Returns:
        Dict[str, float]: Dictionary of intent scores
    """
    keywords = extract_keywords(text)
    scores = {intent: 0.0 for intent in INTENT_LABELS}

    for intent, intent_keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in intent_keywords:
                scores[intent] += 1.0

    # Normalize scores
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    return scores


def classify_prompt_intent(prompt: str) -> str:
    """
    Classify the intent of a prompt using GPT and keyword analysis.

    Args:
        prompt (str): The input prompt

    Returns:
        str: The classified intent label
    """
    print(f"\n🎯 [DEBUG] Classifying intent for prompt: {prompt}")

    # Calculate keyword-based scores
    keyword_scores = calculate_intent_scores(prompt)
    print("\n📊 [DEBUG] Keyword-based scores:")
    for intent, score in sorted(
        keyword_scores.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {intent}: {score:.2f}")

    try:
        # Get GPT classification
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": DEFAULT_INTENT_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        gpt_intent = response.choices[0].message.content.strip().lower()
        print(f"\n🤖 [DEBUG] GPT classification: {gpt_intent}")

        # If GPT's classification has a high keyword score, use it
        if keyword_scores.get(gpt_intent, 0) > 0.3:
            return gpt_intent

        # Otherwise, use the highest scoring intent from keyword analysis
        best_intent = max(keyword_scores.items(), key=lambda x: x[1])[0]
        print(f"\n📌 [DEBUG] Using keyword-based classification: {best_intent}")
        return best_intent

    except Exception as e:
        print(f"\n⚠️ [WARN] GPT classification failed: {e}")
        # Fallback to keyword-based classification
        best_intent = max(keyword_scores.items(), key=lambda x: x[1])[0]
        print(f"\n📌 [DEBUG] Using keyword-based classification: {best_intent}")
        return best_intent


def get_intent_category(intent: str) -> str:
    """
    Get the category for a given intent.

    Args:
        intent (str): The intent label

    Returns:
        str: The category name
    """
    for category, intents in INTENT_CATEGORIES.items():
        if intent in intents:
            return category
    return "other"


def analyze_prompt_context(prompt: str) -> Dict[str, any]:
    """
    Analyze the context of a prompt to provide additional insights.

    Args:
        prompt (str): The input prompt

    Returns:
        Dict[str, any]: Dictionary containing context analysis
    """
    intent = classify_prompt_intent(prompt)
    category = get_intent_category(intent)
    keyword_scores = calculate_intent_scores(prompt)

    return {
        "intent": intent,
        "category": category,
        "keyword_scores": keyword_scores,
        "confidence": max(keyword_scores.values()),
    }
