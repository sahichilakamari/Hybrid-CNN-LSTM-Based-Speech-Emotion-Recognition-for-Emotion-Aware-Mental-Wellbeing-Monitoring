# utils/emotions.py

# ============================================================
# EMOTION LABEL STANDARDIZATION (5-CLASS SETUP)
# ============================================================

EMOTION_MAP = {
    # Core emotions
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fearful": 4,

    # Mapped emotions (dataset normalization)
    "calm": 0,        # Calm merged into Neutral
    "disgust": 3,     # Disgust → Angry
    "surprised": 4    # Surprise → Fearful
}

EMOTION_LABELS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful"
]

EMOTION_DISPLAY_NAMES = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Fearful"
}

EMOTION_COLORS = {
    0: "#6c757d",  # Neutral - Gray
    1: "#ffc107",  # Happy - Yellow
    2: "#0d6efd",  # Sad - Blue
    3: "#dc3545",  # Angry - Red
    4: "#6f42c1"   # Fearful - Purple
}

# ============================================================
# EMOTION SEVERITY (Used for stress & trend analysis)
# ============================================================

EMOTION_SEVERITY = {
    0: 0.1,  # Neutral
    1: 0.2,  # Happy
    2: 0.6,  # Sad
    3: 0.8,  # Angry
    4: 0.9   # Fearful
}

# ============================================================
# SMART RECOMMENDATION ENGINE
# ============================================================

def get_recommendation(emotion_id, confidence, history=None):
    """
    Generate adaptive recommendations based on:
    - Predicted emotion
    - Confidence score
    - Optional emotion history (time-based trends)
    """

    response = {
        "emotion": EMOTION_DISPLAY_NAMES.get(emotion_id, "Unknown"),
        "confidence": round(float(confidence), 3),
        "stress_score": EMOTION_SEVERITY.get(emotion_id, 0.0),
        "recommendation": "",
        "suggestions": []
    }

    # --------------------------------------------------------
    # Low confidence fallback
    # --------------------------------------------------------
    if confidence < 0.55:
        response["recommendation"] = (
            "Emotion detection confidence is low. Please try again in a quieter environment."
        )
        response["suggestions"] = [
            "Reduce background noise",
            "Speak clearly and steadily",
            "Use a good quality microphone"
        ]
        return response

    # --------------------------------------------------------
    # Emotion-based recommendations
    # --------------------------------------------------------
    if emotion_id == 0:  # Neutral
        response["recommendation"] = "Your emotional state appears stable and balanced."
        response["suggestions"] = [
            "Maintain your routine",
            "Practice mindfulness",
            "Continue your current activities"
        ]

    elif emotion_id == 1:  # Happy
        response["recommendation"] = "You sound positive and upbeat."
        response["suggestions"] = [
            "Engage in social interactions",
            "Channel your energy into productive tasks",
            "Share positivity with others"
        ]

    elif emotion_id == 2:  # Sad
        response["recommendation"] = "You seem emotionally low. Support may help."
        response["suggestions"] = [
            "Talk to someone you trust",
            "Listen to calming or uplifting music",
            "Take a short walk or break"
        ]

    elif emotion_id == 3:  # Angry
        response["recommendation"] = "You appear tense or frustrated."
        response["suggestions"] = [
            "Pause and take deep breaths",
            "Step away from stressful situations",
            "Try relaxation exercises"
        ]

    elif emotion_id == 4:  # Fearful
        response["recommendation"] = "Signs of anxiety or fear detected."
        response["suggestions"] = [
            "Practice grounding techniques",
            "Focus on slow breathing",
            "Seek reassurance or professional support if needed"
        ]

    # --------------------------------------------------------
    # Trend-based escalation (optional history)
    # --------------------------------------------------------
    if history and len(history) >= 5:
        negative_count = sum(1 for e in history[-5:] if e in [2, 3, 4])
        if negative_count >= 4:
            response["recommendation"] += (
                " Prolonged emotional stress detected."
            )
            response["suggestions"].append(
                "Consider reaching out to a counselor or trusted person"
            )

    return response
