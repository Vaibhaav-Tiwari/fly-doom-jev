"""Typed Jev question bank (full spec bank, section 5).

Each question maps to a named probability in the Jev decision. Questions are
typed concepts, not free-form prose; the same bank is used by the mock and the
live client. For the live client, every question also carries a TypeSafe
System One question spec (SYSTEMONE_QUESTIONS) — noul (yes/no probability),
score (rubric expectation, normalized to 0..1), or choice (winner probability).
"""

from __future__ import annotations

QUESTION_BANK: dict[str, str] = {
    "ATTACK": "Probability that attacking now is appropriate.",
    "RETREAT": "Probability that retreating / increasing distance is appropriate.",
    "EXPLORE": "Probability that exploring / advancing to search is appropriate.",
    "REPOSITION": "Probability that repositioning (strafe/turn to a better angle) is appropriate.",
    "SEEK_AMMO": "Probability that seeking ammunition/supplies is the priority.",
    "THREAT_LEVEL": "Estimated threat level of the current situation (0-1).",
    "ENEMY_PRESENT": "Probability that an enemy is currently present/visible.",
    "MOVEMENT_INTENT": "Confidence in the best immediate movement (choice).",
    "TARGET_PRIORITY": "Confidence in which enemy to prioritize (choice).",
    "ENGAGEMENT_CONFIDENCE": "Confidence that engaging now will succeed (0-1).",
    "INTENT": "Strategic intent for the next ~1-2 s (choice; persists between calls).",
}

# System One question specs, keyed by the same names. criteria reference the
# structured EnvironmentState fields the API receives as `state`.
SYSTEMONE_QUESTIONS: dict[str, dict] = {
    "ATTACK": {
        "type": "noul",
        "instructions": "Is attacking the enemy right now the appropriate action? "
                        "The state fields are normalized: enemy_distance 0=point-blank, "
                        "1=very far; enemy_angle 0=dead-center in view.",
        "criteria": {
            "true": "enemy_visible is true AND enemy_distance < 0.4 (in weapon range) "
                    "AND |enemy_angle| < 0.3 (roughly aimed) AND ammo > 0",
            "false": "enemy_visible is false, OR enemy_distance >= 0.4 (too far), "
                     "OR the enemy is far off to the side (|enemy_angle| >= 0.3), "
                     "OR ammo is 0",
        },
    },
    "RETREAT": {
        "type": "noul",
        "instructions": "Is retreating (increasing distance from the threat) appropriate right now?",
        "criteria": {
            "true": "Health is low, threat_level is high, or ammo is depleted while an enemy is near",
            "false": "Healthy and armed, or no immediate threat",
        },
    },
    "EXPLORE": {
        "type": "noul",
        "instructions": "Is exploring (advancing to search the area) appropriate right now?",
        "criteria": {
            "true": "No enemy visible and the situation is calm",
            "false": "An enemy is visible or threat_level is high",
        },
    },
    "ENEMY_PRESENT": {
        "type": "noul",
        "instructions": "Is an enemy currently present in view?",
        "criteria": {
            "true": "enemy_visible is true",
            "false": "enemy_visible is false",
        },
    },
    "REPOSITION": {
        "type": "noul",
        "instructions": "Should the agent reposition (turn/strafe to a better angle) right now?",
        "criteria": {
            "true": "An enemy is visible but off-center (|enemy_angle| large) or the current position is disadvantageous",
            "false": "Enemy centered or no enemy; current position is fine",
        },
    },
    "SEEK_AMMO": {
        "type": "noul",
        "instructions": "Should the agent prioritize seeking ammunition or supplies right now?",
        "criteria": {
            "true": "ammo is zero or nearly depleted",
            "false": "ammo is sufficient for the current threat",
        },
    },
    "THREAT_LEVEL": {
        "type": "score",
        "instructions": "Rate the immediate threat level of the current situation.",
        "criteria": ["no threat: safe", "minor threat: enemy far or passive",
                     "moderate threat: enemy near or engaging",
                     "severe threat: about to die"],
    },
    "ENGAGEMENT_CONFIDENCE": {
        "type": "score",
        "instructions": "Rate the confidence that engaging the enemy right now will succeed.",
        "criteria": ["hopeless: no ammo or enemy overwhelming", "low: unfavorable odds",
                     "moderate: even odds", "high: clear advantage"],
    },
    "MOVEMENT_INTENT": {
        "type": "choice",
        "instructions": "Which single movement action best serves survival and scoring right now?",
        "criteria": {
            "forward": "move ahead (close distance or advance to search)",
            "hold": "stay in place",
            "strafe_left": "sidestep left without turning",
            "strafe_right": "sidestep right without turning",
            "turn_left": "rotate left to center a threat or search",
            "turn_right": "rotate right to center a threat or search",
        },
    },
    "TARGET_PRIORITY": {
        "type": "choice",
        "instructions": "Which target should be prioritized right now?",
        "criteria": {
            "nearest_enemy": "the closest visible enemy",
            "most_threatening": "the enemy posing the greatest danger",
            "supplies": "ammunition or health pickups",
            "none": "no target; keep searching",
        },
    },
    "INTENT": {
        # Strategy-layer question: asked at LOW cadence (~1.5 s, or on salient
        # events); the winning intent persists between calls and biases the
        # motor decoder. Jev = strategy, brain = reflexes.
        "type": "choice",
        "instructions": "Choose the strategic INTENT for the next 1-2 seconds of "
                        "play. You are the strategy layer of a fly-brain controller "
                        "playing DOOM; fast reflexes (aiming, firing) are handled "
                        "by the brain between your calls, so pick a posture, not "
                        "individual actions. State fields: enemy_in_view, "
                        "aim_offset_deg (0 = centered), enemy_distance (0 = "
                        "point-blank, 1 = far), health, ammo, "
                        "recent_damage_taken (hp lost in the last second), kills, "
                        "alive_s. On E1M1, state.goal gives exit_dist_units and "
                        "exit_bearing_deg (0 = east, +90 = north) — the level is "
                        "CLEARED by reaching the exit lift, so when no enemy "
                        "threatens, prefer the posture that closes that distance. "
                        "state.memory.recent_episodes lists how your "
                        "recent episodes ended (persistent across restarts) — if "
                        "you keep dying fast, try a different posture.",
        "criteria": {
            "engage": "close with the visible enemy and fight",
            "retreat": "increase distance from the threat (low health or heavy "
                       "recent damage)",
            "circle": "keep turning to search / flank while staying mobile",
            "advance": "move forward to explore or close distance (no urgent "
                       "threat)",
            "attack_now": "enemy is in range and roughly centered — press the "
                          "attack immediately",
        },
    },
}

# score questions normalize the expected level by (len(criteria) - 1)
SCORE_MAX_LEVEL = {q: len(spec["criteria"]) - 1
                   for q, spec in SYSTEMONE_QUESTIONS.items()
                   if spec["type"] == "score"}
