"""Typed Jev question bank (v1 subset).

Each question maps to a named probability in the Jev decision. Questions are
typed concepts, not free-form prose; the same bank is used by the mock and the
live client.
"""

from __future__ import annotations

QUESTION_BANK: dict[str, str] = {
    "ATTACK": "Given the environment state, probability that attacking now is appropriate.",
    "RETREAT": "Probability that retreating / increasing distance is appropriate.",
    "EXPLORE": "Probability that exploring / advancing to search is appropriate.",
    "THREAT_LEVEL": "Estimated threat level of the current situation (0-1).",
}
