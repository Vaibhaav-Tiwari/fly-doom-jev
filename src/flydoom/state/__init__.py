"""Structured environment state (typed, versioned)."""

from .encoder import SCHEMA_VERSION, EnvironmentState, encode_state

__all__ = ["SCHEMA_VERSION", "EnvironmentState", "encode_state"]
