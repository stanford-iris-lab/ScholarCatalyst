# Model registry and provider routing.
from __future__ import annotations


PRICING: dict[str, dict] = {
    # OpenAI
    "gpt-6-astra":              {"provider": "openai"},
    "gpt-5.6-sol":              {"provider": "openai"},
    "gpt-5.6-terra":            {"provider": "openai"},
    "gpt-5.6-luna":             {"provider": "openai"},
    "gpt-5.4":                  {"provider": "openai"},
    "gpt-5.4-mini":             {"provider": "openai"},
    "gpt-4.1":                  {"provider": "openai"},  # pre-2025 cutoff (2024-06-01) backbone
    "gpt-4.1-mini":             {"provider": "openai"},
    "gpt-4.1-nano":             {"provider": "openai"},
    "o3":                       {"provider": "openai"},
    "o4-mini":                  {"provider": "openai"},

    # Anthropic
    "claude-fable-5-1":         {"provider": "anthropic"},
    "claude-opus-5":            {"provider": "anthropic"},
    "claude-sonnet-5":          {"provider": "anthropic"},
    "claude-haiku-4-5":         {"provider": "anthropic"},
    "claude-opus-4-7":          {"provider": "anthropic"},
    "claude-sonnet-4-6":        {"provider": "anthropic"},

    # Gemini
    "gemini-2.5-pro":           {"provider": "google"},
    "gemini-2.5-flash":         {"provider": "google"},
    "gemini-3.1-flash-lite":    {"provider": "google"},
    "gemini-3-flash-preview":   {"provider": "google"},
    "gemini-3.1-pro-preview":   {"provider": "google"},
    "gemini-3.5-flash":         {"provider": "google"},
    "gemini-3.6-flash":         {"provider": "google"},
    "gemini-3.7-flash":         {"provider": "google"},
}


MODEL_ALIASES: dict[str, str] = {
    # legacy shortcuts
    "gpt5":             "gpt-5.4",
    "gpt-5":            "gpt-5.4",
    "opus":             "claude-opus-4-7",
    "sonnet":           "claude-sonnet-4-6",
    # claude-fable-5 routes the same as claude-fable-5-1
    "claude-fable-5":   "claude-fable-5-1",
    # vendor-prefixed CLI ids
    "google/gemini-2.5-pro":         "gemini-2.5-pro",
    "google/gemini-2.5-flash":       "gemini-2.5-flash",
    "google/gemini-3.1-flash-lite":  "gemini-3.1-flash-lite",
    "google/gemini-3-flash-preview": "gemini-3-flash-preview",
    "google/gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "google/gemini-3.5-flash":       "gemini-3.5-flash",
    "google/gemini-3.6-flash":       "gemini-3.6-flash",
    "google/gemini-3.7-flash":       "gemini-3.7-flash",
    "gemini-3.1-pro":                "gemini-3.1-pro-preview",
    "anthropic/claude-fable-5":      "claude-fable-5-1",
    "anthropic/claude-fable-5-1":    "claude-fable-5-1",
    "anthropic/claude-opus-5":       "claude-opus-5",
    "anthropic/claude-sonnet-5":     "claude-sonnet-5",
    "anthropic/claude-haiku-4-5":    "claude-haiku-4-5",
    "anthropic/claude-opus-4-7":     "claude-opus-4-7",
    "anthropic/claude-sonnet-4-6":   "claude-sonnet-4-6",
    "anthropic/claude-fable-5.1":    "claude-fable-5-1",
    "openai/gpt-6-astra":            "gpt-6-astra",
    "openai/gpt-5.6-sol":            "gpt-5.6-sol",
    "openai/o3":                     "o3",
    "openai/o4-mini":                "o4-mini",
}


def resolve_model(name: str) -> str:
    if name in PRICING:
        return name
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    # provider-prefixed ids (google/..., anthropic/...) not covered by an
    # explicit alias still resolve as their bare form
    if "/" in name and name.split("/", 1)[1] in PRICING:
        return name.split("/", 1)[1]
    raise ValueError(f"unknown model: {name!r}. Known: {sorted(PRICING)}")


def all_models() -> list[str]:
    return list(PRICING)


def provider_of(model: str) -> str:
    return PRICING[resolve_model(model)]["provider"]


FILENAME_ALIASES_OVERRIDE: dict[str, str] = {
    "claude-opus-4-7":   "opus-4.7",
    "claude-sonnet-4-6": "sonnet-4.6",
}


def filename_alias(model: str) -> str:
    if "/" in model:
        model = model.split("/", 1)[1]
    if model in FILENAME_ALIASES_OVERRIDE:
        return FILENAME_ALIASES_OVERRIDE[model]
    if model.endswith("-preview"):
        model = model[: -len("-preview")]
    return model
