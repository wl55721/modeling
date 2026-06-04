"""Two-stage YAML rule registration for fusion.

Step-4: replaces the legacy ``registry/platforms/__init__.py`` if/elif
dispatcher and the Python-registered ``registry/builtins.py``.  The new
flow is:

1. ``clear_rules()`` — wipe the registry.
2. Always load ``rules/_common.yaml`` (the migrated builtins).
3. If ``model_id`` resolves to a known model slug, load
   ``rules/<slug>.yaml``.

No platform-specific Python rules are loaded any more — every rule lives
in YAML.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from python.zrt.transform.fusion.registry import clear_rules, register_rule

from .yaml_rule_loader import _model_id_to_key, load_yaml_rules

logger = logging.getLogger(__name__)


# Default directory for built-in YAML rule files.
_RULES_DIR = Path(__file__).resolve().parent.parent / "rules"
_COMMON_FILENAME = "_common.yaml"

# Cache for coarsen rules extracted from model-specific YAML
_COARSEN_RULES_CACHE: dict[str, list] = {}


def initialize_rules(model_id: str = "") -> None:
    """Load all fusion rules for ``model_id``.

    Always loads ``_common.yaml`` first, then ``<model_slug>.yaml`` if a
    matching file exists in :data:`_RULES_DIR`.

    The registry is cleared before loading so the result is deterministic
    regardless of prior state.
    
    Also extracts and caches ``coarsen_rules`` section from model-specific
    YAML files for use by the coarsen post-processing step.
    """
    global _COARSEN_RULES_CACHE
    clear_rules()
    _COARSEN_RULES_CACHE.clear()

    common_path = _RULES_DIR / _COMMON_FILENAME
    if common_path.exists():
        for rule in load_yaml_rules(common_path):
            try:
                register_rule(rule)
            except ValueError as exc:
                logger.warning("Skipping common rule: %s", exc)

    if not model_id:
        return

    slug = _model_id_to_key(model_id)
    if not slug:
        return

    model_path = _RULES_DIR / f"{slug}.yaml"
    if model_path.exists():
        _load_yaml_with_coarsen_rules(model_path, slug)
        logger.info("Loaded model-specific rules from %s", model_path)
        return

    # Prefix-match fallback (parity with the old ``load_model_yaml_rules``):
    # e.g. model_id="deepseek_v4_lite" still loads deepseek_v4.yaml.
    prefix = slug.split("_")[0].replace("-", "")
    for path in sorted(_RULES_DIR.glob("*.yaml")):
        if path.name == _COMMON_FILENAME:
            continue
        stem = path.stem.replace("_", "").replace("-", "")
        if prefix and stem.startswith(prefix):
            _load_yaml_with_coarsen_rules(path, path.stem)
            logger.info("Loaded model-specific rules from %s", path)
            return


def _load_yaml_with_coarsen_rules(path: Path, cache_key: str) -> None:
    """Load fusion rules and extract coarsen_rules from a YAML file.
    
    The YAML file may contain a top-level ``coarsen_rules`` section with
    scope-based aggregation rules for post-fusion processing.
    """
    global _COARSEN_RULES_CACHE
    
    # Load fusion rules (the list at top level)
    for rule in load_yaml_rules(path):
        try:
            register_rule(rule)
        except ValueError as exc:
            logger.warning("Skipping model rule (%s): %s", cache_key, exc)
    
    # Extract coarsen_rules section if present
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_yaml = yaml.safe_load(f)
        
        if isinstance(raw_yaml, dict) and "coarsen_rules" in raw_yaml:
            coarsen_rules_raw = raw_yaml["coarsen_rules"]
            if isinstance(coarsen_rules_raw, list):
                # Keep as plain dicts (not ModuleFusionRule objects)
                coarsen_rules = [
                    rule_dict for rule_dict in coarsen_rules_raw
                    if isinstance(rule_dict, dict)
                ]
                _COARSEN_RULES_CACHE[cache_key] = coarsen_rules
                logger.info("Extracted %d coarsen rules from %s", 
                          len(coarsen_rules), path.name)
    except Exception as exc:
        logger.debug("Could not extract coarsen_rules from %s: %s", path, exc)


def get_coarsen_rules(model_id: str = "") -> list:
    """Get coarsen rules for ``model_id``.
    
    Returns cached coarsen rules (ModuleFusionRule objects) from the model-specific YAML file.
    Falls back to empty list if no rules are available.
    """
    if not model_id:
        return []
    
    slug = _model_id_to_key(model_id)
    if not slug:
        return []
    
    # Try exact match first
    if slug in _COARSEN_RULES_CACHE:
        return _COARSEN_RULES_CACHE[slug]
    
    # Try prefix match
    prefix = slug.split("_")[0].replace("-", "")
    for cache_key in _COARSEN_RULES_CACHE:
        cache_prefix = cache_key.replace("_", "").replace("-", "")
        if prefix and cache_prefix.startswith(prefix):
            return _COARSEN_RULES_CACHE[cache_key]
    
    return []


__all__ = ["initialize_rules", "get_coarsen_rules"]
