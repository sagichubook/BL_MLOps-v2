"""Compact stand-in for the ``names_dataset`` runtime.

Both vendored scripts build ``NameDataset()`` at import for one lookup, first
name -> (gender, confidence). Measured: 1,914 MB RSS and 5.6 s per process,
plus a deepcopy and pycountry lookups on each of the 11 calls a request makes.

``build_index()`` distils the finished answer offline; ``install_shim()``
installs it as ``names_dataset`` before the vendored import runs. 169 MB,
0.6 s, 1 us.

Equivalence is proven, not assumed. The index stores (is_male, confidence) and
the shim synthesises counts that drive the *unmodified* vendored arithmetic to
the same result -- the trap being that it branches on ``m >= f`` but reports
``round(x, 3)``, so a 50.01% female name rounds to 0.5 and a naive 500/500
reconstruction would flip it to male. ``build_index()`` checks every name in
the source dataset and refuses to write an index that disagrees anywhere.
"""
from __future__ import annotations

import gzip
import json
import logging
import sys
import types
from pathlib import Path

logger = logging.getLogger("bl_ranking.name_index")

# Denominator used to synthesise gender counts from a stored confidence.
# Large enough that a 3-decimal confidence reconstructs exactly, and that the
# one-count bias used to break a rounded 0.500 tie stays invisible to
# round(x, 3).
_SYNTH_TOTAL = 1_000_000

INDEX_FILENAME = "name_gender_index.json.gz"


def _unknown() -> tuple[str, float]:
    return ("unknown", 0.0)


def _answer_from_gender_dict(gender_data: dict | None) -> tuple[str, float]:
    """Byte-for-byte reimplementation of the vendored
    ``detect_gender_with_confidence`` tail, operating on the ``gender`` dict
    that ``NameDataset.search()[...]['gender']`` returns."""
    if gender_data is None:
        return _unknown()
    m = gender_data.get("Male", 0)
    f = gender_data.get("Female", 0)
    total = m + f
    if total == 0:
        return _unknown()
    if m >= f:
        return ("male", round(m / total, 3))
    return ("female", round(f / total, 3))


def _synthesise_gender_dict(is_male: bool, confidence: float) -> dict[str, int]:
    """Counts that drive the vendored arithmetic to ``(label, confidence)``.

    Male: ``m = confidence * T``, so ``m >= f`` holds (confidence >= 0.5
    whenever the real data said male) and ``round(m / T, 3)`` returns the
    stored value.

    Female: mirrored -- but when the stored confidence is exactly ``0.5``
    (a real 50.01% female name, rounded), a plain mirror would produce
    ``f == m`` and the vendored ``m >= f`` branch would report *male*. One
    extra count on the female side restores the branch; the resulting ratio
    0.500001 still rounds to 0.5, so the reported confidence is unchanged.
    """
    primary = round(confidence * _SYNTH_TOTAL)
    other = _SYNTH_TOTAL - primary
    if is_male:
        return {"Male": primary, "Female": other}
    if primary <= other:
        primary, other = other + 1, other - 1
    return {"Male": other, "Female": primary}


class NameGenderIndex:
    """First name -> ``(gender, confidence)``, loaded from the built artifact."""

    __slots__ = ("_male", "_female", "source_version", "n_names")

    def __init__(self, male: dict[str, float], female: dict[str, float], source_version: str = "unknown") -> None:
        self._male = male
        self._female = female
        self.source_version = source_version
        self.n_names = len(male) + len(female)

    def gender_dict_for(self, name: str) -> dict[str, int] | None:
        """Synthetic counts for the vendored arithmetic, or None if unknown."""
        key = name.strip().title()
        conf = self._male.get(key)
        if conf is not None:
            return _synthesise_gender_dict(True, conf)
        conf = self._female.get(key)
        if conf is not None:
            return _synthesise_gender_dict(False, conf)
        return None

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": 1,
            "source_version": self.source_version,
            "synth_total": _SYNTH_TOTAL,
            "male": self._male,
            "female": self._female,
        }
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as fh:
            json.dump(payload, fh, separators=(",", ":"))

    @classmethod
    def load(cls, path: Path) -> NameGenderIndex:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != 1:
            raise ValueError(f"unsupported name index format: {payload.get('format')!r}")
        return cls(payload["male"], payload["female"], payload.get("source_version", "unknown"))


# --------------------------------------------------------------------------
# Offline build (never runs in the serving path)
# --------------------------------------------------------------------------


def build_index(verify: bool = True) -> NameGenderIndex:
    """Distil the real dataset to its answer set, verifying every name."""
    from names_dataset import NameDataset  # heavy import, offline only

    try:
        from importlib.metadata import version as _pkg_version

        source_version = _pkg_version("names-dataset")
    except Exception:  # pragma: no cover - provenance is best-effort
        source_version = "unknown"

    logger.info("loading real names_dataset (first names only) to build the index")
    # load_last_names=False: the vendored code only ever reads ['first_name'].
    nd = NameDataset(load_first_names=True, load_last_names=False)

    male: dict[str, float] = {}
    female: dict[str, float] = {}
    for key, record in nd.first_names.items():
        raw = record.get("gender") or {}
        # Raw records use 'M'/'F'; _post_process() renames them before the
        # vendored function sees them. Normalise to the post-processed shape.
        gender_data = {"Male": raw.get("M", 0), "Female": raw.get("F", 0)}
        label, confidence = _answer_from_gender_dict(gender_data)
        if label == "male":
            male[key] = confidence
        elif label == "female":
            female[key] = confidence
        # 'unknown' names are omitted: absence already means ('unknown', 0.0).

    index = NameGenderIndex(male, female, source_version)
    logger.info("index built: %s names with a gender answer", index.n_names)

    if verify:
        mismatches = _verify_against_real(nd, index)
        if mismatches:
            raise RuntimeError(
                f"name index disagrees with names_dataset on {len(mismatches)} name(s); "
                f"first few: {mismatches[:5]}"
            )
        logger.info("index verified against every name in names_dataset: exact match")

    return index


def _verify_against_real(nd, index: NameGenderIndex, sample: int = 3000) -> list[str]:
    """Check the index against the library's own lookup semantics.

    Must be apples-to-apples: ``search(q)`` looks up ``q.strip().title()``, so
    a key that is not title-stable (``'Ami\u0307l'``) is unreachable in the
    real library too, and comparing a raw record against a normalised lookup
    would flag agreement as mismatch. Pass 1 covers every key cheaply; pass 2
    drives a sample through the genuine ``search()``, so a change to the
    library's post-processing cannot slip past.
    """
    import random

    keys = list(nd.first_names.keys())
    mismatches: list[str] = []

    # -- pass 1: every key, via replicated search() semantics ---------------
    for key in keys:
        record = nd.first_names.get(key.strip().title())
        if record is None:
            truth = _unknown()
        else:
            raw = record.get("gender") or {}
            truth = _answer_from_gender_dict({"Male": raw.get("M", 0), "Female": raw.get("F", 0)})
        served = _answer_from_gender_dict(index.gender_dict_for(key))
        if truth != served:
            mismatches.append(key)
            if len(mismatches) > 50:
                return mismatches

    # -- pass 2: a sample through the library's genuine search() ------------
    rng = random.Random(1729)
    for key in rng.sample(keys, min(sample, len(keys))):
        result = nd.search(key)
        first = result.get("first_name")
        truth = _answer_from_gender_dict(first.get("gender") if first else None)
        served = _answer_from_gender_dict(index.gender_dict_for(key))
        if truth != served:
            mismatches.append(f"{key} (via real search())")
            if len(mismatches) > 50:
                return mismatches

    return mismatches


# --------------------------------------------------------------------------
# Runtime shim
# --------------------------------------------------------------------------


class _ShimNameDataset:
    """``NameDataset`` as the vendored scripts use it: ``search()`` only.
    Anything else raises rather than silently answering wrong."""

    def __init__(self, load_first_names: bool = True, load_last_names: bool = True) -> None:
        self._index = _ACTIVE_INDEX
        if self._index is None:  # pragma: no cover - guarded by install_shim
            raise RuntimeError("name index shim installed without an index")

    def search(self, name: str) -> dict:
        gender = self._index.gender_dict_for(name)
        if gender is None:
            return {"first_name": None, "last_name": None}
        return {"first_name": {"gender": gender}, "last_name": None}

    def __getattr__(self, item):  # pragma: no cover - defensive
        raise AttributeError(
            f"names_dataset shim implements only search(); {item!r} was requested. "
            "If the vendored code grows a new dependency on this library, rebuild "
            "the shim deliberately instead of falling through to wrong answers."
        )


_ACTIVE_INDEX: NameGenderIndex | None = None


def install_shim(index_path: Path, *, required: bool = False) -> bool:
    """Install the index as ``names_dataset``, before the vendored imports.

    Without the artifact this degrades to the real 1.9 GB library: a missing
    build output should make the process fat, never wrong. Serving passes
    ``required=True`` because paying that silently per worker is a failure.
    """
    global _ACTIVE_INDEX

    # Validate the path before the idempotency check: `required=True` is an
    # assertion about *this* path, so a caller naming a nonexistent index must
    # be told, even if some earlier call already installed a different one.
    if not Path(index_path).exists():
        message = (
            f"name gender index not found at {index_path}; falling back to the real "
            "names_dataset (~1.9 GB RSS, ~6 s startup). Build it with "
            "`python scripts/build_name_index.py`."
        )
        if required:
            raise RuntimeError(message)
        logger.warning(message)
        return False

    existing = sys.modules.get("names_dataset")
    if existing is not None and getattr(existing, "_bl_ranking_shim", False):
        return True  # already installed by an earlier call in this process

    _ACTIVE_INDEX = NameGenderIndex.load(Path(index_path))

    module = types.ModuleType("names_dataset")
    module.NameDataset = _ShimNameDataset  # type: ignore[attr-defined]
    module._bl_ranking_shim = True  # type: ignore[attr-defined]
    module.__doc__ = "bl_ranking compact shim — see src/bl_ranking/name_index.py"
    sys.modules["names_dataset"] = module

    logger.info(
        "name_index_shim_installed",
        extra={"n_names": _ACTIVE_INDEX.n_names, "source_version": _ACTIVE_INDEX.source_version},
    )
    return True


def active_index() -> NameGenderIndex | None:
    return _ACTIVE_INDEX
