"""The index replaces a 1.9 GB dependency in the feature path, so these check
equivalence rather than plausibility."""
from __future__ import annotations

import gzip
import json

import pytest

from bl_ranking import REPO_ROOT
from bl_ranking.name_index import (
    INDEX_FILENAME,
    NameGenderIndex,
    _answer_from_gender_dict,
    _synthesise_gender_dict,
    install_shim,
)

INDEX_PATH = REPO_ROOT / "assets" / INDEX_FILENAME


# --------------------------------------------------------------------------
# The synthesis arithmetic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.5, 0.501, 0.734, 0.9, 0.987, 0.999, 1.0])
@pytest.mark.parametrize("is_male", [True, False])
def test_synthesised_counts_reproduce_label_and_confidence(is_male, confidence):
    """Round-trip through the vendored arithmetic, not a reimplementation."""
    label, got = _answer_from_gender_dict(_synthesise_gender_dict(is_male, confidence))
    assert label == ("male" if is_male else "female")
    assert got == pytest.approx(confidence)


def test_female_at_exactly_half_does_not_flip_to_male():
    """A 50.01% female name rounds to 0.500; reconstructing 500/500 would
    satisfy the vendored `m >= f` and report male, inverting the feature."""
    counts = _synthesise_gender_dict(is_male=False, confidence=0.5)
    assert counts["Female"] > counts["Male"]
    assert _answer_from_gender_dict(counts) == ("female", 0.5)


def test_unknown_cases_match_the_vendored_contract():
    assert _answer_from_gender_dict(None) == ("unknown", 0.0)
    assert _answer_from_gender_dict({}) == ("unknown", 0.0)
    assert _answer_from_gender_dict({"Male": 0, "Female": 0}) == ("unknown", 0.0)


def test_male_wins_ties_exactly_as_the_original_does():
    # The vendored branch is `if m >= f`, so an even split is male.
    assert _answer_from_gender_dict({"Male": 10, "Female": 10}) == ("male", 0.5)


# --------------------------------------------------------------------------
# Index object
# --------------------------------------------------------------------------


def test_index_roundtrip_and_normalisation(tmp_path):
    index = NameGenderIndex({}, {"Maria": 0.987}, source_version="test")
    path = tmp_path / "idx.json.gz"
    index.save(path)
    loaded = NameGenderIndex.load(path)

    def answer(name):
        return _answer_from_gender_dict(loaded.gender_dict_for(name))

    # Lookup must apply the same .strip().title() the real search() applies.
    assert answer("  maria  ") == answer("MARIA") == ("female", 0.987)
    assert answer("definitely-not-a-name") == ("unknown", 0.0)


def test_load_rejects_an_unknown_format(tmp_path):
    path = tmp_path / "bad.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"format": 99, "male": {}, "female": {}}, fh)
    with pytest.raises(ValueError, match="unsupported name index format"):
        NameGenderIndex.load(path)


# --------------------------------------------------------------------------
# The shipped artifact
# --------------------------------------------------------------------------


@pytest.mark.skipif(not INDEX_PATH.exists(), reason="index artifact not built")
def test_shipped_index_is_present_and_substantial():
    index = NameGenderIndex.load(INDEX_PATH)
    # ~714k in the real dataset; far fewer means a truncated build.
    assert index.n_names > 500_000
    assert INDEX_PATH.stat().st_size < 20_000_000, "index has grown past what belongs in git"


@pytest.mark.skipif(not INDEX_PATH.exists(), reason="index artifact not built")
def test_vendored_function_returns_real_answers_through_the_shim():
    """The unmodified vendored method, served by the index."""
    from bl_ranking.original.bl_exp_payout_predictor import BLPayoutModelsPredict

    instance = BLPayoutModelsPredict.__new__(BLPayoutModelsPredict)
    assert instance.detect_gender_with_confidence("Maria")[0] == "female"
    assert instance.detect_gender_with_confidence("Rigoberto")[0] == "male"
    assert instance.detect_gender_with_confidence("Zzzzqqqx") == ("unknown", 0.0)


@pytest.mark.skipif(not INDEX_PATH.exists(), reason="index artifact not built")
def test_index_matches_the_real_library_on_a_sample():
    """Spot-check against the genuine dataset. A subprocess is required: this
    process has the shim in ``sys.modules``."""
    import subprocess
    import sys

    script = f"""
import os, sys, random
os.environ["BL_DISABLE_NAME_INDEX"] = "1"
sys.path.insert(0, {str(REPO_ROOT / "src")!r})
from names_dataset import NameDataset
from bl_ranking.name_index import NameGenderIndex, _answer_from_gender_dict
nd = NameDataset(load_first_names=True, load_last_names=False)
idx = NameGenderIndex.load({str(INDEX_PATH)!r})
keys = list(nd.first_names.keys())
random.Random(7).shuffle(keys)
bad = []
for k in keys[:4000]:
    r = nd.search(k)
    first = r.get("first_name")
    truth = _answer_from_gender_dict(first.get("gender") if first else None)
    served = _answer_from_gender_dict(idx.gender_dict_for(k))
    if truth != served:
        bad.append((k, truth, served))
print("MISMATCHES", len(bad), bad[:3])
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        pytest.skip(f"real names_dataset unavailable in a clean process: {proc.stderr[-300:]}")
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("MISMATCHES")][-1]
    assert line.startswith("MISMATCHES 0"), f"index disagrees with the real library: {line}"


def test_missing_index_is_loud_when_required(tmp_path):
    """A missing artifact must never silently cost 1.9 GB in production."""
    with pytest.raises(RuntimeError, match="name gender index not found"):
        install_shim(tmp_path / "nope.json.gz", required=True)
