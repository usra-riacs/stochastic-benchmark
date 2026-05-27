from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_doc(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_readme_documents_repeat_reliability_boundaries():
    readme = " ".join(read_doc("README.md").split())

    for required in [
        "Noori, Moslem",
        "Physical Review Applied 25",
        "cross-instance Window Sticker uncertainty",
        "Per-instance repeat-count reliability",
        "Bernoulli success events",
        "`R_c`",
        "RTT/TTS",
        "CETS",
        "thresholded continuous metrics",
        "continuous Response curves",
        "continuous PerfRatio curves",
    ]:
        assert required in readme


def test_general_workflow_maps_repeat_reliability_criticisms():
    workflow = " ".join(read_doc("examples/general_workflow.md").split())

    for criticism in [
        "Repeat-count sufficiency",
        "Bootstrap-only uncertainty",
        "Noisy HPO choices",
        "CI-overlap ambiguity",
        "Virtual-best optimism",
    ]:
        assert criticism in workflow

    assert "cross-instance Window Sticker uncertainty" in workflow
    assert "Bernoulli success event" in workflow
    assert "Continuous Response curves and continuous PerfRatio curves" in workflow
