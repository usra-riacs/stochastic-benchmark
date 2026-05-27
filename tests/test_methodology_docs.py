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
        "https://link.springer.com/article/10.1007/s42484-025-00311-2",
        "https://arxiv.org/abs/2402.10255",
        "https://doi.org/10.1103/PhysRevApplied.25.034081",
        "https://arxiv.org/abs/2503.16589",
        "https://doi.org/10.1145/3678184",
        "https://arxiv.org/abs/2302.02278",
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
    assert "https://doi.org/10.1145/3678184" in workflow
    assert "https://arxiv.org/abs/2302.02278" in workflow


def test_noori_validation_doc_links_published_and_arxiv_versions():
    validation_doc = read_doc("docs/noori_repeat_reliability_validation.md")

    assert "https://doi.org/10.1103/PhysRevApplied.25.034081" in validation_doc
    assert "https://arxiv.org/abs/2503.16589" in validation_doc
