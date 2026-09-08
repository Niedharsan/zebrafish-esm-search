from pathlib import Path

import scientific_candidate_repair as repair


def _write_cache(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "aliases.txt").write_text(
        "\n".join(
            [
                "ZDB-GENE-1\tGATA binding protein 1a\tgata1a\tgata1\tSO:0000704",
                "ZDB-GENE-2\tT-cell acute lymphocytic leukemia 1\ttal1\ttal\tSO:0000704",
                "ZDB-GENE-3\tinsulin\tins\tinsulin\tSO:0000704",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "human_orthos.txt").write_text(
        "\n".join(
            [
                "ZDB-GENE-1\tgata1a\tGATA binding protein 1a\tGATA1\tname\t\t\t\t\t\t\t\t",
                "ZDB-GENE-2\ttal1\tT-cell acute lymphocytic leukemia 1\tTAL1\tname\t\t\t\t\t\t\t\t",
                "ZDB-GENE-3\tins\tinsulin\tINS\tname\t\t\t\t\t\t\t\t",
                "ZDB-GENE-4\tklf1\tKruppel like factor 1\tKLF1\tname\t\t\t\t\t\t\t\t",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "mouse_orthos.txt").write_text(
        "\n".join(
            [
                "ZDB-GENE-3\tins\tinsulin\tIns1\tname\t\t\t\t\t\t\t",
                "ZDB-GENE-3\tins\tinsulin\tIns2\tname\t\t\t\t\t\t\t",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "uniprot.txt").write_text(
        "\n".join(
            [
                "ZDB-GENE-5\tSO:0000704\tctnnb1\tQ00001",
                "ZDB-GENE-6\tSO:0000704\tlef1\tQ00002",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_normalize_plan_repairs_aliases_and_curated_orthology(monkeypatch, tmp_path):
    _write_cache(tmp_path)
    monkeypatch.setenv("ZFIN_CACHE_DIR", str(tmp_path))
    lookup = repair.ZfinLookup()

    plan = {
        "normalized_question": "test",
        "zebrafish_candidates": [
            {"gene": "gata1", "species": "zebrafish", "reason": "erythroid TF"},
            {"gene": "tal", "species": "zebrafish", "reason": "hematopoietic TF"},
            {"gene": "ins1", "species": "zebrafish", "reason": "beta-cell marker"},
        ],
        "reference_candidates": [
            {"gene": "KLF1", "species": "human", "reason": "erythroid TF"},
            {"gene": "INS", "species": "human", "reason": "beta-cell marker"},
        ],
    }

    normalized, trace = repair.normalize_plan(plan, lookup=lookup)
    genes = [x["gene"] for x in normalized["zebrafish_candidates"]]

    assert genes[:3] == ["gata1a", "tal1", "ins"]
    assert "klf1" in genes
    assert len([g for g in genes if g == "ins"]) == 1
    assert not trace["errors"]


def test_quickgo_annotation_evidence_returns_zebrafish_symbols(monkeypatch, tmp_path):
    _write_cache(tmp_path)
    monkeypatch.setenv("ZFIN_CACHE_DIR", str(tmp_path))
    lookup = repair.ZfinLookup()

    calls = []

    def fake_http_json(url, **kwargs):
        calls.append(url)
        return {
            "results": [
                {"geneProductId": "UniProtKB:Q00001", "symbol": "ctnnb1"},
                {"geneProductId": "UniProtKB:Q00002", "symbol": ""},
                {"geneProductId": "UniProtKB:Q00001", "symbol": "ctnnb1"},
            ]
        }

    evidence, errors = repair.quickgo_annotation_evidence(
        [{"source": "QuickGO", "go_id": "GO:1990907", "name": "beta-catenin-TCF complex"}],
        fake_http_json,
        lookup=lookup,
    )

    assert not errors
    assert len(evidence) == 1
    assert evidence[0]["annotation_candidates"] == ["ctnnb1", "lef1"]
    assert "taxonId=7955" in calls[0]
    assert "goId=GO%3A1990907" in calls[0]
