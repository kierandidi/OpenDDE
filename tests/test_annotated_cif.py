import gzip
from pathlib import Path

import pytest
from biotite.sequence import ProteinSequence
from biotite.structure.io.pdbx import CIFCategory, CIFFile

from opendde.data.inference.json_maker import annotated_cif_to_input_json, ccd


def test_annotated_cif_resolves_chain_files_and_preserves_disulfide(
    tmp_path, monkeypatch
):
    # This fixture contains only standard amino acids. Keep the unit test independent of a model installation.
    monkeypatch.setattr(ccd, "get_one_letter_code", ProteinSequence.convert_letter_3to1)
    monkeypatch.setattr(ccd, "get_mol_type", lambda _: "protein")
    cif = CIFFile.read(Path(__file__).resolve().parents[1] / "examples/2lwu.cif")
    source = tmp_path / "fixed.cif"
    cif.write(source)
    with pytest.raises(ValueError, match="cyclic"):
        annotated_cif_to_input_json(source)

    # Keep the deposited disulfide, leaving an open backbone.
    links = cif.block["struct_conn"]
    keep = links["conn_type_id"].as_array() == "disulf"
    cif.block["struct_conn"] = CIFCategory(
        {key: col.as_array()[keep] for key, col in links.items()}
    )
    cif.block["template_hits_by_chain_id"] = CIFCategory({"A": ["hits.hhr"]})
    cif.block["msa_paths_by_chain_id"] = CIFCategory({"A": ["query.a3m.gz"]})
    (tmp_path / "hits.hhr").touch()
    with gzip.open(tmp_path / "query.a3m.gz", "wt") as out:
        out.write(">query\nGLC\n")
    cif.write(source)
    job = annotated_cif_to_input_json(source)
    chain = job["sequences"][0]["proteinChain"]
    assert chain["id"] == ["A"]
    assert chain["templatesPath"] == str(tmp_path / "hits.hhr")
    assert chain["unpairedMsa"] == ">query\nGLC\n"
    (bond,) = job["covalent_bonds"]
    assert {bond["position1"], bond["position2"]} == {3, 12}
    assert bond["atom1"] == bond["atom2"] == "SG"

    cif.block["template_hits_by_chain_id"] = CIFCategory({"B": ["hits.hhr"]})
    cif.write(source)
    with pytest.raises(ValueError, match="chain B"):
        annotated_cif_to_input_json(source)
