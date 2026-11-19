import pytest
from biotite.structure.io.pdbx import CIFCategory

from opendde.data.tools.rewrite_biotite import _parse_inter_residue_bonds


@pytest.mark.parametrize(("value", "order"), [(None, 1), ("?", 1), ("sing", 1), ("DOUB", 2), ("trip", 3)])
def test_struct_conn_preserves_explicit_order(value, order):
    atoms = CIFCategory({"label_atom_id": ["NZ", "C15"]})
    link = {"conn_type_id": ["covale"], "ptnr1_label_atom_id": ["NZ"], "ptnr2_label_atom_id": ["C15"]}
    if value is not None:
        link["pdbx_value_order"] = [value]
    assert _parse_inter_residue_bonds(atoms, CIFCategory(link)).as_array().tolist() == [[0, 1, order]]
