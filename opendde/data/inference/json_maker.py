# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import argparse
import copy
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from biotite.structure import AtomArray, get_chain_starts, get_residue_starts

from opendde.data.constants import STD_RESIDUES
from opendde.data.core import ccd
from opendde.data.core.filter import Filter
from opendde.data.core.parser import AddAtomArrayAnnot, MMCIFParser
from opendde.data.utils import (
    get_lig_lig_bonds,
    get_ligand_polymer_bond_mask,
    get_polymer_polymer_bond,
)


def merge_covalent_bonds(
    covalent_bonds: list[dict], all_entity_counts: dict[str, int]
) -> list[dict]:
    """
    Merge covalent bonds with same entity and position.

    Args:
        covalent_bonds (list[dict]): A list of covalent bond dicts.
        all_entity_counts (dict[str, int]): A dict of entity id to chain count.

    Returns:
        list[dict]: A list of merged covalent bond dicts.
    """
    bonds_recorder = defaultdict(list)
    bonds_entity_counts = {}
    for bond_dict in covalent_bonds:
        bond_unique_string = [str(bond_dict.get("bondOrder", 1))]
        entity_counts = (
            all_entity_counts[str(bond_dict["entity1"])],
            all_entity_counts[str(bond_dict["entity2"])],
        )
        for i in range(2):
            for j in ["entity", "position", "atom"]:
                k = f"{j}{i + 1}"
                bond_unique_string.append(str(bond_dict[k]))
        bond_unique_string = "_".join(bond_unique_string)
        bonds_recorder[bond_unique_string].append(bond_dict)
        bonds_entity_counts[bond_unique_string] = entity_counts

    merged_covalent_bonds = []
    for k, v in bonds_recorder.items():
        counts1 = bonds_entity_counts[k][0]
        counts2 = bonds_entity_counts[k][1]
        if counts1 == counts2 == len(v):
            bond_dict_copy = copy.deepcopy(v[0])
            del bond_dict_copy["copy1"]
            del bond_dict_copy["copy2"]
            merged_covalent_bonds.append(bond_dict_copy)
        else:
            merged_covalent_bonds.extend(v)
    return merged_covalent_bonds


def atom_array_to_input_json(
    atom_array: AtomArray,
    parser: MMCIFParser,
    assembly_id: str | None = None,
    output_json: str | None = None,
    sample_name: str | None = None,
    save_entity_and_asym_id: bool = False,
    include_discont_poly_poly_bonds: bool = False,
) -> dict:
    """
    Convert a Biotite AtomArray to a dict that can be used as input to the model.

    Args:
        atom_array (AtomArray): Biotite Atom array.
        parser (MMCIFParser): Instantiated OpenDDE MMCIFParer.
        assembly_id (str, optional): Assembly ID. Defaults to None.
        output_json (str, optional): Output json file path. Defaults to None.
        sample_name (str, optional): The "name" filed in json file. Defaults to None.
        save_entity_and_asym_id (bool, optional): Whether to save entity and asym ids to json.
                                                  Defaults to False.
        include_discont_poly_poly_bonds (bool, optional): Whether to include discontinuous polymer-polymer bonds.
                                                          Defaults to False.

    Returns:
        dict: OpenDDE input json dict.
    """
    # get sequences after modified AtomArray
    entity_seq = parser.get_sequences(atom_array)

    # add unique chain id
    atom_array = AddAtomArrayAnnot.unique_chain_and_add_ids(atom_array)

    # get lig entity sequences and position
    label_entity_id_to_sequences = {}
    lig_chain_ids = []  # record chain_id of the first asym chain
    for label_entity_id in np.unique(atom_array.label_entity_id):
        if label_entity_id not in parser.entity_poly_type:
            current_lig_chain_ids = np.unique(
                atom_array.chain_id[atom_array.label_entity_id == label_entity_id]
            ).tolist()
            lig_chain_ids += current_lig_chain_ids
            for chain_id in current_lig_chain_ids:
                lig_atom_array = atom_array[atom_array.chain_id == chain_id]
                starts = get_residue_starts(lig_atom_array, add_exclusive_stop=True)
                seq = lig_atom_array.res_name[starts[:-1]].tolist()
                label_entity_id_to_sequences[label_entity_id] = seq

    # find polymer modifications
    entity_id_to_mod_list = {}
    for entity_id, res_names in parser.get_poly_res_names(atom_array).items():
        modifications_list = []
        for idx, res_name in enumerate(res_names):
            if res_name not in STD_RESIDUES:
                position = idx + 1
                modifications_list.append([position, f"CCD_{res_name}"])
        if modifications_list:
            entity_id_to_mod_list[entity_id] = modifications_list

    chain_starts = get_chain_starts(atom_array, add_exclusive_stop=False)
    chain_starts_atom_array = atom_array[chain_starts]

    json_dict: dict[str, Any] = {
        "sequences": [],
    }
    if assembly_id is not None:
        json_dict["assembly_id"] = assembly_id

    unique_label_entity_id = np.unique(atom_array.label_entity_id)
    chain_id_to_copy_id_dict = {}
    for label_entity_id in unique_label_entity_id:
        chain_ids_in_entity = chain_starts_atom_array.chain_id[
            chain_starts_atom_array.label_entity_id == label_entity_id
        ]
        for chain_count, chain_id in enumerate(chain_ids_in_entity):
            chain_id_to_copy_id_dict[chain_id] = chain_count + 1
    copy_id = np.vectorize(chain_id_to_copy_id_dict.get)(atom_array.chain_id)
    atom_array.set_annotation("copy_id", copy_id)

    all_entity_counts = {}
    label_entity_id_to_entity_id_in_json = {}
    entity_idx = 0
    for label_entity_id in unique_label_entity_id:
        entity_dict = {}
        asym_chains = chain_starts_atom_array[
            chain_starts_atom_array.label_entity_id == label_entity_id
        ]
        entity_type = parser.entity_poly_type.get(label_entity_id, "ligand")
        if entity_type != "ligand":
            if entity_type == "polypeptide(L)":
                entity_type = "proteinChain"
            elif entity_type == "polydeoxyribonucleotide":
                entity_type = "dnaSequence"
            elif entity_type == "polyribonucleotide":
                entity_type = "rnaSequence"
            else:
                # DNA/RNA hybrid, polypeptide(D), etc.
                continue

            sequence = entity_seq.get(label_entity_id)
            entity_dict["sequence"] = sequence
        else:
            # ligand
            lig_ccd = "_".join(label_entity_id_to_sequences[label_entity_id])
            entity_dict["ligand"] = f"CCD_{lig_ccd}"
        entity_dict["count"] = len(asym_chains)
        entity_idx += 1
        entity_id_in_json = str(entity_idx)
        label_entity_id_to_entity_id_in_json[label_entity_id] = entity_id_in_json
        all_entity_counts[entity_id_in_json] = len(asym_chains)
        if save_entity_and_asym_id:
            entity_dict["label_entity_id"] = str(label_entity_id)
            entity_dict["label_asym_id"] = asym_chains.label_asym_id.tolist()

        # add PTM info
        if label_entity_id in entity_id_to_mod_list:
            modifications = entity_id_to_mod_list[label_entity_id]
            if entity_type == "proteinChain":
                entity_dict["modifications"] = [
                    {"ptmPosition": position, "ptmType": mod_ccd_code}
                    for position, mod_ccd_code in modifications
                ]
            else:
                entity_dict["modifications"] = [
                    {"basePosition": position, "modificationType": mod_ccd_code}
                    for position, mod_ccd_code in modifications
                ]

        json_dict["sequences"].append({entity_type: entity_dict})

    # skip some uncommon entities
    atom_array = atom_array[
        np.isin(
            atom_array.label_entity_id,
            list(label_entity_id_to_entity_id_in_json.keys()),
        )
    ]

    # add covalent bonds
    atom_array = AddAtomArrayAnnot.add_token_mol_type(
        atom_array, parser.entity_poly_type
    )
    lig_polymer_bonds = get_ligand_polymer_bond_mask(atom_array, lig_include_ions=False)
    lig_lig_bonds = get_lig_lig_bonds(atom_array, lig_include_ions=False)
    token_bonds = np.vstack((lig_polymer_bonds, lig_lig_bonds))

    lig_indices = np.where(np.isin(atom_array.chain_id, lig_chain_ids))[0]
    lig_bond_mask = np.any(np.isin(token_bonds[:, :2], lig_indices), axis=1)
    token_bonds = token_bonds[lig_bond_mask]  # select bonds of ligands

    if include_discont_poly_poly_bonds:
        polymer_polymer_bond = get_polymer_polymer_bond(
            atom_array, parser.entity_poly_type
        )
        token_bonds = np.vstack((polymer_polymer_bond, token_bonds))

    if token_bonds.size != 0:
        covalent_bonds = []
        for atoms in token_bonds:
            order = int(atoms[2])
            bond_dict = {"bondOrder": order} if order in (2, 3) else {}
            for i in range(2):
                positon = atom_array.res_id[atoms[i]]
                bond_dict[f"entity{i + 1}"] = int(
                    label_entity_id_to_entity_id_in_json[
                        atom_array.label_entity_id[atoms[i]]
                    ]
                )
                bond_dict[f"position{i + 1}"] = int(positon)
                bond_dict[f"atom{i + 1}"] = atom_array.atom_name[atoms[i]]
                bond_dict[f"copy{i + 1}"] = int(atom_array.copy_id[atoms[i]])

            covalent_bonds.append(bond_dict)

        # merge covalent_bonds for same entity
        merged_covalent_bonds = merge_covalent_bonds(covalent_bonds, all_entity_counts)
        json_dict["covalent_bonds"] = merged_covalent_bonds

    json_dict["name"] = sample_name

    if output_json is not None:
        with open(output_json, "w") as f:
            json.dump([json_dict], f, indent=4)
    return json_dict


def _read_chain_references(block, directory, chains):
    for category, field in (
        ("msa_paths_by_chain_id", "unpairedMsa"),
        ("template_hits_by_chain_id", "templatesPath"),
    ):
        if category not in block:
            continue
        for chain_id, column in block[category].items():
            allowed = (
                ("proteinChain", "rnaSequence")
                if field == "unpairedMsa"
                else ("proteinChain",)
            )
            if chain_id not in chains or chains[chain_id][1] not in allowed:
                raise ValueError(
                    f"{category} refers to an absent or incompatible chain {chain_id}"
                )
            ref = directory / column.as_item()
            entity = chains[chain_id][2]
            if field == "templatesPath":
                if ref.suffix not in {".a3m", ".hhr"} or not ref.is_file():
                    raise ValueError(
                        f"Chain {chain_id} needs a readable .a3m or .hhr template hit file: {ref}"
                    )
                entity[field] = str(ref.resolve())
            else:
                with (gzip.open if ref.suffix == ".gz" else open)(ref, "rt") as msa:
                    entity[field] = msa.read()


def _fixed_assembly_bonds(atoms, positions, chains):
    bonds = []
    edges = atoms.bonds.as_array()
    left_atom, right_atom = edges[:, 0], edges[:, 1]
    inter_residue = (atoms.chain_id[left_atom] != atoms.chain_id[right_atom]) | (
        positions[left_atom] != positions[right_atom]
    )
    for a, b, order in edges[inter_residue]:
        if positions[a] > positions[b]:
            a, b = b, a
        i, j = int(positions[a]), int(positions[b])
        chain_a, chain_b = atoms.chain_id[[a, b]]
        atom_a, atom_b = atoms.atom_name[[a, b]]
        left, kind, entity = chains[chain_a]
        right = chains[chain_b][0]
        if left == right:
            if kind == "proteinChain" and (i, j, atom_a, atom_b) == (
                1,
                len(entity["sequence"]),
                "N",
                "C",
            ):
                raise ValueError(
                    f"Head-to-tail cyclic chain {chain_a} needs a cyclic predictor runtime"
                )
            backbone = ("C", "N") if kind == "proteinChain" else ("O3'", "P")
            if (
                kind != "ligand"
                and j == i + 1
                and (atom_a, atom_b) == backbone
                and order in (0, 1)
            ):
                continue
        if order not in (0, 1, 2, 3):
            raise ValueError(f"Unsupported inter-residue bond order {order}")
        bond = {
            "entity1": left,
            "position1": i,
            "atom1": str(atom_a),
            "copy1": 1,
            "entity2": right,
            "position2": j,
            "atom2": str(atom_b),
            "copy2": 1,
        }
        if order > 1:
            bond["bondOrder"] = int(order)
        bonds.append(bond)
    return bonds


def annotated_cif_to_input_json(
    mmcif_file: str | Path, sample_name: str | None = None
) -> dict:
    """Read a fixed, annotated assembly. Per-chain references are relative to its CIF."""
    path = Path(mmcif_file)
    parser = MMCIFParser(str(path))
    block = parser.cif.block
    coords = block.get("condition_coordinate_1_atom")
    if coords is not None and any(
        np.isfinite(coords[f"val{i}"].as_array(float)).any() for i in range(3)
    ):
        raise ValueError(
            "Coordinate annotations cannot be used as deposited template hits"
        )
    atoms = parser.get_structure("first", model=1, bond_lenth_threshold=None)
    chain_types = {
        "polypeptide(L)": "proteinChain",
        "polyribonucleotide": "rnaSequence",
        "polydeoxyribonucleotide": "dnaSequence",
    }
    chains = {}
    sequences = []
    positions = np.empty(len(atoms), dtype=int)
    for chain_id in dict.fromkeys(atoms.chain_id):
        indices = np.flatnonzero(atoms.chain_id == chain_id)
        chain = atoms[indices]
        bounds = get_residue_starts(chain, add_exclusive_stop=True)
        codes = chain.res_name[bounds[:-1]].tolist()
        positions[indices] = np.repeat(np.arange(1, len(bounds)), np.diff(bounds))
        polymer = parser.entity_poly_type.get(chain.label_entity_id[0])
        if polymer is not None and polymer not in chain_types:
            raise ValueError(f"Unsupported polymer type {polymer} in chain {chain_id}")
        kind = chain_types.get(polymer, "ligand")
        entity = {"id": [str(chain_id)], "count": 1}
        if kind == "ligand":
            entity["ligand"] = "CCD_" + "_".join(codes)
        else:
            sequence = ccd.res_names_to_sequence(codes)
            protein = kind == "proteinChain"
            entity["sequence"] = sequence if protein else sequence.replace("X", "N")
            pos, code = (
                ("ptmPosition", "ptmType")
                if protein
                else ("basePosition", "modificationType")
            )
            modifications = [
                {pos: i, code: "CCD_" + res}
                for i, res in enumerate(codes, 1)
                if res not in STD_RESIDUES
            ]
            if modifications:
                entity["modifications"] = modifications
        chains[chain_id] = (len(sequences) + 1, kind, entity)
        sequences.append({kind: entity})

    _read_chain_references(block, path.parent, chains)
    return {
        "name": sample_name or path.stem,
        "sequences": sequences,
        "covalent_bonds": _fixed_assembly_bonds(atoms, positions, chains),
    }


def cif_to_input_json(
    mmcif_file: str,
    assembly_id: str | None = None,
    altloc="first",
    output_json: str | None = None,
    sample_name: str | None = None,
    save_entity_and_asym_id: bool = False,
    include_discont_poly_poly_bonds: bool = False,
) -> dict:
    """
    Convert mmcif file to OpenDDE input json file.

    Args:
        mmcif_file (str): mmCIF file path.
        assembly_id (str, optional): Assembly ID. Defaults to None.
        altloc (str, optional): Altloc selection. Defaults to "first".
        output_json (str, optional): Output json file path. Defaults to None.
        sample_name (str, optional): The "name" filed in json file. Defaults to None.
        save_entity_and_asym_id (bool, optional): Whether to save entity and asym ids to json.
                                                  Defaults to False.
        include_discont_poly_poly_bonds (bool, optional): Whether to include discontinuous polymer-polymer bonds.
                                                          Defaults to False.

    Returns:
        dict: OpenDDE input json dict.
    """
    parser = MMCIFParser(mmcif_file)
    atom_array = parser.get_structure(altloc, model=1, bond_lenth_threshold=None)

    # remove HOH from entities
    atom_array = Filter.remove_water(atom_array)
    atom_array = Filter.remove_hydrogens(atom_array)
    atom_array = parser.mse_to_met(atom_array)
    atom_array = Filter.remove_element_X(atom_array)

    # remove crystallization_aids
    if any(["DIFFRACTION" in m for m in parser.methods]):
        atom_array = Filter.remove_crystallization_aids(
            atom_array, parser.entity_poly_type
        )

    if assembly_id is not None:
        # expand created AtomArray by expand bioassembly
        atom_array = parser.expand_assembly(atom_array, assembly_id)

    if sample_name is None:
        sample_name = os.path.basename(mmcif_file).split(".")[0]

    json_dict = atom_array_to_input_json(
        atom_array,
        parser,
        assembly_id,
        output_json,
        sample_name,
        save_entity_and_asym_id=save_entity_and_asym_id,
        include_discont_poly_poly_bonds=include_discont_poly_poly_bonds,
    )
    return json_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cif_file", type=str, required=True, help="The cif file to parse"
    )
    parser.add_argument(
        "-j",
        "--json_file",
        type=str,
        required=False,
        default=None,
        help="The json file path to generate",
    )
    args = parser.parse_args()
    print(
        cif_to_input_json(
            args.cif_file,
            output_json=args.json_file,
        )
    )
