import torch

from opendde.model.modules.embedders import shortest_cyclic_offset


def test_shortest_cyclic_offset_wraps_terminal_residues() -> None:
    residue_index = torch.arange(6).unsqueeze(0)
    offset = residue_index[..., :, None] - residue_index[..., None, :]
    period = torch.full_like(residue_index, 6)
    same_chain = torch.ones_like(offset, dtype=torch.bool)

    result = shortest_cyclic_offset(offset, period, same_chain)

    assert result[0, 0, 5].item() == 1
    assert result[0, 5, 0].item() == -1
    assert result[0, 0, 3].item() == -3  # Preserve even-period ties.


def test_shortest_cyclic_offset_preserves_linear_and_cross_chain_pairs() -> None:
    offset = torch.tensor([[[0, -5], [5, 0]]])
    period = torch.tensor([[6, 6]])

    assert torch.equal(
        shortest_cyclic_offset(offset, None, torch.ones_like(offset, dtype=torch.bool)),
        offset,
    )
    assert torch.equal(
        shortest_cyclic_offset(offset, period, torch.zeros_like(offset, dtype=torch.bool)),
        offset,
    )
