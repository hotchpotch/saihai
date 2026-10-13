"""Held-out selection, scheduling and example-weighted loss."""

import pytest
import torch

from saihai.training import select_validation_records, validation_loss, validation_steps


def test_ten_evenly_spaced_validations_including_final_step():
    assert validation_steps(101, 10) == {11, 21, 31, 41, 51, 61, 71, 81, 91, 101}
    assert validation_steps(6, 10) == {1, 2, 3, 4, 5, 6}
    assert validation_steps(10, 10) == set(range(1, 11))


def test_subset_selection_is_deterministic_and_bounded():
    rows = [{"id": str(i), "subset": str(i // 5)} for i in range(12)]
    selected = select_validation_records(rows, 3, 42)
    assert selected == select_validation_records(list(reversed(rows)), 3, 42)
    assert len(selected) == 8
    assert len({r["id"] for r in selected}) == 8
    assert all(sum(r["subset"] == s for r in selected) <= 3 for s in {"0", "1", "2"})


def test_validation_loss_uses_all_examples_and_restores_training_mode():
    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            assert not self.training
            assert not torch.is_grad_enabled()
            return torch.stack([input_ids[:, 0].float(), torch.zeros(len(input_ids))], dim=1)

    model = Model().train()
    rows = [
        {"id": str(i), "subset": "s", "input_ids": [i], "probabilities": [0.0, 1.0]}
        for i in range(3)
    ]
    result = validation_loss(model, rows, 0, batch_size=2, device="cpu")
    expected = torch.nn.functional.softplus(torch.arange(3).float()).mean().item()
    assert result["loss"] == pytest.approx(expected)
    assert result["subsets"]["s"]["loss"] == pytest.approx(expected)
    assert result["examples"] == 3
    assert model.training
