import copy

import pytest

from tools.combine_downstream_cell_lists import combine_cell_lists
from tools.select_downstream_cells import select_cells


def _rows() -> list[dict]:
    return [
        {
            "cell_id": f"cell-{index}",
            "target_fraction": fraction,
            "target_recording": recording,
            "overrides": [f"run.cell_id=cell-{index}"],
        }
        for index, (recording, fraction) in enumerate(
            [
                ("recording-a", 0.05),
                ("recording-b", 0.05),
                ("recording-a", 1.0),
                ("recording-b", 0.05),
            ]
        )
    ]


def test_select_cells_preserves_exact_rows_and_order() -> None:
    rows = _rows()
    original = copy.deepcopy(rows)
    selected = select_cells(
        rows,
        fraction=0.05,
        recordings={"recording-a", "recording-b"},
        offset=1,
        count=2,
    )
    assert selected == [rows[1], rows[3]]
    assert rows == original


def test_select_cells_rejects_short_selection() -> None:
    with pytest.raises(ValueError, match="only 1 rows matched"):
        select_cells(
            _rows(),
            fraction=1.0,
            recordings={"recording-a"},
            offset=0,
            count=2,
        )


def test_combine_cell_lists_preserves_rows_and_rejects_duplicates() -> None:
    rows = _rows()
    combined = combine_cell_lists([[rows[0]], [rows[1], rows[2]]])
    assert combined == rows[:3]
    with pytest.raises(ValueError, match="duplicate cell IDs"):
        combine_cell_lists([[rows[0]], [rows[0]]])
