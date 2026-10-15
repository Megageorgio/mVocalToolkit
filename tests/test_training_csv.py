import pytest

from mvocaltoolkit.formats.ds_csv import label_from_row, row_from_label


@pytest.mark.parametrize("phones,durations", [("a b", "1"), ("a", "1 2"), ("a", "nan"), ("a", "inf"), ("a", "-1"), ("a", "0")])
def test_invalid_csv_sequences_are_not_silently_truncated(phones, durations):
    with pytest.raises(ValueError):
        label_from_row({"ph_seq": phones, "ph_dur": durations})


def test_note_slurs_survive_dataset_conversion():
    row = {"ph_seq": "a b", "ph_dur": "0.5 0.5", "note_seq": "C4 D4", "note_dur": "0.5 0.5", "note_slur": "0 1"}
    label = label_from_row(row)
    assert row_from_label("song", label)["note_slur"] == "0 1"
