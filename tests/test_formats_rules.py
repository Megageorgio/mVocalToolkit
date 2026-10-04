from conftest import make_sofa_model  # noqa: F401

from mvocaltoolkit import formats
from mvocaltoolkit.formats import ds_csv
from mvocaltoolkit.labels import Interval, Label
from mvocaltoolkit.text import normalize
from mvocaltoolkit.text.dictionary import Dictionary
from mvocaltoolkit.text.rules import apply_rule_sets, apply_rules


def sample_label() -> Label:
    return Label(
        tiers={
            "phones": [
                Interval(start=0.0, end=0.1, text="SP"),
                Interval(start=0.1, end=0.25, text="hh"),
                Interval(start=0.25, end=0.5, text="ah"),
                Interval(start=0.5, end=0.6, text="SP"),
            ],
            "words": [
                Interval(start=0.0, end=0.1, text="SP"),
                Interval(start=0.1, end=0.5, text="hello"),
                Interval(start=0.5, end=0.6, text="SP"),
            ],
        }
    )


def test_htk_roundtrip():
    label = sample_label()
    text = formats.dumps(label, "htk")
    assert text.splitlines()[1] == "1000000 2500000 hh"
    back = formats.loads(text, "lab")
    assert [iv.text for iv in back.phones] == ["SP", "hh", "ah", "SP"]
    assert abs(back.phones[2].end - 0.5) < 1e-9


def test_textgrid_roundtrip():
    label = sample_label()
    text = formats.dumps(label, "textgrid")
    back = formats.loads(text, "textgrid")
    assert [iv.text for iv in back.tiers["words"]] == ["SP", "hello", "SP"]
    assert [iv.text for iv in back.tiers["phones"]] == ["SP", "hh", "ah", "SP"]


def test_textgrid_short_format():
    short = 'File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\n1\n<exists>\n1\n"IntervalTier"\n"phones"\n0\n1\n2\n0\n0.5\n"a"\n0.5\n1\n"SP"\n'
    label = formats.loads(short, "textgrid")
    assert [iv.text for iv in label.phones] == ["a", "SP"]


def test_ds_csv():
    row = ds_csv.row_from_label("song", sample_label())
    assert row["ph_seq"] == "SP hh ah SP"
    assert row["ph_num"] == "1 2 1"
    text = ds_csv.dumps_rows([row])
    back = ds_csv.label_from_row(ds_csv.loads_rows(text)[0])
    assert abs(back.phones[-1].end - 0.6) < 1e-6


def test_audacity_and_json():
    label = sample_label()
    assert formats.loads(formats.dumps(label, "audacity"), "audacity").phones[1].text == "hh"
    assert formats.loads(formats.dumps(label, "json"), "json").tiers["words"][1].text == "hello"


def test_rules_en_fixes():
    label = Label(
        tiers={
            "phones": [
                Interval(start=0.0, end=0.1, text="ah"),
                Interval(start=0.1, end=0.13, text="t"),
                Interval(start=0.13, end=0.3, text="ah"),
                Interval(start=0.3, end=0.4, text="uh"),
                Interval(start=0.4, end=0.5, text="r"),
                Interval(start=0.5, end=0.505, text="hh"),
                Interval(start=0.505, end=0.6, text="ah"),
                Interval(start=0.6, end=0.7, text="ah"),
            ]
        }
    )
    fixed = apply_rule_sets(label, ["en_fixes"])
    assert [iv.text for iv in fixed.phones] == ["ah", "dx", "ah", "er", "ah"]
    assert fixed.phones[3].start == 0.3 and fixed.phones[3].end == 0.505


def test_rules_custom_replace_and_min_duration():
    label = Label(tiers={"phones": [Interval(start=0, end=0.002, text="a"), Interval(start=0.002, end=1, text="pau")]})
    out = apply_rules(label, [{"op": "replace", "from": "pau", "to": "SP"}, {"op": "min_duration", "min_dur": 0.01}])
    assert [(iv.text, iv.start, iv.end) for iv in out.phones] == [("SP", 0, 1)]


def test_normalize():
    assert normalize.normalize("Hello, World! (x2) don't-stop", "en") == ["hello", "world", "don't", "stop"]
    assert normalize.normalize("J'aime l'amour", "fr") == ["j'", "aime", "l'", "amour"]
    assert normalize.normalize("Привет, Мир — ёлка!", "ru") == ["привет", "мир", "ёлка"]


def test_dictionary(tmp_path):
    path = tmp_path / "dict.txt"
    path.write_text("елка\te l k a\nhello\thh ah l ow\n", encoding="utf-8")
    d = Dictionary.load(path)
    assert d.lookup("ёлка") == ["e", "l", "k", "a"]
    assert d.lookup("Hello") == ["hh", "ah", "l", "ow"]
    assert d.missing(["hello", "nope", "SP", "nope"]) == ["nope"]
