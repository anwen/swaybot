from swaybot.output_limits import truncate_lines, truncate_text, write_overflow


def test_truncate_text_returns_short_text_unchanged(tmp_path):
    text = "short"
    assert truncate_text(text, tmp_path) == text


def test_truncate_text_writes_overflow_for_long_text(tmp_path):
    text = "line\n" * 300
    result = truncate_text(text, tmp_path, max_chars=1000, max_lines=10)
    assert "truncated" in result
    assert "saved to .swaybot/overflows" in result
    overflow = result.split("saved to ")[-1].rstrip("]")
    full_path = tmp_path / overflow
    assert full_path.exists()
    assert full_path.read_text(encoding="utf-8") == text


def test_truncate_lines_returns_short_list_unchanged(tmp_path):
    lines = ["a", "b", "c"]
    assert truncate_lines(lines, tmp_path, max_lines=10) == lines


def test_truncate_lines_writes_overflow_for_long_list(tmp_path):
    lines = [f"line {i}" for i in range(250)]
    result = truncate_lines(lines, tmp_path, max_lines=50)
    assert len(result) == 51
    assert "truncated" in result[-1]
    overflow = result[-1].split("saved to ")[-1].rstrip("]")
    full_path = tmp_path / overflow
    assert full_path.exists()
    assert full_path.read_text(encoding="utf-8") == "\n".join(lines)


def test_write_overflow_for_string(tmp_path):
    rel = write_overflow("hello", tmp_path)
    assert (tmp_path / rel).read_text(encoding="utf-8") == "hello"
