"""Full text access for agents: read a paper by id, whole or by section, under a read policy.

Policies: full, abs, abs+intro, abs+method, abs+refs, abs+intro+refs, choose (the agent picks sections).

    python agentic/fulltext.py --full-text corpus_full_text.jsonl --doc-id arxiv_0809.1493 --policy abs+intro
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

POLICIES = ("full", "abs", "abs+intro", "abs+method", "abs+refs", "abs+intro+refs", "choose")
METHOD_WORDS = ("method", "approach", "model", "framework", "algorithm", "architecture")
REFS_MAX_ENTRIES = 80
REFS_MAX_CHARS = 6_000
_ID_RE = re.compile(rb'^\{"id":\s*"([^"]+)"')
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.M)
_REF_START_RE = re.compile(r"^(\[\d+\]|\d+\.|[-*])\s+")  # "[1] ", "1. ", "- ", "* "


def build_offsets(full_text_path: Path, index_path: Path) -> dict[str, list[int]]:
    """{doc_id: [offset, length]} for every line, cached at index_path (rebuilt when the
    corpus file size changes). Reads the file once, ~7.7 GB, without parsing the JSON."""
    size = full_text_path.stat().st_size
    if index_path.exists():
        cached = json.loads(index_path.read_text())
        if cached.get("size") == size:
            return cached["offsets"]
    offsets: dict[str, list[int]] = {}
    with full_text_path.open("rb") as f:
        pos = 0
        for line in f:
            m = _ID_RE.match(line)
            if m:
                offsets[m.group(1).decode()] = [pos, len(line)]
            pos += len(line)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"size": size, "offsets": offsets}))
    return offsets


def split_sections(text: str) -> list[tuple[str, str]]:
    """[(title, body)] from markdown headings; a paper without headings is [("full", text)]."""
    heads = list(_HEADING_RE.finditer(text))
    if not heads:
        return [("full", text)]
    out = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.append((h.group(2), text[h.end():end].strip()))
    return out


def is_refs_title(title: str) -> bool:
    return title.lower().startswith(("references", "bibliography"))


def split_references(body: str) -> list[str]:
    """Entries of a bibliography section: one per "[1]" / "1." / "- " line when the body uses
    them (later lines continue the entry), else one per blank-line block, else one per line.
    Capped at REFS_MAX_ENTRIES entries and REFS_MAX_CHARS chars."""
    lines = [l.strip() for l in body.splitlines()]
    if any(_REF_START_RE.match(l) for l in lines):
        entries: list[str] = []
        for line in lines:
            if _REF_START_RE.match(line):
                entries.append(line)
            elif line and entries:
                entries[-1] += " " + line
    else:
        entries = [" ".join(b.split()) for b in re.split(r"\n\s*\n", body) if b.strip()]
        if len(entries) == 1:
            entries = [l for l in lines if l]
    out, total = [], 0
    for e in entries[:REFS_MAX_ENTRIES]:
        if total + len(e) > REFS_MAX_CHARS:
            break
        out.append(e)
        total += len(e)
    return out


def refs_block(entries: list[str]) -> str:
    return f"## References ({len(entries)} entries)\n" + "\n".join(entries)


def allowed_titles(sections: list[tuple[str, str]], policy: str) -> list[str]:
    titles = [t for t, _ in sections]
    if policy in ("full", "choose"):
        return titles
    keep = [t for t in titles if t.lower().startswith("abstract")]
    if policy in ("abs+intro", "abs+intro+refs"):
        keep += [t for t in titles if "introduction" in t.lower()]
    if policy == "abs+method":
        keep += [t for t in titles if any(w in t.lower() for w in METHOD_WORDS)]
    if policy.endswith("+refs"):
        keep += [t for t in titles if is_refs_title(t)]
    return keep or titles[:1]  # no headings, or no match: the first section, which is the abstract-like head


def _display_title(title: str, body: str) -> str:
    """Section title as listed to the agent; the bibliography carries its entry count."""
    return f"{title} ({len(split_references(body))} entries)" if is_refs_title(title) else title


_ENUM_RE = re.compile(r"^\s*(?:[IVXLC]+(?:-[A-Z])?|\d+(?:\.\d+)*)[.:)]?\s+")  # "I ", "II-A ", "1. ", "3.2 "
_ENTRIES_RE = re.compile(r"\s*\(\d+ entries\)$")


def norm_title(title: str) -> str:
    """A section title the way agents type it: no leading section number, no bibliography entry
    count, one space between words, lower case. "I INTRODUCTION" and "1. Introduction" both
    become "introduction"."""
    return " ".join(_ENUM_RE.sub("", _ENTRIES_RE.sub("", title)).lower().split())


class FullText:
    def __init__(self, full_text_path: Path, policy: str = "full", max_chars: int = 12_000, index_path: Path | None = None):
        if policy not in POLICIES:
            raise ValueError(f"unknown read policy {policy!r}; choose from {POLICIES}")
        self.path, self.policy, self.max_chars = Path(full_text_path), policy, max_chars
        self.offsets = build_offsets(self.path, index_path or self.path.with_suffix(".offsets.json"))

    def get(self, doc_id: str) -> dict | None:
        span = self.offsets.get(doc_id)
        if not span:
            return None
        with self.path.open("rb") as f:
            f.seek(span[0])
            return json.loads(f.read(span[1]))

    def _sections(self, doc_id: str) -> list[tuple[str, str]]:
        doc = self.get(doc_id)
        return split_sections(doc.get("text") or "") if doc else []

    def sections(self, doc_id: str) -> list[str]:
        secs = self._sections(doc_id)
        allowed = allowed_titles(secs, self.policy)
        return [_display_title(t, b) for t, b in secs if t in allowed]

    def has_refs(self, doc_id: str) -> bool:
        """True when the paper has a bibliography section with at least one entry."""
        return any(is_refs_title(t) and split_references(b) for t, b in self._sections(doc_id))

    def read(self, doc_id: str, section: str | None = None) -> str:
        """Text the policy allows: one named section when given, else every allowed section.
        The body is cut to max_chars; the bibliography, when allowed, follows it with its own cap."""
        secs = self._sections(doc_id)
        if not secs:
            return f"no full text for {doc_id}"
        allowed = allowed_titles(secs, self.policy)
        wanted = [(t, b) for t, b in secs if t in allowed]
        if section:
            # papers number their headings ("I Introduction", "3. Methodology") and agents do not, so
            # titles are compared without the number; a partial title ("Method") takes every section
            # whose title contains it when no title equals it
            want = norm_title(section)
            exact = [(t, b) for t, b in wanted if norm_title(t) == want]
            wanted = exact or [(t, b) for t, b in wanted if want and want in norm_title(t)]
            if not wanted:
                return f"section {section!r} is not available; choose from: {self.sections(doc_id)}"
        text = "\n\n".join(f"## {t}\n{b}" for t, b in wanted if not is_refs_title(t))
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + f"\n[truncated to {self.max_chars} chars]"
        refs = [e for t, b in wanted if is_refs_title(t) for e in split_references(b)]
        if refs:
            text = f"{text}\n\n{refs_block(refs)}" if text else refs_block(refs)
        return text


HELPER = '''#!/usr/bin/env python3
"""read_paper.py <doc_id> [section]: print a paper's full text (or one section) under the run's read policy."""
import sys
sys.path.insert(0, {src!r})
from agentic.fulltext import FullText
ft = FullText({path!r}, policy={policy!r}, max_chars={max_chars})
if len(sys.argv) < 2:
    sys.exit("usage: read_paper.py <doc_id> [section]")
if len(sys.argv) == 2 and ft.policy == "choose":
    print("sections:", ft.sections(sys.argv[1]))
print(ft.read(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
'''


def write_helper(ft: FullText, view_dir: Path) -> Path:
    """Drop read_paper.py into a corpus view so bash-driven agents can read papers too."""
    script = view_dir / "read_paper.py"
    script.write_text(HELPER.format(src=str(Path(__file__).resolve().parents[1]), path=str(ft.path),
                                    policy=ft.policy, max_chars=ft.max_chars))
    script.chmod(0o755)
    return script


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full-text", type=Path, required=True)
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--policy", default="full", choices=POLICIES)
    ap.add_argument("--section", default=None)
    args = ap.parse_args()
    ft = FullText(args.full_text, args.policy)
    print("sections:", ft.sections(args.doc_id))
    print(ft.read(args.doc_id, args.section))


if __name__ == "__main__":
    main()
