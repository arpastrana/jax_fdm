"""
Render the BibTeX blocks on the citation page from CITATION.cff.

Wired into mkdocs.yml under ``hooks:``, this replaces the marker comment in
docs/citation.md with BibTeX generated at build time, so CITATION.cff is the
single source of truth for citation metadata and the docs page cannot drift
from it. The README carries a hand-written copy that GitHub renders directly;
running this file as a script checks that the README still cites the preferred
DOI, and pre-commit runs that check.
"""

import logging
import sys
from pathlib import Path

import yaml

log = logging.getLogger("mkdocs.hooks.citation")

REPO = Path(__file__).resolve().parents[2]
CFF_PATH = REPO / "CITATION.cff"
MARKER = "<!-- bibtex: generated from CITATION.cff at build time -->"


def _author(person):
    return f"{person['family-names']}, {person['given-names']}"


def _authors(people):
    return " and ".join(_author(person) for person in people)


def _entry(kind, key, fields):
    lines = [f"@{kind}{{{key},"]
    lines += [f"    {name} = {{{value}}}," for name, value in fields.items()]
    lines[-1] = lines[-1][:-1]
    lines.append("}")
    return "\n".join(lines)


def _article(record):
    key = f"{record['authors'][0]['family-names'].lower()}_{record['year']}"
    pages = str(record["start"])
    if "end" in record:
        pages += f"--{record['end']}"
    fields = {
        "title": f"{{{record['title']}}}",
        "author": _authors(record["authors"]),
        "year": record["year"],
        "journal": record["journal"],
        "volume": record["volume"],
        "pages": pages,
        "issn": record["issn"],
        "doi": record["doi"],
    }
    return _entry("article", key, fields)


def _conference_paper(record):
    key = f"{record['authors'][0]['family-names'].lower()}_{record['year']}_workshop"
    fields = {
        "title": f"{{{record['title']}}}",
        "author": _authors(record["authors"]),
        "booktitle": " ".join(record["collection-title"].split()),
        "year": record["year"],
        "address": record["location"]["name"],
        "url": record["url"],
    }
    return _entry("inproceedings", key, fields)


def _software(cff):
    key = f"{cff['authors'][0]['family-names'].lower()}_software"
    fields = {
        "title": f"{{{cff['title']}}}",
        "author": _authors(cff["authors"]),
        "url": cff["repository-code"],
    }
    return _entry("software", key, fields)


def _bibtex_blocks(cff):
    entries = [_article(cff["preferred-citation"])]
    for record in cff.get("references", []):
        if record["type"] == "conference-paper":
            entries.append(_conference_paper(record))
    entries.append(_software(cff))
    return "\n\n".join(f"```bibtex\n{entry}\n```" for entry in entries)


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri != "citation.md":
        return markdown
    if MARKER not in markdown:
        log.warning("citation.md is missing the CITATION.cff marker comment")
        return markdown
    cff = yaml.safe_load(CFF_PATH.read_text())
    return markdown.replace(MARKER, _bibtex_blocks(cff))


if __name__ == "__main__":
    cff = yaml.safe_load(CFF_PATH.read_text())
    doi = cff["preferred-citation"]["doi"]
    readme = (REPO / "README.md").read_text()
    if doi not in readme:
        sys.exit(f"README.md does not cite the preferred DOI {doi} from CITATION.cff")
