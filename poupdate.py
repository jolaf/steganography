#!/usr/bin/env python3

from collections.abc import Iterator, Mapping
from pathlib import Path
from re import escape, search
from itertools import chain
from typing import Final

from polib import pofile, POFile

from beartype import beartype as typechecked

type Occurrence = tuple[str, str]
type Occurrences = list[Occurrence]

KEEP: Final[str] = 'keep'
PATTERNS: Mapping[str, str] = {
    'html': r'''[>]%s:?[<]|"%s[)]?"|[(]%s[)]|^\s+%s:?$|,\s+%s\s\d+''',
    'py': r'''_[(]"?%s"?[)]|"%s"''',
}

@typechecked
def getOccurrences(fileName: str, msgid: str) -> Iterator[Occurrence]:
    path = Path(fileName)
    pattern = PATTERNS[path.suffix[1:].lower()].replace('%s', escape(msgid))
    with path.open('r') as f:
        for (n, line) in enumerate(f, 1):
            if search(pattern, line):
                print(f"# {fileName}:{n} {msgid}\n{line.strip()}\n")
                occurrence: Occurrence = (fileName, str(n))
                yield occurrence

@typechecked
def main() -> None:
    poFileName = './gettext/ru.po'
    po = pofile(poFileName, wrapwidth = 999, encoding = 'utf-8', check_for_duplicates = True)

    for entry in po:
        fileNames = tuple(sorted({occurrence[0] for occurrence in entry.occurrences}))
        if occurrences := list(chain.from_iterable(getOccurrences(fileName, entry.msgid) for fileName in fileNames if fileName != KEEP)):
            entry.occurrences = occurrences
        elif KEEP not in fileNames:
            po.remove(entry)

    newPO = POFile(wrapwidth = 999, encoding = 'utf-8', check_for_duplicates = True)
    newPO.header = po.header
    newPO.metadata = po.metadata

    for entry in sorted(po, key = lambda entry: (entry.occurrences[0][0], int(entry.occurrences[0][1]))):
        newPO.append(entry)

    newPO.save(poFileName)
    newPO.save_as_mofile(poFileName.replace('.po', '.mo'))

if __name__ == '__main__':
    main()
