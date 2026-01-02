#!/usr/bin/env python3

from base64 import b64encode
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, UTC
from pathlib import Path
from re import sub, Match
from sys import argv
from urllib.request import urlopen
from zipfile import ZipFile, ZIP_DEFLATED

ROOT_PATH = Path(argv[0]).parent

HTML_SOURCE = 'index.html'
HTML_TARGET = 'steganography.html'
ZIP_TARGET = 'steganography.zip'

BASE64 = 'BASE64'
NO_INDENT = ''

SubPattern = str | Callable[[Match[str]], str]

HTML_PATTERNS: Sequence[tuple[str, SubPattern]] = (
    (r'([ \t]*)<link rel="stylesheet" href="(\S+)">',
        lambda match: loadFile(match, r'\1<style>\n%s\1</style>', 2)),
    (r'([ \t]*)<script src="(\S+)"></script>',
        lambda match: loadFile(match, r'\1<script>\n%s\1</script>', 2)),
    (r'(?s)(];\n)</script>\n[ \t]*<script>.*"use strict";\n',
        r'\1'),
    (r'([ \t]*)<object type="image/svg\+xml" data="(\S+)".*?></object>\n',
        lambda match: loadFile(match, r'%s', 2, NO_INDENT)),
    (r'type="(\S+)" href="(\S+?)(\?[^"]+)?"',
        lambda match: loadFile(match, r'type="\1" href="data:\1;base64,%s"', 2, BASE64)),
    (r'<img ([^<>]*) src="((\S+)\.(\S+?))(\?[^"]+)?"',
        lambda match: loadImage(match, r'<img \1 src="%s"', 2)),
    (r' url\("((\S+)\.(\S+?))(\?[^"]+)?"\)',
        lambda match: loadImage(match, r' url("%s")')),
    (r'(\sid="build">)\S+?(</)',
        lambda match: match.expand(rf'\1{datetime.now(UTC).strftime("B%Y%m%d-%H%MG")}\2')),
)

def getFilePath(fileName: str) -> Path:
    return ROOT_PATH / fileName

def loadFile(match: Match[str], replacePattern: str, fileNamePos: int, mode: str | None = None) -> str:
    fileName = match.group(fileNamePos)
    print('F', fileName)
    if fileName.startswith('http'):  # noqa: SIM108  # pylint: disable=consider-ternary-expression
        f = urlopen(fileName.replace('jquery.js', 'jquery.slim.min.js'))  # noqa: S310
    else:
        f = getFilePath(fileName).open('rb')
    with f:
        data = f.read()
    if mode is BASE64:
        data = b64encode(data).decode()
    else:
        data = data.decode()
        if fileName.endswith('.js'):
            data = data.replace('\\', '\\\\')
        if mode is not None:
            indent = match.group(1)
            linePattern = '%s\n'
            indentedPattern = ''.join((indent if indent and not indent.strip() else '', mode, linePattern))
            data = ''.join(((linePattern if line.startswith('\t') else indentedPattern) % line) if line.strip() else '\n' for line in data.splitlines())
    return match.expand(replacePattern % data)

def loadImage(match: Match[str], pattern: str, fileNamePos: int = 1) -> str:
    fileName = match.group(fileNamePos)
    print('I', fileName)
    with getFilePath(fileName).open('rb') as f:
        data = f.read()
    return match.expand(pattern % f'data:image/{match.group(fileNamePos + 2).lower()};base64,{b64encode(data).decode()}')

def processFile(source: str, target: str, patterns: Iterable[tuple[str, SubPattern]]) -> None:
    with getFilePath(source).open('rb') as f:
        data = f.read().decode()
    for (pattern, replace) in patterns:
        if (newData := sub(pattern, replace, data)) == data:
            print('-', pattern)
        else:
            if not callable(replace):
                print('+', pattern)
            data = newData
    with getFilePath(target).open('wb') as f:
        f.write(data.encode())

def createZip() -> None:
    with ZipFile(getFilePath(ZIP_TARGET), 'w', ZIP_DEFLATED) as f:
        f.write(getFilePath(HTML_TARGET))

def main() -> None:
    print("\nCompiling HTML...")
    processFile(HTML_SOURCE, HTML_TARGET, HTML_PATTERNS)
    print("\nCreating ZIP...")
    createZip()
    print("\nDONE")

if __name__ == '__main__':
    main()
