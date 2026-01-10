from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Buffer, Callable, Mapping
from contextlib import suppress
from io import BytesIO
from mimetypes import guess_type
from re import split
from secrets import choice
from sys import argv, exit as sysExit, stderr
from typing import Any, Final, IO

try:
    from beartype import beartype as typechecked
except ImportError:
    def typechecked(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[no-redef]
        return func

from PIL import __version__ as pilVersion
from PIL.Image import new as imageNew, open as imageOpen, Dither, Image, Resampling
from PIL.ImageMode import getmode
from PIL._typing import StrOrBytesPath

type ImagePath = StrOrBytesPath | Buffer | IO[bytes] | BytesIO  # The last one is needed by beartype, though it shouldn't be

IMAGE_MODE_DESCRIPTIONS: Final[Mapping[str, str]] = {
    '1': '1-bit B&W',
    'CMYK': '8-bit CMYK',
    'F': '32-bit Float',
    'HSV': '8-bit HSV',
    'L': '8-bit Grayscale',
    'LA': '8-bit Grayscale with Alpha',
    'La': '8-bit Grayscale with premultiplied Alpha',
    'LAB': '8-bit Lab',
    'P': '8-bit Palette',
    'PA': '8-bit Palette with Alpha',
    'RGB': '8-bit RGB',
    'RGBA': '8-bit RGB with Alpha',
    'RGBa': '8-bit RGB with premultiplied Alpha',
    'RGBX': '8-bit RGB with padding',
    'YCbCr': '8-bit YCbCr',
}

MAPPING_MODE_PARAMETERS: Final[tuple[Mapping[str, str | int], ...]] = (
    { '<': 'little', '>': 'big' },
    { 'i': 'signed', 'u': 'unsigned' },
    { '2': 16, '4': 32 },
)

@typechecked
def log(*args: Any) -> None:
    print(*args)

@typechecked
def error(*args: Any) -> None:
    print("ERROR:", *args, file = stderr)
    sysExit(1)

@typechecked
def loadImage(source: ImagePath | Buffer, fileName: str | None = None) -> Image:
    """
    Returns `Image` loaded from the specified file path,
    input stream or any bytes `Buffer`.

    `fileName` can be added as a hint to image format in cases
    when `source` does not provide such a clue, like `BytesIO`.
    """
    image = imageOpen(BytesIO(source) if isinstance(source, Buffer) else source)
    if not image.format:
        image.format = getImageFormatFromExtension(fileName or source)
    return image

@typechecked
def saveImage(image: Image, path: ImagePath) -> None:
    image.save(path, getImageFormatFromExtension(path), optimize = True,  # type: ignore[arg-type]
               transparency = 1 if image.mode == '1' else None)  # `transparency` here sets the index of the color to make transparent, 1 is usually white

@typechecked
def imageToBytes(image: Image) -> Buffer:
    """
    Returns the `Image` as bytes `Buffer` that can be written to a file
    with extension corresponding to image format.
    """
    stream = BytesIO()
    saveImage(image, stream)
    return stream.getbuffer()

@typechecked
def hasAlpha(image: Image) -> bool:
    return 'A' in image.getbands() or image.info.get('transparency') is not None  # The latter is for 1-bit B&W

@typechecked
def getImageMode(image: Image) -> str:
    """Returns human-readable description of the specified `Image` mode."""
    mode = image.mode
    if mode.startswith('I'):
        typeStr = getmode(mode).typestr
        assert len(typeStr) == 3, typeStr
        (endian, signed, bits) = tuple(values[x] for (values, x) in zip(MAPPING_MODE_PARAMETERS, typeStr, strict = True))
        return f"{bits}-bit {endian} endian {signed} Int"
    description = IMAGE_MODE_DESCRIPTIONS.get(mode, mode)
    if mode == '1' and hasAlpha(image):
        return f"{description} with Alpha"
    return description

@typechecked
def getMimeTypeFromImage(image: Image) -> str:
    """
    Returns MIME type corresponding to the specified `Image` format.
    If format is unknown or not specified, returns `image/png`.
    """
    if image.format:
        with suppress(Exception):
            (mimeType, _encoding) = guess_type('image.' + image.format.lower())
            if mimeType:
                return mimeType
    return 'image/png'

@typechecked
def getImageFormatFromExtension(path: ImagePath) -> str:
    with suppress(Exception):
        (mimeType, _encoding) = guess_type(path)  # type: ignore[arg-type]
        if mimeType and mimeType.startswith('image/'):
            return mimeType.split('/')[1].upper()
    return 'PNG'

@typechecked
def finalizeImage(image: Image) -> None:
    """Improves further work with 1-bit B&W with Alpha images."""
    assert image.mode == '1'
    image.info = {'transparency': 1}
    if not image.format:
        image.format = 'PNG'

@typechecked
def processImage(image: Image, *,
                 resizeFactor: float | None = None,
                 resizeWidth: int | None = None,
                 resizeHeight: int | None = None,
                 rotate: bool | None = None,
                 dither: bool| None = None) -> Image:
    """Converts the arbitrary `Image` to 1-bit B&W with Alpha format."""
    processed = grayscale = image.convert('L')  # 8-bit grayscale
    if resizeFactor not in (None, 1) or resizeWidth is not None or resizeHeight is not None:
        if resizeFactor is not None and (not isinstance(resizeFactor, int | float) or resizeFactor <= 0):
            raise ValueError(f"Bad resizeFactor {resizeFactor}, must be positive int or float")
        if resizeWidth is not None and (not isinstance(resizeWidth, int) or resizeWidth <= 0):
            raise ValueError(f"Bad resizeWidth {resizeWidth}, must be positive int")
        if resizeHeight is not None and (not isinstance(resizeHeight, int) or resizeHeight <= 0):
            raise ValueError(f"Bad resizeHeight {resizeHeight}, must be positive int")
        if resizeFactor:
            if resizeWidth or resizeHeight:
                raise ValueError("Either resizeFactor or resizeWidth/resizeHeight can be specified")
            resizeWidth = round(resizeFactor * image.width)
            resizeHeight = round(resizeFactor * image.height)
        elif resizeWidth and not resizeHeight:
            resizeHeight = round(float(image.height) * resizeWidth / image.width)
        elif resizeHeight and not resizeWidth:
            resizeWidth = round(float(image.width) * resizeHeight / image.height)
        processed = processed.resize((resizeWidth, resizeHeight), Resampling.BICUBIC)
    if rotate:  # ToDo: Should we save rotate angle somewhere?
        processed = processed.rotate(choice(range(1, 359 + 1)), Resampling.BICUBIC, expand = True, fillcolor = 255)  # White background

    if image.mode == '1' and hasAlpha(image) and processed is grayscale:
        return image  # Return original image as it's in correct format and no changes were made
    processed = processed.convert('1', dither = Dither.FLOYDSTEINBERG if dither else Dither.NONE)  # 1-bit B&W
    finalizeImage(processed)
    return processed

@typechecked
def encrypt(source: Image, lockMask: Image | None = None, keyMask: Image | None = None, *, smooth: bool | None = None) -> tuple[Image, Image]:  # noqa: ARG001  # pylint: disable=unused-argument
    """
    Generates lock and key images from the specified source `Image`.
    If `lockMask` and/or `keyMask` are provided,
    they are used as visible hints on the corresponding output images.
    """
    arraySize = source.width * source.height
    lockData = bytearray(arraySize)
    keyData = bytearray(arraySize)
    for (i, b) in enumerate(source.getdata()):
        r = choice((0, 1))
        lockData[i] = r
        keyData[i] = r if b else 1 - r
    synthLock = imageNew('1', source.size)
    synthLock.putdata(lockData)
    finalizeImage(synthLock)
    synthKey = imageNew('1', source.size)
    synthKey.putdata(keyData)
    finalizeImage(synthKey)
    return (synthLock, synthKey)

@typechecked
def overlay(lock: Image, key: Image, *, border: bool | None = None) -> Image:  # noqa: ARG001  # pylint: disable=unused-argument
    """
    Emulates overlaying two 1-bit images one on top of the other,
    as if they were printed on transparent film.
    """
    assert lock.size == key.size
    retData = bytearray(lock.width * lock.height)
    for (i, (lb, kb)) in enumerate(zip(lock.getdata(), key.getdata(), strict = True)):
        retData[i] = min(lb, kb)
    ret = imageNew('1', lock.size)
    ret.putdata(retData)
    finalizeImage(ret)
    return ret

@typechecked
def main(*args: str) -> None:
    parser = ArgumentParser()
    parser.add_argument('-r', '--rotate', help = 'rotate image to a random angle', action = 'store_true')
    parser.add_argument('-s', '--resize', help = 'resize image to the specified factor or size')
    parser.add_argument('-d', '--dither', help = 'dither image when converting to 1-bit format', action = 'store_true')
    parser.add_argument('-b', '--smooth', help = 'smoother background (doubles the resolution)', action = 'store_true')
    parser.add_argument('inputImage', help = 'path to input image file to process')
    options = parser.parse_args(args)
    size: int | str
    if size := options.resize:
        try:
            options.resize = int(size)
        except ValueError:
            tokens = split(r'\D', size)
            if len(tokens) != 2:
                error('Invalid resize argument: ', size)
            options.resize = tuple(tokens)  # pylint: disable=redefined-variable-type
    image = loadImage(options.inputImage)
    processed = processImage(image, **vars(options))
    saveImage(processed, 'processed.png')  # ToDo: Generate proper file names
    sysExit(0)

__all__ = (
    'Image',
    'ImagePath',
    'encrypt',
    'getImageMode',
    'getMimeTypeFromImage',
    'imageToBytes',
    'loadImage',
    'overlay',
    'pilVersion',
    'processImage',
)

if __name__ == '__main__':
    main(*argv[1:])
