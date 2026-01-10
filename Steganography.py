from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from mimetypes import guess_type
from random import randint
from re import split
from sys import argv, exit as sysExit, stderr
from typing import Any, Final, IO

try:
    from beartype import beartype as typechecked
except ImportError:
    print("WARNING: beartype is not available, running fast with typing unchecked")

    def typechecked(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[no-redef]
        return func

from PIL.Image import open as imageOpen, Dither, Image as Image, Resampling  # noqa: PLC0414
from PIL.ImageMode import getmode
# noinspection PyProtectedMember
from PIL._typing import StrOrBytesPath

type ImagePath = StrOrBytesPath | IO[bytes] | BytesIO  # The last one is needed by beartype, though it's a bug

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
def loadImage(inp: ImagePath | bytes, fileName: str | None = None) -> Image:
    image = imageOpen(BytesIO(inp) if isinstance(inp, bytes) else inp)
    if not image.format:
        image.format = getImageFormatFromExtension(fileName or inp)
    return image

@typechecked
def saveImage(image: Image, path: ImagePath) -> None:
    image.save(path, getImageFormatFromExtension(path), optimize = True,
               transparency = 1 if image.mode == '1' else None)  # `transparency` here sets the index of a color to make transparent, 1 is usually white

@typechecked
def imageToBytes(image: Image) -> bytes:
    stream = BytesIO()
    saveImage(image, stream)
    return stream.getvalue()

@typechecked
def hasAlpha(image: Image) -> bool:
    return 'A' in image.getbands() or image.info.get('transparency') is not None  # The latter covers 1-bit B&W

@typechecked
def getImageMode(image: Image) -> str:
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
    if image.format:
        try:
            (mimeType, _encoding) = guess_type('image.' + image.format.lower())
            if mimeType:
                return mimeType
        except Exception:  # noqa: BLE001, S110
            pass
    return 'image/png'

@typechecked
def getImageFormatFromExtension(path: ImagePath) -> str:
    try:
        (mimeType, _encoding) = guess_type(path)  # type: ignore[arg-type]
        if mimeType and mimeType.startswith('image/'):
            return mimeType.split('/')[1].upper()
    except Exception:  # noqa: BLE001, S110
        pass
    return 'PNG'

@typechecked
def finalizeImage(image: Image) -> None:
    assert image.mode == '1'
    image.info = {'transparency': 1}
    if not image.format:
        image.format = 'PNG'

@typechecked
def processImage(image: Image, **options: Any) -> Image:
    # Checks options:
    # - resize: int | float | tuple[int | None, int | None]
    # - rotate: bool
    # - dither: bool
    processed = grayscale = image.convert('L')  # 8-bit grayscale
    if (resize := options.get('resize')) not in (None, 0, 1, (), (None, None)):  # int size tuple or float factor
        assert resize
        if isinstance(resize, Sequence):
            assert len(resize) == 2 and all(r is None or isinstance(r, int) for r in resize) and any(isinstance(r, int) for r in resize), f"Bad resize options: {resize!r}"  # noqa: PT018
            for r in resize:
                if r is not None:
                    assert isinstance(r, int) and r > 0, f"Bad resize options: {resize!r}"  # noqa: PT018
            if resize[0] is None:
                assert resize[1]
                resize = (round(float(image.size[0]) * resize[1] / image.size[1]), resize[1])
            elif resize[1] is None:
                assert resize[0]
                resize = (resize[0], round(float(image.size[1]) * resize[0] / image.size[0]))
        else:  # resize is factor
            assert isinstance(resize, (int, float)), f"Bad resize options: {resize!r}"
            resize = tuple(round(d * resize) for d in image.size)
        assert isinstance(resize, tuple) and len(resize) == 2 and all(isinstance(r, int) for r in resize), repr(resize) # noqa: PT018
        processed = processed.resize(resize, Resampling.BICUBIC)
    if options.get('rotate'):  # bool  # ToDo: Should we save rotate angle somewhere?
        processed = processed.rotate(randint(1, 359), Resampling.BICUBIC, expand = True, fillcolor = 255)  # White background  # noqa: S311

    if image.mode == '1' and hasAlpha(image) and processed is grayscale:
        return image  # Return original image as no changes were made
    processed = processed.convert('1', dither = Dither.FLOYDSTEINBERG if options.get('dither') else Dither.NONE)  # 1-bit B&W
    finalizeImage(processed)
    return processed

@typechecked
def synthesize(source: Image, lock: Image | None, key: Image | None, **options: Any) -> Image:  # noqa: ARG001  # pylint: disable=unused-argument
    # Uses options:
    # - smooth: bool
    return Image()

@typechecked
def testOverlay(lock: Image, key: Image, **options: Any) -> Image:  # noqa: ARG001  # pylint: disable=unused-argument
    # Uses options:
    # - border: bool
    return Image()

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

if __name__ == '__main__':
    main(*argv[1:])
