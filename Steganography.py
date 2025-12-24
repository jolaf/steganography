from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Iterable, Mapping
from io import BytesIO
from mimetypes import guess_type
from random import randint
from re import split
from sys import argv, exit as sysExit, stderr
from typing import cast, Any, IO, TYPE_CHECKING

from PIL.Image import open as imageOpen, Dither, Image as Image, Resampling  # noqa: PLC0414

if TYPE_CHECKING:
    # noinspection PyProtectedMember
    from PIL._typing import StrOrBytesPath
    ImagePath = StrOrBytesPath | IO[bytes]
    Option = bool | str | int | tuple[int, int]

IMAGE_MODE_DESCRIPTIONS: Mapping[str, str] = {
    '1': '1-bit B&W',
    'CMYK': '8-bit CMYK',
    'F': '32-bit Float',
    'HSV': '8-bit HSV',
    'I': '32-bit signed Int',
    'I;16': '16-bit unsigned Int',
    'I;16B': '16-bit big endian unsigned Int',
    'I;16L': '16-bit little endian unsigned Int',
    'I;16N': '16-bit native endian unsigned Int',
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

def log(*args: Any) -> None:
    print(*args)

def error(*args: Any) -> None:
    print("ERROR:", *args, file = stderr)
    sysExit(1)

def loadImage(inp: ImagePath | bytes) -> Image:
    #log(f"Image: {image.format} {image.mode} {image.width}x{image.height}")
    return imageOpen(BytesIO(inp) if isinstance(inp, bytes) else inp)

def getImageMode(image: Image) -> str:
    return IMAGE_MODE_DESCRIPTIONS.get(image.mode, image.mode)

def processImage(image: Image, **options: Option) -> Image:
    processed = grayscale = image.convert("L")  # 8-bit grayscale
    if (resize := options.get('resize')) not in (None, 1):  # size tuple or int factor
        assert resize
        r = cast(tuple[int, int], tuple(resize) if isinstance(resize, Iterable) else tuple(round(d * resize) for d in image.size))
        log(r)
        processed = processed.resize(r, Resampling.BICUBIC)
    if options.get('rotate'):  # bool
        processed = processed.rotate(randint(0, 359), Resampling.BICUBIC, expand = True, fillcolor = 255)  # White background  # noqa: S311
    if image.mode == '1' and processed is grayscale:  # No changes were made
        return image
    return processed.convert("1", dither = Dither.FLOYDSTEINBERG if options.get('dither') else Dither.NONE)  # 1-bit B&W

def getImageFormatFromExtension(path: ImagePath) -> str:
    try:
        (mimeType, _encoding) = guess_type(path)  # type: ignore[arg-type]
        if mimeType and mimeType.startswith('image/'):
            return mimeType.split('/')[1].upper()
    except Exception:  # noqa: BLE001, S110
        pass  # log("Exception:", e)
    return 'PNG'

def saveImage(image: Image, path: ImagePath) -> None:
    image.save(path, getImageFormatFromExtension(path), optimize = True,
               transparency = 1 if image.mode == '1' else None)  # White is transparent

def imageToBytes(image: Image) -> bytes:
    stream = BytesIO()
    saveImage(image, stream)
    return stream.getvalue()

def main(*args: str) -> None:
    parser = ArgumentParser()
    parser.add_argument('-r', '--rotate', help = 'rotate image to a random angle', action = 'store_true')
    parser.add_argument('-s', '--resize', help = 'resize image to the specified factor or size')
    parser.add_argument('-d', '--dither', help = 'dither image when converting to 1-bit format', action = 'store_true')
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
