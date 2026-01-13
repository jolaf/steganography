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
    from PIL.Image import fromarray as ImageFromArray, new as ImageNew, open as imageOpen, Dither, Image, Resampling, Transpose
    from PIL.ImageMode import getmode as imageGetMode
    from PIL._typing import StrOrBytesPath
except ImportError as ex:
    raise ImportError(f"{type(ex).__name__}: {ex}\n\nThis module requires Pillow, please install v11.3 or later: https://pypi.org/project/pillow/\n") from ex

try:
    import numpy as np
    import numpy.typing as npt
    np.zeros((100,100))  # Warm-up JIT
except ImportError as ex:
    raise ImportError(f"{type(ex).__name__}: {ex}\n\nThis module requires NumPy, please install v2.2.5 or later: https://pypi.org/project/numpy/\n") from ex

try:
    from beartype import beartype as typechecked
except ImportError:
    def typechecked(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[no-redef]
        return func

type ImagePath = StrOrBytesPath | Buffer | IO[bytes] | BytesIO  # The last one is needed for beartype, though it shouldn't be

PNG: Final[str] = 'PNG'
BW1: Final[str] = '1'

IMAGE_MODE_DESCRIPTIONS: Final[Mapping[str, str]] = {
    BW1: '1-bit B&W',
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

SMOOTH_COMBINATIONS: Final[tuple[npt.NDArray[np.bool], ...]] = tuple(np.array(v, bool) for v in (
    ((0, 0), (1, 1)),
    ((0, 1), (0, 1)),
    ((0, 1), (1, 0)),
    ((1, 0), (0, 1)),
    ((1, 0), (1, 0)),
    ((1, 1), (0, 0)),
))

@typechecked
def log(*args: Any) -> None:
    print(*args)

@typechecked
def error(*args: Any) -> None:
    print("ERROR:", *args, file = stderr)
    sysExit(1)

@typechecked
def loadImage(source: ImagePath, fileName: str | None = None) -> Image:
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
               transparency = 1 if image.mode == BW1 else None)  # `transparency` here sets the index of the color to make transparent, 1 is usually white

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
        typeStr = imageGetMode(mode).typestr
        assert len(typeStr) == 3, typeStr
        (endian, signed, bits) = tuple(values[x] for (values, x) in zip(MAPPING_MODE_PARAMETERS, typeStr, strict = True))
        return f"{bits}-bit {endian} endian {signed} Int"
    description = IMAGE_MODE_DESCRIPTIONS.get(mode, mode)
    if mode == BW1 and hasAlpha(image):
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
    return PNG

@typechecked
def finalizeImage(image: Image) -> None:
    """Improves further work with 1-bit B&W with Alpha images."""
    assert image.mode == BW1
    image.info = {'transparency': 1}
    if not image.format:
        image.format = PNG

@typechecked
def processImage(image: Image,
                 *,
                 resizeFactor: float | None = None,
                 resizeWidth: int | None = None,
                 resizeHeight: int | None = None,
                 randomRotate: bool | None = None,
                 randomFlip: bool | None = None,
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
        assert resizeWidth
        assert resizeHeight
        processed = processed.resize((resizeWidth, resizeHeight), Resampling.BICUBIC)
    if randomRotate:  # ToDo: Should we save rotate angle and flip bit somewhere?
        processed = processed.rotate(choice(range(1, 359 + 1)), Resampling.BICUBIC, expand = True, fillcolor = 255)  # White background
    if randomFlip and choice((False, True)):
        processed = processed.transpose(Transpose.FLIP_LEFT_RIGHT)
    if image.mode == BW1 and hasAlpha(image) and processed is grayscale:
        return image  # Return original image as it's in correct format and no changes were actually made
    processed = processed.convert(BW1, dither = Dither.FLOYDSTEINBERG if dither else Dither.NONE)
    finalizeImage(processed)
    return processed

@typechecked
def encrypt(source: Image, lockMask: Image | None = None, keyMask: Image | None = None, *, smooth: bool | None = None) -> tuple[Image, Image]:  # noqa: ARG001  # pylint: disable=unused-argument
    """
    Generates lock and key images from the specified source `Image`.
    If `lockMask` and/or `keyMask` are provided,
    they are used as visible hints on the corresponding output images.
    """
    # PIL/NumPy array indexes are `y` first, `x` second
    dimensions = (source.height * 2, source.width * 2) if smooth else (source.height, source.width)
    lockArray = np.empty(dimensions, bool)
    keyArray = np.empty(dimensions, bool)
    for ((y, x), b) in np.ndenumerate(np.asarray(source, bool)):
        # b: False is black, True is transparent
        if smooth:
            x *= 2 ; y *= 2  # noqa: E702, PLW2901  # pylint: disable=multiple-statements, redefined-loop-name
            lockArray[y : y + 2, x : x + 2] = r2 = choice(SMOOTH_COMBINATIONS)
            keyArray[y : y + 2, x : x + 2] = r2 if b else np.invert(r2)
        else:
            lockArray[y, x] = r = choice((False, True))
            keyArray[y, x] = r if b else not r
    lockImage = ImageFromArray(lockArray)
    keyImage = ImageFromArray(keyArray)
    finalizeImage(lockImage)
    finalizeImage(keyImage)
    return (lockImage, keyImage)

@typechecked
def overlay(lockImage: Image, keyImage: Image, *, border: bool | None = None) -> Image:  # noqa: ARG001  # pylint: disable=unused-argument
    """
    Emulates precise overlaying of two 1-bit images one on top of the other,
    as if they were printed on transparent film.
    """
    assert lockImage.mode == BW1
    assert keyImage.mode == BW1
    assert lockImage.size == keyImage.size
    retData = bytearray(lockImage.width * lockImage.height)  # ToDo: rewrite using NumPy
    for (i, (lb, kb)) in enumerate(zip(lockImage.getdata(), keyImage.getdata(), strict = True)):
        retData[i] = min(lb, kb)
    ret = ImageNew(BW1, lockImage.size)
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
    'processImage',
)

if __name__ == '__main__':
    main(*argv[1:])
