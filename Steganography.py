from __future__ import annotations

from argparse import ArgumentParser
from asyncio import run, sleep, to_thread
from collections.abc import Buffer, Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from io import BytesIO
from itertools import product
from math import factorial as f
from mimetypes import guess_type
from re import split
from secrets import choice
from sys import argv, exit as sysExit, stderr
from typing import cast, Any, ClassVar, Final, IO, Literal

try:
    from PIL.Image import fromarray as ImageFromArray, new as imageNew, open as imageOpen, Dither, Image, Resampling, Transpose
    from PIL.ImageMode import getmode as imageGetMode
    from PIL.ImageOps import pad as imagePad, scale as imageScale
    from PIL._typing import StrOrBytesPath
except ImportError as ex:
    raise ImportError(f"{type(ex).__name__}: {ex}\n\nThis module requires Pillow, please install v11.3 or later: https://pypi.org/project/pillow/\n") from ex

try:
    import numpy as np
    np.zeros((100, 100))  # Warm-up JIT
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
GRAYSCALE: Final[str] = 'L'

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

MAPPING_MODE_PARAMETERS: Final[Sequence[Mapping[str, str | int]]] = (
    { '<': 'little', '>': 'big' },
    { 'i': 'signed', 'u': 'unsigned' },
    { '2': 16, '4': 32 },
)

RESAMPLING = Resampling.LANCZOS

FLIP = Transpose.FLIP_LEFT_RIGHT
ROTATE_90 = Transpose.ROTATE_90
ROTATE_180 = Transpose.ROTATE_180
ROTATE_270 = Transpose.ROTATE_270

N: Final[int] = 2 * 2  # Size of "unit" pixel block to operate on
type Bit = Literal[0, 1]
BitsN = tuple[Bit, Bit, Bit, Bit]  # Superclass for BitBlock
Array = np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.bool]]
type ArrayMap = tuple[Array, Mapping[int, Mapping[int, Sequence[Array]]]]

EXPECTED_COMPLIMENT_LENGTHS: Final[Mapping[tuple[int, int, int], int]] = {
    (2, 2, 3): 4,  # I was too lazy to figure out the formula
    (2, 2, 4): 1,
    (2, 3, 3): 2,
    (2, 3, 4): 2,
    (3, 2, 3): 3,
    (3, 2, 4): 3,
    (3, 3, 3): 1,
    (3, 3, 4): 3,
}

@typechecked
class BitBlock(BitsN):
    __slots__ = ()  # Required for effective tuple inheritance, not really used

    blockMap: ClassVar[Mapping[int, Sequence[ArrayMap]]] = {}  # Fully coordinated map of blocks for image processing

    def n(self) -> int:
        return self.count(0)

    def toArray(self) -> Array:
        return np.array((self[:2], self[2:]), bool)

    @classmethod
    def _init(cls) -> None:
        assert not cls.blockMap, "BitBlock.init() should only be called once"

        nBlocks: Final[Sequence[BitBlock]] = tuple(BitBlock(cast(BitsN, p)) for p in product((0, 1), repeat = N))
        assert len(nBlocks) == 2 ** N, len(nBlocks)

        def generateNBitSet(n: int) -> Sequence[BitBlock]:
            ret = tuple(b for b in nBlocks if b.n() == n)
            expectedLength = f(N) / (f(n) * f(N - n))
            assert len(ret) == expectedLength, f"{len(ret)} != {expectedLength}"
            return ret

        nBitsSet: Final[Mapping[int, Sequence[BitBlock]]] = {n: generateNBitSet(n) for n in range(2, N)}

        def compliment(block: BitBlock, n: int, total: int) -> Sequence[BitBlock]:
            assert 2 <= n < N
            assert 3 <= total <= N

            def filterByOverlay(block: BitBlock, blocks: Iterable[BitBlock], n: int) -> Iterable[BitBlock]:
                def overlayBits(a: BitBlock, b: BitBlock) -> BitBlock:  # Simulates overlaying two blocks
                    return BitBlock(tuple(min(*bit) for bit in zip(a, b, strict = True)))

                assert 3 <= n <= N
                return (b for b in blocks if overlayBits(block, b).n() == n)

            ret: Sequence[BitBlock] = tuple(filterByOverlay(block, nBitsSet[n], total))
            assert len(ret) == EXPECTED_COMPLIMENT_LENGTHS[(block.n(), n, total)], (len(ret), block.n(), n, total, block, ret)
            return ret

        blockMap: Final[dict[int, Sequence[ArrayMap]]] = {}

        for a in range(2, N):
            data: list[ArrayMap] = []
            for block in nBitsSet[a]:
                compliments: dict[int, Mapping[int, Sequence[Array]]] = {}
                for n in range(2, N):
                    totals: dict[int, Sequence[Array]] = {}
                    for total in range(3, N + 1):
                        totals[total] = tuple(b.toArray() for b in compliment(block, n, total))
                    compliments[n] = totals
                data.append((block.toArray(), compliments))
            blockMap[a] = tuple(data)

        cls.blockMap = blockMap

    @classmethod
    def getRandomPair(cls, n1: int, n2: int, total: int) -> tuple[Array, Array]:
        assert 2 <= n1 < N
        assert 2 <= n2 < N
        assert 3 <= total <= N
        if not cls.blockMap:
            cls._init()
        assert cls.blockMap
        (a1, nA) = choice(cls.blockMap[n1])
        a2 = choice(nA[n2][total])
        assert np.sum(np.minimum(a1, a2) == 0) == total, (n1, a1, n2, a2, np.minimum(a1, a2), np.minimum(a1, a2) == 0, np.sum(np.minimum(a1, a2) == 0), total)  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        return (a1, a2)

@typechecked
def log(*args: Any) -> None:
    print(*args)

@typechecked
def error(*args: Any) -> None:
    print("ERROR:", *args, file = stderr)
    sysExit(1)

@typechecked
async def loadImage(source: ImagePath, fileName: str | None = None) -> Image:
    """
    Returns `Image` loaded from the specified file path,
    input stream or any bytes `Buffer`.

    `fileName` can be added as a hint to image format in cases
    when `source` does not provide such a clue, like `BytesIO`.
    """
    image = await to_thread(imageOpen, BytesIO(source) if isinstance(source, Buffer) else source)
    if not image.format:
        image.format = getImageFormatFromExtension(fileName or source)
    return image

@typechecked
async def saveImage(image: Image, path: ImagePath) -> None:
    await to_thread(image.save, path, getImageFormatFromExtension(path), optimize = True,  # type: ignore[arg-type]
               transparency = 1 if image.mode == BW1 else None)  # `transparency` here sets the index of the color to make transparent, 1 is usually white

@typechecked
async def imageToBytes(image: Image) -> Buffer:
    """
    Returns the `Image` as bytes `Buffer` that can be written to a file
    with extension corresponding to image format.
    """
    stream = BytesIO()
    await saveImage(image, stream)
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
async def processImage(image: Image,
                       *,
                       resizeFactor: float | int | None = None,  # noqa: PYI041  # beartype is right enforcing this: https://github.com/beartype/beartype/issues/66
                       resizeWidth: int | None = None,
                       resizeHeight: int | None = None,
                       padWidth: int | None = None,
                       padHeight: int | None = None,
                       randomRotate: bool | None = None,
                       randomFlip: bool | None = None,
                       dither: bool | None = None) -> Image | None:
    """
    Converts the arbitrary `Image` to 1-bit B&W with Alpha format,
    performing additional modifications during the process.
    """
    if resizeFactor is not None and (not isinstance(resizeFactor, int | float) or resizeFactor <= 0):
        raise ValueError(f"Bad resizeFactor {resizeFactor}, must be positive int or float")
    if resizeWidth is not None and (not isinstance(resizeWidth, int) or resizeWidth <= 0):
        raise ValueError(f"Bad resizeWidth {resizeWidth}, must be positive int")
    if resizeHeight is not None and (not isinstance(resizeHeight, int) or resizeHeight <= 0):
        raise ValueError(f"Bad resizeHeight {resizeHeight}, must be positive int")
    if padWidth is not None and (not isinstance(padWidth, int) or padWidth <= 0):
        raise ValueError(f"Bad padWidth {padWidth}, must be positive int")
    if padHeight is not None and (not isinstance(padHeight, int) or padHeight <= 0):
        raise ValueError(f"Bad padHeight {padHeight}, must be positive int")
    if bool(padWidth) ^ bool(padHeight):
        raise ValueError("Either both padWidth and padHeight should be specified, or none of them")
    if sum(bool(v) for v in (resizeFactor, resizeWidth or resizeHeight, padWidth or padHeight)) > 1:
        raise ValueError("Either resizeFactor or resizeWidth/resizeHeight or padWidth/padHeight can be specified")
    processed = grayscale = await to_thread(image.convert, GRAYSCALE)  # 8-bit grayscale
    if resizeFactor and resizeFactor != 1:
        processed = await to_thread(imageScale, processed, resizeFactor, RESAMPLING)
    elif resizeWidth or resizeHeight:
        if not resizeHeight:
            assert resizeWidth
            resizeHeight = round(float(processed.height) * resizeWidth / image.width)
        elif not resizeWidth:
            assert resizeHeight
            resizeWidth = round(float(processed.width) * resizeHeight / image.height)
        if (resizeWidth, resizeHeight) != processed.size:
            processed = await to_thread(processed.resize, (resizeWidth, resizeHeight), RESAMPLING)
    elif padWidth and padHeight and (padWidth, padHeight) != processed.size:
        processed = await to_thread(imagePad, processed, (padWidth, padHeight), RESAMPLING, color = 255)  # White background in grayscale
    # ToDo: Should we save rotate angle and flip bit somewhere?
    if randomRotate and (method := choice((None, ROTATE_90, ROTATE_180, ROTATE_270))) is not None:
        processed = await to_thread(processed.transpose, method)
    if randomFlip and choice((False, True)):
        processed = await to_thread(processed.transpose, FLIP)
    if image.mode == BW1 and hasAlpha(image) and processed is grayscale:
        return None  # Indicates that no processing was actually performed and original image could be used as it was
    processed = await to_thread(processed.convert, BW1, dither = Dither.FLOYDSTEINBERG if dither else Dither.NONE)
    finalizeImage(processed)
    return processed

@typechecked
async def encrypt(source: Image, lockMask: Image | None = None, keyMask: Image | None = None, *, smooth: bool | None = None) -> tuple[Image, Image]:
    """
    Generates lock and key images from the specified source `Image`.
    If `lockMask` and/or `keyMask` are provided,
    they are used as visible hints on the corresponding output images.
    """
    # PIL/NumPy array indexes are `y` first, `x` second
    if lockMask or keyMask:
        if lockMask:
            assert lockMask.size == source.size
        else:
            lockMask = imageNew(BW1, source.size, 1)  # White/Transparent background
        if keyMask:
            assert keyMask.size == source.size
        else:
            keyMask = imageNew(BW1, source.size, 1)  # White/Transparent background
        dimensions = (source.height * 2, source.width * 2)
        lockArray = np.empty(dimensions, bool)
        keyArray = np.empty(dimensions, bool)
        lockMaskArray = np.asarray(lockMask, bool)
        keyMaskArray = np.asarray(keyMask, bool)
        for ((y, x), s) in np.ndenumerate(np.asarray(source, bool)):
            if x == 0:  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
                await sleep(0)
            l = lockMaskArray[y, x].item()  # noqa: E741
            k = keyMaskArray[y, x].item()
            s = s.item()  # noqa: PLW2901  # pylint: disable=redefined-loop-name
            x *= 2 ; y *= 2  # noqa: E702, PLW2901  # pylint: disable=multiple-statements, redefined-loop-name
            (a1, a2) = BitBlock.getRandomPair(3 - l, 3 - k, 4 - s)
            lockArray[y : y + 2, x : x + 2] = a1
            keyArray[y : y + 2, x : x + 2] = a2
    elif smooth:
        dimensions = (source.height * 2, source.width * 2)
        lockArray = np.empty(dimensions, bool)
        keyArray = np.empty(dimensions, bool)
        for ((y, x), s) in np.ndenumerate(np.asarray(source, bool)):
            if x == 0:  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
                await sleep(0)
            x *= 2 ; y *= 2  # noqa: E702, PLW2901  # pylint: disable=multiple-statements, redefined-loop-name
            (a1, a2) = BitBlock.getRandomPair(2, 2, 4)
            lockArray[y : y + 2, x : x + 2] = a1
            keyArray[y : y + 2, x : x + 2] = a1 if s else a2
    else:
        lockArray = np.empty(source.size, bool)
        keyArray = np.empty(source.size, bool)
        for ((y, x), s) in np.ndenumerate(np.asarray(source, bool)):
            if x == 0:  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
                await sleep(0)
            lockArray[y, x] = r = choice((False, True))
            keyArray[y, x] = r if s else not r

    await sleep(0)
    lockImage = ImageFromArray(lockArray)
    await sleep(0)
    keyImage = ImageFromArray(keyArray)
    await sleep(0)
    finalizeImage(lockImage)
    finalizeImage(keyImage)
    return (lockImage, keyImage)

@typechecked
async def overlay(lockImage: Image, keyImage: Image, *, border: bool | None = None) -> Image:  # noqa: ARG001  # pylint: disable=unused-argument
    """
    Emulates precise overlaying of two 1-bit images one on top of the other,
    as if they were printed on transparent film.
    """
    assert lockImage.mode == BW1
    assert keyImage.mode == BW1
    assert lockImage.size == keyImage.size
    ret = await to_thread(lambda: ImageFromArray(np.minimum(np.asarray(lockImage), np.asarray(keyImage))))
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
    image = run(loadImage(options.inputImage))
    processed = run(processImage(image, **vars(options))) or image
    run(saveImage(processed, 'processed.png'))  # ToDo: Copy pipeline from `main.py` here
    sysExit(0)

if __name__ == '__main__':
    main(*argv[1:])
else:
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
