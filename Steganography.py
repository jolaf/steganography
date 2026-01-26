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
from secrets import choice, token_bytes
from sys import argv, exit as sysExit, stderr
from time import time
from typing import cast, Any, ClassVar, Final, IO, Literal

try:
    from PIL.Image import fromarray as imageFromArray, frombuffer as imageFromBuffer, new as imageNew, open as imageOpen
    from PIL.Image import Dither, Image, Resampling, Transpose
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

INVERSE_TRANSPOSE: Mapping[Transpose | None, Transpose | None] = {
    None: None,
    ROTATE_90: ROTATE_270,
    ROTATE_180: ROTATE_180,
    ROTATE_270: ROTATE_90,
}

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
def elapsedTime(startTime: float) -> str:
    dt = time() - startTime
    return f"{round(dt)}s" if dt >= 1 else f"{round(dt * 1000)}ms"

@typechecked
def funcName(func: Callable[..., Any]) -> str:
    module = func.__module__
    if not module or module in ('builtin', 'builtins', '__builtin__', '__builtins__'):  # pylint: disable=use-set-for-membership
        return func.__qualname__
    return f"{module}.{func.__qualname__}"

@typechecked
async def timeToThread[T](func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    startTime = time()
    ret = await to_thread(func, *args, **kwargs)
    print(f"[steganography] {elapsedTime(startTime)}: {funcName(func)}")
    return ret

@typechecked
async def loadImage(source: ImagePath, fileName: str | None = None) -> Image:
    """
    Returns `Image` loaded from the specified file path,
    input stream or any bytes `Buffer`.

    `fileName` can be added as a hint to image format in cases
    when `source` does not provide such a clue, like `BytesIO`.
    """
    image = await timeToThread(imageOpen, BytesIO(source) if isinstance(source, Buffer) else source)
    if not image.format:
        image.format = getImageFormatFromExtension(fileName or source)
    return image

@typechecked
async def saveImage(image: Image, path: ImagePath) -> None:
    kwargs = {
        'optimize': True,  # Smallest size but longest compression time
        'transparency': 1,  # Index of the color to make transparent, 1 is usually white
    } if image.mode == BW1 else {
        'optimize': True,
    }
    await timeToThread(image.save, path, getImageFormatFromExtension(path), **kwargs)

@typechecked
async def imageToBytes(image: Image) -> memoryview:
    """
    Returns the `Image` as bytes `Buffer` that can be written to a file
    with extension corresponding to image format.
    """
    stream = BytesIO()
    await saveImage(image, stream)
    return stream.getbuffer()

@typechecked
def imageToJS(image: Image) -> Sequence[Any]:  # ToDo: Rewrite using dict() for better transport readability
    return (image.mode, image.size, image.tobytes(), 'raw', image.mode, 0, 1)

@typechecked
def imageFromJS(serialized: Sequence[Any]) -> Image:
    image = imageFromBuffer(*serialized)
    if not image.format:
        image.format = getImageFormatFromExtension('')
    return image

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
def finalize(image: Image) -> None:
    """Improves further work with 1-bit B&W with Alpha images."""
    assert image.mode == BW1, image.mode
    image.info = {'transparency': 1}
    if not image.format:
        image.format = PNG

@typechecked
async def prepare(image: Image,
                  *,
                  resizeFactor: float | int | None = None,  # noqa: PYI041  # beartype is right enforcing this: https://github.com/beartype/beartype/issues/66
                  resizeWidth: int | None = None,
                  resizeHeight: int | None = None,
                  dither: bool | None = None) -> Image | None:
    """
    Converts the arbitrary `Image` to 1-bit B&W with Alpha format,
    performing additional modifications during the process.
    """
    if resizeFactor is not None and (not isinstance(resizeFactor, float | int) or resizeFactor <= 0):
        raise ValueError(f"Bad resizeFactor {resizeFactor}, must be a positive float or int")
    if resizeWidth is not None and (not isinstance(resizeWidth, int) or resizeWidth <= 0):
        raise ValueError(f"Bad resizeWidth {resizeWidth}, must be a positive int")
    if resizeHeight is not None and (not isinstance(resizeHeight, int) or resizeHeight <= 0):
        raise ValueError(f"Bad resizeHeight {resizeHeight}, must be a positive int")
    if resizeFactor and (resizeWidth or resizeHeight):
        raise ValueError("Either resizeFactor or resizeWidth/resizeHeight can be specified")
    processed = grayscale = await timeToThread(image.convert, GRAYSCALE)  # 8-bit grayscale
    if resizeFactor and resizeFactor != 1:
        processed = await timeToThread(imageScale, processed, resizeFactor, RESAMPLING)
    elif resizeWidth or resizeHeight:
        if not resizeHeight:
            assert resizeWidth
            resizeHeight = round(processed.height * resizeWidth / image.width)
        elif not resizeWidth:
            assert resizeHeight
            resizeWidth = round(processed.width * resizeHeight / image.height)
        if (resizeWidth, resizeHeight) != processed.size:
            processed = await timeToThread(processed.resize, (resizeWidth, resizeHeight), RESAMPLING)
    if image.mode == BW1 and hasAlpha(image) and processed is grayscale:
        return None  # Indicates that no processing was actually performed and original image could be used as it was
    processed = await timeToThread(processed.convert, BW1, dither = Dither.FLOYDSTEINBERG if dither else Dither.NONE)
    finalize(processed)
    return processed

@typechecked
async def encrypt(source: Image,  # noqa: C901
                  lockMask: Image | None = None,
                  keyMask: Image | None = None,
                  *,
                  lockFactor: float | int | None = None,  # noqa: PYI041  # beartype is right enforcing this: https://github.com/beartype/beartype/issues/66
                  lockWidth: int | None = None,
                  lockHeight: int | None = None,
                  randomRotate: bool | None = None,
                  randomFlip: bool | None = None,
                  smooth: bool | None = None) -> tuple[Image, Image, tuple[int, int], Transpose | None, bool]:
    """
    Generates lock and key images from the specified source `Image`.
    If `lockMask` and/or `keyMask` are provided,
    they are used as visible hints on the corresponding output images.
    """
    if lockFactor is not None and (not isinstance(lockFactor, float | int) or lockFactor < 1):
        raise ValueError(f"Bad lockFactor {lockFactor}, must be a positive float no less than 1")
    if lockWidth is not None and (not isinstance(lockWidth, int) or lockWidth <= source.width):
        raise ValueError(f"Bad lockWidth {lockWidth}, must be a positive int at least equal to source.width ({source.width})")
    if lockHeight is not None and (not isinstance(lockHeight, int) or lockHeight <= source.height):
        raise ValueError(f"Bad lockHeight {lockHeight}, must be a positive int at least equal to source.height ({source.height})")
    if lockFactor and (lockWidth or lockHeight):
        raise ValueError("Either lockFactor or lockWidth/lockHeight can be specified")

    if randomRotate:  # Pad source to square for it to be rotatable 90° without visible change of orientation
        if source.width != source.height:
            source = await timeToThread(imagePad, source, (m := max(*source.size), m), RESAMPLING, color = 255)  # white background in grayscale
        assert source.width == source.height
        rotateMethod = choice((None, ROTATE_90, ROTATE_180, ROTATE_270))
    else:
        rotateMethod = None
    flip: bool = bool(randomFlip) and choice((False, True))

    if lockFactor and lockFactor != 1:
        lockSize = (round(source.width * lockFactor),
                    round(source.height * lockFactor))
    elif lockWidth or lockHeight:
        if not lockHeight:
            assert lockWidth
            if source.width == source.height:  # noqa: SIM108
                lockSize = (lockWidth, lockWidth)
            else:
                lockSize = (lockWidth, round(source.height * lockWidth / source.width))
        elif not lockWidth:
            assert lockHeight
            if source.width == source.height:  # noqa: SIM108
                lockSize = (lockHeight, lockHeight)
            else:
                lockSize = (round(source.width * lockHeight / source.height), lockHeight)
        else:
            lockSize = (lockWidth, lockHeight)
    else:
        lockSize = source.size
    (lockWidth, lockHeight) = lockSize

    # (posX, posY) is (random) position of the key relative to the lock
    posX = choice(range(lockWidth - source.width + 1))  # 0 if equal  # ToDo: instead of fully random position, make it along the border
    posY = choice(range(lockHeight - source.height + 1))  # 0 if equal

    if lockMask or keyMask or smooth:
        # 2x2 pixels

        N = 2  # pylint: disable=redefined-outer-name
        (lockWidth, lockHeight) = lockSize = (lockWidth * N, lockHeight * N)
        (keyWidth, keyHeight) = keySize = (source.width * N, source.height * N)

        # PIL/NumPy array indexes are `y` first, `x` second
        lockArray = np.empty((lockHeight, lockWidth), bool)  # These arrays are write-only, so we don't care about the initial values.
        keyArray = np.empty((keyHeight, keyWidth), bool)   # Also creating empty arrays is faster than filling with 1's or 0's.
        # Also empty arrays produce recognizable visual pattern in images, and that allows to notice
        # if some part of the image was not written to, as it should be.

    else:
        # 1x1 pixels

        N = 1
        lockArea = lockWidth * lockHeight
        randomBytes = await timeToThread(token_bytes, (lockArea + 7) // 8)
        randomBytesArray = np.frombuffer(randomBytes, np.uint8)
        unpackedArray = await timeToThread(np.unpackbits, randomBytesArray)
        lockArray = unpackedArray[:lockArea].view(bool).reshape((lockHeight, lockWidth))
        keyArray = np.empty((source.height, source.width), bool)

    sourceArray = np.asarray(source, bool)

    # Color False/0 is black, and True/1 is white/transparent
    if lockMask or keyMask:
        # 2x2 pixels with masks

        if lockMask:
            if lockMask.size != lockSize:
                # ToDo: Make sure (and assert here) that this is padding only, no resizing, because resizing 1-bit image is bad
                # ToDo: If that's not possible, maybe stop processing key and lock images?
                #assert lockMask.width == lock.width or lockMask.height == lock.height  # ToDo: enable after masks are properly resized
                lockMask = await timeToThread(imagePad, lockMask, lockSize, RESAMPLING, color = 255)  # White background in grayscale
        else:
            lockMask = imageNew(BW1, lockSize, 1)  # white/transparent background

        if keyMask:
            if rotateMethod is not None:
                keyMask = await timeToThread(keyMask.transpose, rotateMethod)
            if flip:
                keyMask = await timeToThread(keyMask.transpose, FLIP)
            if keyMask.size != source.size:
                #assert keyMask.width == source.width or keyMask.height == source.height  # ToDo: enable after masks are properly resized
                keyMask = await timeToThread(imagePad, keyMask, source.size, RESAMPLING, color = 255)  # White background in grayscale
        else:
            keyMask = imageNew(BW1, source.size, 1)  # white/transparent background

        lockMaskArray = np.asarray(lockMask, bool)
        keyMaskArray = np.asarray(keyMask, bool)

        for ((y, x), s) in np.ndenumerate(sourceArray):
            if x == 0:  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
                await sleep(0)

            lockMaskValue = lockMaskArray[y + posY, x + posX].item()
            keyMaskValue = keyMaskArray[y, x].item()
            sourceValue = s.item()
            (a1, a2) = BitBlock.getRandomPair(3 - lockMaskValue, 3 - keyMaskValue, 4 - sourceValue)
            (lockX, lockY) = ((x + posX) * N, (y + posY) * N)
            (keyX, keyY) = (x * N, y * N)
            lockArray[lockY : lockY + N, lockX : lockX + N] = a1
            keyArray[keyY : keyY + N, keyX : keyX + N] = a2

    elif smooth:
        # 2x2 pixels without masks

        for lockY in range(0, lockHeight, N):
            for lockX in range(0, lockWidth, N):
                if lockX == 0:  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
                    await sleep(0)

                (a1, a2) = BitBlock.getRandomPair(2, 2, 4)
                lockArray[lockY : lockY + N, lockX : lockX + N] = a1
                x = lockX // N - posX
                if x < 0 or x >= source.width:
                    continue
                y = lockY // N - posY
                if y < 0 or y >= source.height:
                    continue
                s = sourceArray[y, x]
                (keyX, keyY) = (x * N, y * N)
                keyArray[keyY : keyY + N, keyX : keyX + N] = a1 if s else a2

    else:
        # 1x1 pixels

        for ((y, x), s) in np.ndenumerate(sourceArray):
            if x == 0:  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
                await sleep(0)

            r = lockArray[y + posY, x + posX]  # Already filled with random data
            keyArray[y, x] = r if s else not r

    await sleep(0)
    lockImage = imageFromArray(lockArray)
    await sleep(0)
    keyImage = imageFromArray(keyArray)
    await sleep(0)
    if flip:
        keyImage = await timeToThread(keyImage.transpose, FLIP)
    if (inverseRotateMethod := INVERSE_TRANSPOSE[rotateMethod]) is not None:
        keyImage = await timeToThread(keyImage.transpose, inverseRotateMethod)
    await sleep(0)
    finalize(lockImage)
    finalize(keyImage)
    return (lockImage, keyImage, (posX * N, posY * N), rotateMethod, flip)  # `rotateMethod` says what to do with the key for decryption; flip after rotate!

@typechecked
async def overlay(lockImage: Image, keyImage: Image, *, position: tuple[int, int] = (0, 0), rotate: Transpose | None = None, flip: bool | None = None) -> Image:
    """
    Emulates precise overlaying of two 1-bit images one on top of the other,
    as if they were printed on transparent film.
    """
    assert lockImage.mode == BW1, lockImage.mode  # ToDo: replace these asserts with proper ValueError checks
    assert keyImage.mode == BW1, keyImage.mode

    assert lockImage.width >= keyImage.width, (lockImage.size, keyImage.size)
    assert lockImage.height >= keyImage.height, (lockImage.size, keyImage.size)

    (posX, posY) = position
    assert posX >= 0, position
    assert posY >= 0, position

    if rotate is not None:
        keyImage = await timeToThread(keyImage.transpose, rotate)  # ToDo: Check if timeToThread() is really helping, on a large image
    if flip:
        keyImage = await timeToThread(keyImage.transpose, FLIP)

    assert keyImage.width + posX <= lockImage.width, (keyImage.size, position, lockImage.size)
    assert keyImage.height + posY <= lockImage.height, (keyImage.size, position, lockImage.size)

    if lockImage.size != keyImage.size:
        newKeyImage = imageNew(BW1, lockImage.size, 1)  # white/transparent background
        newKeyImage.paste(keyImage, position)
        keyImage = newKeyImage

    assert lockImage.size == keyImage.size, (lockImage.size, keyImage.size)
    lockArray = await timeToThread(np.asarray, lockImage)
    keyArray = await timeToThread(np.asarray, keyImage)
    minArray = await timeToThread(np.minimum, lockArray, keyArray)
    ret = await timeToThread(imageFromArray, minArray)
    finalize(ret)
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
    processed = run(prepare(image, **vars(options))) or image
    run(saveImage(processed, 'processed.png'))  # ToDo: Copy pipeline from `main.py` here
    sysExit(0)

if __name__ == '__main__':
    main(*argv[1:])
else:
    __all__ = (
        'Image',
        'ImagePath',
        'Transpose',
        'encrypt',
        'getImageMode',
        'getMimeTypeFromImage',
        'imageFromJS',
        'imageToBytes',
        'imageToJS',
        'loadImage',
        'overlay',
        'prepare',
    )
