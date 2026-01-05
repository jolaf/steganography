# ruff: noqa: E402  # pylint: disable=wrong-import-position
# Note: this module is PyScript-only, it won't work outside of browser
from __future__ import annotations

print("[python] Loading app")

from asyncio import create_task, get_running_loop, sleep, AbstractEventLoop
from collections.abc import Callable, Iterable, Iterator, Mapping  # noqa: TC003
from contextlib import suppress
from enum import Enum
from gettext import translation, GNUTranslations
from itertools import chain
from re import findall, match, split
import sys
from sys import version as pythonVersion
from traceback import extract_tb
from types import TracebackType  # noqa: TC003
from typing import cast, Any, ClassVar, TYPE_CHECKING

try:
    from coolname import generate_slug  # type: ignore[attr-defined]
    def getRandomName() -> str:
        return cast(str, generate_slug(2))
except ImportError:
    print("[python] WARNING: coolname is not available, using 'steganography' as a sample name")
    def getRandomName() -> str:
        return 'steganography'

try:
    from beartype import beartype as typechecked
except ImportError:
    print("[python] WARNING: beartype is not available, typing is unchecked")
    def typechecked(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[no-redef]
        return func

from pyscript import config as pyscriptConfig, when, storage, Storage  # type: ignore[attr-defined]  # pylint: disable=no-name-in-module
from pyscript.web import page  # type: ignore[import-not-found]  # pylint: disable=import-error, no-name-in-module
from pyscript.ffi import to_js  # type: ignore[import-not-found]  # pylint: disable=import-error, no-name-in-module

try:  # Try to identify PyScript version
    from pyscript import version as pyscriptVersion  # type: ignore[attr-defined]
except ImportError:
    try:
        from pyscript import __version__ as pyscriptVersion
    except ImportError:
        urls = tuple(url for url in (tag.src for tag in page['script']) if url.endswith('core.js'))
        pyscriptVersion = urls[0].split('/')[-2] if urls else "UNKNOWN"

if TYPE_CHECKING:
    def _(s: str) -> str:  # stub for gettext string wrapper
        return s

    # Workarounds for mypy, as stuff cannot be imported from PyScript when not in a browser
    type Element = Any
    Node = Any  # beartype objects if we use 'type' here
    NodeFilter = Any  # beartype objects if we use 'type' here
    type TreeWalker = Any

    def newCSSStyleSheet() -> CSSStyleSheet:
        pass
    def newBlob(_blobParts: Iterable[Any], _options: Any) -> Blob:
        pass
    def newFile(_fileBits: Iterable[Any], _fileName: str, _options: Any) -> File:
        pass
    def newUint8Array(_bytes: bytes) -> Uint8Array:
        pass
    def newEvent(_name: str) -> Event:
        pass
    def createObjectURL(_file: File) -> str:
        return ''
    def revokeObjectURL(_url: str) -> None:
        pass
    def createTreeWalker(_root: Element, _whatToShow: int = -1, _filter: Callable[[Node], int] | None = None) -> TreeWalker:
        pass
    def reload() -> None:
        pass
    adoptedStyleSheets = Any
else:
    from pyscript import document  # type: ignore[attr-defined]
    from pyscript.web import Element  # type: ignore[import-not-found]

    from js import location, Blob, CSSStyleSheet, Event, File, Node, NodeFilter, Text, TreeWalker, Uint8Array, URL  # type: ignore[attr-defined]

    # We'll redefine these classes to Any below, so we have to save all needed references
    newCSSStyleSheet = CSSStyleSheet.new
    newBlob = Blob.new
    newFile = File.new
    newUint8Array = Uint8Array.new
    newEvent = Event.new

    # Simplifying addressing to JS functions
    createObjectURL = URL.createObjectURL
    revokeObjectURL = URL.revokeObjectURL
    del URL  # We won't need it anymore
    adoptedStyleSheets = document.adoptedStyleSheets
    createTreeWalker = document.createTreeWalker
    del document  # We won't need it anymore
    reload = location.reload
    del location  # We won't need it anymore

# JS types that don't work as runtime type annotations
# noinspection PyTypeAliasRedeclaration
type CSSStyleSheet = Any
Blob = Any  # beartype objects if we use 'type' here
# noinspection PyTypeAliasRedeclaration
type Event = Any  # Neither PyScript nor JS versions work as runtime type annotation
# noinspection PyTypeAliasRedeclaration
type File = Any
# noinspection PyTypeAliasRedeclaration
type Text = Any
# noinspection PyTypeAliasRedeclaration
type Uint8Array = Any

try:
    from pyodide_js import version as pyodideVersion  # type: ignore[import-not-found]
except ImportError:
    pyodideVersion = "UNKNOWN"

from Steganography import getImageMode, getMimeTypeFromImage, imageToBytes, loadImage, processImage, synthesize, testOverlay, Image

TagAttrValue = str | int | float | bool

# Event names
CLICK = 'click'
CHANGE = 'change'

TEXT = 'text'  # Shortcut for innerText attribute

@typechecked
def log(*args: Any) -> None:
    message = ' '.join(str(arg) for arg in args)
    print("[python]", message)
    logTag = getTagByID('log')
    # noinspection PyProtectedMember
    logTag._dom_element.append(message + '\n')  # We have to use _dom_element because PyScript version of append() is not working with strings as of v2025.11.2  # noqa: SLF001  # pylint: disable=protected-access
    test = message.lower()
    if any(word in test for word in ('error', 'exception')):
        logTag.classes.add('error')

@typechecked
def getFileNameFromPath(path: str) -> str:
    # It looks like 'pathlib' and `os` modules fail to parse `C:\fakepath\` paths generated by browser when uploading files
    return split(r'[/\\]', path)[-1]

@typechecked
async def repaint() -> None:
    await sleep(0.1)  # Yield control to the browser so that repaint could happen

@typechecked
def createObjectURLFromBytes(byteArray: bytes, mimeType: str) -> str:
    if isinstance(byteArray, bytes):
        byteArray = newUint8Array(byteArray)
    blob = newBlob([byteArray,], to_js({'type': mimeType}))  # to_js() converts Python dict into JS object
    return createObjectURL(blob)

@typechecked
async def blobToBytes(blob: Blob) -> bytes:
    return cast(bytes, (await blob.arrayBuffer()).to_bytes())

@typechecked
def getTagByID(tagID: str) -> Element:
    try:
        return page['#' + tagID][0]
    except IndexError:
        log("ERROR at getTagByID(): No tag ID found:", tagID)
        raise

@typechecked
def hide(element: str | Element) -> None:
    if isinstance(element, str):
        element = getTagByID(element)
    element.classes.add('hidden')

@typechecked
def show(element: str | Element) -> None:
    if isinstance(element, str):
        element = getTagByID(element)
    element.classes.remove('hidden')

@typechecked
def getAttr(element: str | Element, attr: str) -> str:
    if isinstance(element, str):
        element = getTagByID(element)
    if attr == TEXT:
        return cast(str, element.textContent)
    return cast(str, element.getAttribute(attr))

@typechecked
def setAttr(element: str | Element, attr: str, value: TagAttrValue, onlyIfAbsent: bool = False) -> None:
    if isinstance(element, str):
        element = getTagByID(element)
    if attr == TEXT:
        if not onlyIfAbsent or not element.textContent:
            element.textContent = value
    elif not onlyIfAbsent or not element.getAttribute(attr):
        element.setAttribute(attr, value)

@typechecked
def dispatchEvent(element: str | Element, eventType: str) -> None:
    if isinstance(element, str):
        element = getTagByID(element)
    element.dispatchEvent(newEvent(eventType))

@typechecked
def iterTextNodes(root: Element | None = None) -> Iterator[Text]:
    # noinspection PyProtectedMember
    walker = createTreeWalker((root or page.html)._dom_element, NodeFilter.SHOW_TEXT)  # noqa: SLF001  # pylint: disable=protected-access
    while node := walker.nextNode():
        assert node.nodeType == Node.TEXT_NODE, node.nodeType
        yield node

@typechecked
class Options(Storage):  # type: ignore[misc, no-any-unimported]
    TAG_ARGS: ClassVar[Mapping[type[TagAttrValue], Mapping[str, TagAttrValue]]] = {
        str: {
        },
        int: {
            'inputmode': 'numeric',
            'placeholder': 'integer',
            'pattern': r'[0-9]+',
            'default': 0,
        },
        float: {
            'inputmode': 'decimal',
            'placeholder': 'float resize factor',
            'pattern': r'\.[0-9]{1,2}|[0-9]+\.[0-9]{1,2}|[0-9]+\.?',
            'default': 1.0,
        },
        bool: {
            'type': 'checkbox',
            'default': False,
        },
    }

    LANGUAGES: ClassVar[Mapping[str, str]] = {
        'en_US': "English",
        'ru_RU': "Русский",
    }

    TRANSLATIONS: ClassVar[Mapping[str, GNUTranslations]] = {language: translation('Steganography', './gettext/', (language,)) for language in LANGUAGES}

    @classmethod
    def setLanguage(cls, language: str) -> None:
        (tr := cls.TRANSLATIONS[language]).install()
        if _('GETTEXT_TEST') != 'GETTEXT_TEST_' + language:
            log(f"ERROR: gettext.{type(tr).__name__}({language}) is not configured properly, expected GETTEXT_TEST_{language}, got {_('GETTEXT_TEST')}")
            return
        log("Language set to", language)
        for textNode in iterTextNodes():  # ToDo: exclude log from parsing, probably by whitelisting valid roots and chaining results
            if (value := textNode.nodeValue).isspace():
                continue
            assert value
            if m := match(r'^(\W*)(.*?)(\W*)$', value):  # ToDo: Improve regex to exclude digits from middle group
                (prefix, translatable, suffix) = m.groups()
                if not translatable:
                    continue
                if (translated := _(translatable)) == translatable:
                    continue
                textNode.nodeValue = f'{prefix}{translated}{suffix}'

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # Constructor is only called internally, so we don't know the args and don't care
        super().__init__(*args, **kwargs)

        # These fields define names, types and DEFAULT values for options, actual values are stored in Storage
        self.language = next(iter(self.LANGUAGES))
        self.taskName = ''
        self.maxPreviewWidth = 500
        self.maxPreviewHeight = 200
        self.resizeFactor = 1.0
        self.resizeWidth = 0
        self.resizeHeight = 0
        self.cropWidth = 0
        self.cropHeight = 0
        self.keyFactor = 1.0
        self.keyWidth = 0
        self.keyHeight = 0
        self.keyRotate = False
        self.lockFactor = 1.0
        self.lockWidth = 0
        self.lockHeight = 0
        self.keyResize = False
        self.dither = False
        self.smooth = False

        tags: list[Element] = []
        for (name, defaultValue) in vars(self).items():
            if isinstance(defaultValue, TagAttrValue):
                tags.append(self.configureTag(name, defaultValue))

        self.styleSheet = newCSSStyleSheet()
        adoptedStyleSheets.push(self.styleSheet)
        self.updateStyle()

        @when(CLICK, '#options-reset')  # type: ignore[untyped-decorator]
        @typechecked
        def resetEventHandler(_e: Event) -> None:
            for tag in tags:
                if tag.type == 'checkbox':
                    tag.checked = tag.default
                else:
                    tag.value = tag.default
                dispatchEvent(tag, CHANGE)

    def configureTag(self, name: str, defaultValue: TagAttrValue) -> Element:
        valueType = type(defaultValue)
        value = self.get(name, defaultValue)  # Read from database
        assert isinstance(value, valueType), f"Incorrect type for option {name}: {type(value).__name__}, expected {valueType.__name__}"
        tagID = '-'.join(chain(('option',), (word.lower() for word in findall(r'[a-z]+|[A-Z][a-z]*|[A-Z]+', name))))
        tag = getTagByID(tagID)
        assert tag.tagName.lower() in ('input', 'select'), tag.tagName  # pylint: disable=use-set-for-membership
        tag.default = defaultValue
        if name == 'taskName':
            exclude = r'''\/:\\?*'<">&\|'''  # ToDo: Generate title and placeholder too
            setAttr(tag, 'pattern', rf'[^{exclude}\s][^{exclude}]+[^{exclude}\s]')  # Also avoid whitespace at start and end
        elif name == 'language':
            tag.innerHTML = ''.join(f'<option value="{lang}">{name}</option>' for (lang, name) in self.LANGUAGES.items())
            self.setLanguage(value)
        if tag.tagName.lower() == 'input':
            args = self.TAG_ARGS[valueType]
            for (attr, attrValue) in args.items():
                setAttr(tag, attr, attrValue, onlyIfAbsent = True)  # Set <input> tag attributes according to value type
            inputType = args.get('type', 'text')
            setAttr(tag, 'type', inputType, onlyIfAbsent = True)  # Make sure <input type="text"> is always specified explicitly
            if inputType == 'text':
                setAttr(tag, 'maxlength', 4, onlyIfAbsent = True)
        else:
            inputType = None  # For select tags
        if inputType == 'checkbox':
            tag.checked = value
        else:
            tag.value = value

        @when(CHANGE, tag)  # type: ignore[untyped-decorator]
        @typechecked
        async def changeEventHandler(_e: Event) -> None:
            newValue: TagAttrValue = tag.value
            assert isinstance(newValue, str), type(newValue)
            if valueType is str:
                if name == 'taskName' and not newValue:
                    newValue = getRandomName()  # Provide random task name to save user from extra thinking
                elif name == 'language':
                    reload()
            elif valueType is bool:
                newValue = tag.checked
            elif newValue:  # int or float from non-empty string
                try:
                    newValue = valueType(newValue)
                except ValueError:
                    newValue = defaultValue
            else:  # empty string
                newValue = defaultValue
            self[name] = newValue  # Save to database
            tag.value = newValue  # Write processed value back to the input tag
            if name in ('maxPreviewWidth', 'maxPreviewHeight'):  # pylint: disable=use-set-for-membership
                self.updateStyle()
            await self.sync()  # Make sure database is really updated

        return tag

    def updateStyle(self) -> None:
        self.styleSheet.replaceSync(f'''
.image-display {{
    max-width: {self.maxPreviewWidth}px;
    max-height: {self.maxPreviewHeight}px;
}}
        ''')

    def saveFile(self, name: str, data: bytes | None, fileName: str | None = None) -> None:
        if data:
            assert fileName
            self[f'file-{name}'] = bytearray(data)
            self[f'file-{name}-fileName'] = fileName
        else:
            self.delFile(name)

    def loadFile(self, name: str) -> tuple[bytes | None, str | None]:
        data: bytearray = self.get(f'file-{name}')
        # noinspection PyRedundantParentheses
        return (bytes(data) if data else None, self.get(f'file-{name}-fileName'))

    def delFile(self, name: str) -> None:
        with suppress(KeyError):
            del self[f'file-{name}']
        with suppress(KeyError):
            del self[f'file-{name}-fileName']

    def __getattribute__(self, name: str) -> Any:
        defaultValue = super().__getattribute__(name)
        if not isinstance(defaultValue, TagAttrValue):
            return defaultValue  # Not an option field
        ret = self.get(name, None)
        return defaultValue if ret is None else ret

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            defaultValue = super().__getattribute__(name)
            if isinstance(defaultValue, TagAttrValue):
                if isinstance(defaultValue, float):
                    assert isinstance(value, int | float), f"Incorrect type for option {name}: {type(value).__name__}, expected int or float"
                else:
                    assert isinstance(value, type(defaultValue)), f"Incorrect type for option {name}: {type(value).__name__}, expected {type(defaultValue).__name__}"
                self[name] = value
                self.sync()
                return
        except AttributeError:
            pass
        super().__setattr__(name, value)

@typechecked
class Stage(Enum):
    SOURCE = 1
    LOCK = 2
    KEY = 3
    PROCESSED_SOURCE = 4
    PROCESSED_LOCK = 5
    PROCESSED_KEY = 6
    GENERATED_LOCK = 7
    GENERATED_KEY = 8
    TEST = 9

@typechecked
class ImageBlock:
    ID_PREFIX: ClassVar[str] = 'image-'

    DEPENDENCIES: ClassVar[Mapping[Stage, tuple[Stage, ...]]] = {
        Stage.SOURCE: (Stage.PROCESSED_SOURCE,),
        Stage.LOCK: (Stage.PROCESSED_LOCK,),
        Stage.KEY: (Stage.PROCESSED_KEY,),
        Stage.PROCESSED_SOURCE: (Stage.GENERATED_LOCK, Stage.GENERATED_KEY),
        Stage.PROCESSED_LOCK: (Stage.GENERATED_LOCK, Stage.GENERATED_KEY),
        Stage.PROCESSED_KEY: (Stage.GENERATED_LOCK, Stage.GENERATED_KEY),
        Stage.GENERATED_LOCK: (Stage.TEST,),
        Stage.GENERATED_KEY: (Stage.TEST,),
        Stage.TEST: (),
    }

    ImageBlocks: ClassVar[dict[Stage, ImageBlock]] = {}

    options: ClassVar[Options | None] = None

    @classmethod
    async def init(cls) -> None:
        assert cls.options is None  # This method should only be called once
        log("Initializing Options")
        cls.options = await storage('steganography', storage_class = Options)
        log("Rendering HTML blocks")
        await repaint()
        for stage in Stage:
            cls.ImageBlocks[stage] = ImageBlock(stage)
            await repaint()
        for (stage, block) in cls.ImageBlocks.items():
            block.dependencies = tuple(cls.ImageBlocks[s] for s in cls.DEPENDENCIES[stage])

    @classmethod
    async def process(cls, targetStages: Stage | Iterable[Stage], processFunction: Callable[..., Image | tuple[Image, ...]], sourceStages: Stage | Iterable[Stage]) -> None:
        assert cls.options is not None
        sources = tuple(cls.ImageBlocks[stage] for stage in ((sourceStages,) if isinstance(sourceStages, Stage) else sourceStages))
        targets = tuple(cls.ImageBlocks[stage] for stage in ((targetStages,) if isinstance(targetStages, Stage) else targetStages))
        for t in targets:
            t.startOperation(_("Processing image"))
        await repaint()
        try:
            ret = processFunction(*(source.image for source in sources), **vars(cls.options))
            if isinstance(ret, Image):
                ret = (ret,)
            assert len(ret) == len(targets), f"{processFunction.__name__} returned {len(ret)} images, expected {len(targets)}"
            for (target, image) in zip(targets, ret, strict = True):
                target.completeOperation(image, imageToBytes(image))
        except Exception as ex:  # noqa : BLE001
            for target in targets:
                target.error(_("processing image"), ex)
            return

    @classmethod
    async def pipeline(cls) -> None:  # Called from upload event handler to generate secondary images
        await cls.process(Stage.PROCESSED_SOURCE, processImage, Stage.SOURCE)
        await cls.process(Stage.PROCESSED_LOCK, processImage, Stage.LOCK)
        await cls.process(Stage.PROCESSED_KEY, processImage, Stage.KEY)
        if not cls.ImageBlocks[Stage.PROCESSED_SOURCE].image:
            return
        await cls.process((Stage.GENERATED_LOCK, Stage.GENERATED_KEY), synthesize, (Stage.PROCESSED_LOCK, Stage.PROCESSED_KEY))
        await cls.process(Stage.TEST, testOverlay, (Stage.GENERATED_LOCK, Stage.GENERATED_KEY))  # ToDo: Somehow handle random rotation and location

    def __init__(self, stage: Stage) -> None:
        self.name = stage.name.lower()
        log("Stage", self.name)
        self.isUpload = (stage.value <= Stage.LOCK.value)  # pylint: disable=superfluous-parens
        self.isProcessed = not self.isUpload and stage.value <= Stage.PROCESSED_KEY.value
        self.isGenerated = not self.isUpload and not self.isProcessed
        self.image: Image | None = None
        self.fileName: str | None = None
        self.dependencies: tuple[ImageBlock, ...] = ()

        # Create DOM tag
        block = getTagByID('template').clone(self.getTagID('block'))
        getTagByID('uploaded' if self.isUpload else 'generated').append(block)  # ToDo: Add third section: uploaded, processed, generated?

        # Assign named IDs to all children that have image-* classes
        for tag in block.find('*'):
            for clas in tag.classes:
                if clas.startswith(self.ID_PREFIX):
                    tag.id = f'{clas}-{self.name}'
                    break

        # Adjust children attributes
        self.setAttr('title', TEXT, self.name.capitalize() + " file")  # ToDo: Translate

        if self.isUpload:
            self.hide('description')
            self.show('block')
        else:
            self.hide('upload-block')

        (imageBytes, fileName) = self.loadFile()
        if imageBytes:
            try:
                image = loadImage(imageBytes, fileName)
            except Exception as ex:  # noqa : BLE001
                self.error(_("loading image"), ex)
                return
            self.setFileName(fileName)
            self.completeOperation(image, imageToBytes(image))
        else:
            self.setFileName()

        if self.isUpload:
            uploadTag = self.getTag('upload')

            @when(CHANGE, uploadTag)  # type: ignore[untyped-decorator]
            @typechecked
            async def uploadEventHandler(_e: Event) -> None:
                # noinspection PyShadowingNames
                if fileName := uploadTag.value:
                    await self.upload(getFileNameFromPath(fileName), uploadTag.files.item(0))
                else:
                    await self.upload()  # E.g. Esc was pressed at upload dialog

            @when(CLICK, self.getTag('remove'))  # type: ignore[untyped-decorator]
            @typechecked
            async def removeEventHandler(_e: Event) -> None:
                uploadTag.value = ''
                self.remove()

    def getTagID(self, detail: str) -> str:
        return f'{self.ID_PREFIX}{detail}-{self.name}'

    def getTag(self, detail: str) -> Element:
        return getTagByID(self.getTagID(detail))

    def hide(self, name: str) -> None:
        hide(self.getTag(name))

    def show(self, name: str) -> None:
        show(self.getTag(name))

    def getAttr(self, name: str, attr: str) -> str:
        return getAttr(self.getTag(name), attr)

    def setAttr(self, name: str, attr: str, value: str) -> None:
        setAttr(self.getTag(name), attr, value)

    def setFileName(self, fileName: str | None = None) -> None:
        assert self.options is not None
        if not fileName:
            fileName = f'{self.options.taskName}-{self.name}.png'
        self.fileName = fileName
        self.setAttr('name', TEXT, fileName)
        self.setAttr('download-link', 'download', fileName)

    def setURL(self, url: str = '') -> None:
        self.setAttr('display', 'src', url)
        self.setAttr('display-link', 'href', url)
        self.setAttr('download-link', 'href', url)

    def setDescription(self, message: str) -> None:
        self.setAttr('description', TEXT, message)
        self.show('description')

    def error(self, message: str, exception: BaseException | None = None) -> None:
        self.delFile()
        self.setDescription(f"{_("ERROR")} {message}{f": {exception}" if exception else ''}")
        self.hide('remove')

    def resetBlock(self, description: str | None = None) -> None:
        if src := self.getAttr('display', 'src'):
            revokeObjectURL(src)
        self.setURL()
        self.hide('display-block')
        self.hide('remove')
        if description:
            self.setDescription(description)
            self.show('block')
        else:
            self.hide('description')
            if not self.isUpload:
                self.hide('block')

    def startOperation(self, message: str) -> None:
        self.resetBlock(message + "…")

    def completeOperation(self, image: Image, imageBytes: bytes) -> None:
        self.image = image
        self.setDescription(f"{len(imageBytes)} {_("bytes")} {image.format} {getImageMode(image)} {image.width}x{image.height}")
        self.setURL(createObjectURLFromBytes(imageBytes, getMimeTypeFromImage(image)))
        self.show('display-block')
        self.show('block')
        if self.isUpload:
            self.show('remove')

    def saveFile(self, byteArray: bytes) -> None:
        assert self.options is not None
        assert self.fileName
        self.options.saveFile(self.name, byteArray, self.fileName)

    def loadFile(self) -> tuple[bytes | None, str | None]:
        assert self.options is not None
        return self.options.loadFile(self.name)

    def delFile(self) -> None:
        assert self.options is not None
        self.options.delFile(self.name)

    def remove(self) -> None:
        self.delFile()
        self.resetBlock()
        for block in self.dependencies:
            block.remove()

    async def upload(self, fileName: str | None = None, data: Blob | None = None) -> None:
        assert self.isUpload
        if not fileName:  # E.g. Esc was pressed at upload dialog
            return
        self.startOperation(_("Loading image"))
        await repaint()
        try:
            assert data, data
            imageBytes = await blobToBytes(data)
            image = loadImage(imageBytes)
        except Exception as ex:  # noqa : BLE001
            self.error(_("loading image"), ex)
            return
        self.setFileName(fileName)
        self.saveFile(imageBytes)
        self.completeOperation(image, imageBytes)
        self.show('remove')
        await self.pipeline()

@typechecked
def exceptionHandler(source: str, exceptionType: type[BaseException | None] | None = None, exception: BaseException | None = None, traceback: TracebackType | None = None) -> None:
    if exceptionType is None:
        exceptionType = type(exception)
    if traceback is None and exception:
        traceback = exception.__traceback__
    # Filter traceback to remove empty lines
    tracebackStr = '\n' + '\n'.join(line for line in '\n'.join(extract_tb(traceback).format()).splitlines() if line.strip()) if traceback else ''
    log(f"\nERROR Uncaught exception in {source}, type {exceptionType.__name__}: {exception}{tracebackStr}\n\nPlease make a screenshot and report it to @jolaf at Telegram or VK or to vmzakhar@gmail.com. Thank you!\n")

@typechecked
def mainExceptionHandler(exceptionType: type[BaseException] | None = None, exception: BaseException | None = None, traceback: TracebackType | None = None) -> None:
    exceptionHandler("main thread", exceptionType, exception, traceback)

@typechecked
def loopExceptionHandler(_loop: AbstractEventLoop, context: dict[str, Any]) -> None:
    exceptionHandler("async loop", exception = context.get('exception'))

@typechecked
async def main() -> None:
    log("Starting app")
    log("PyScript v" + pyscriptVersion)
    log("Pyodide v" + pyodideVersion)
    log("Python v" + pythonVersion)
    log("PyScript config:", pyscriptConfig)
    sys.excepthook = mainExceptionHandler
    get_running_loop().set_exception_handler(loopExceptionHandler)
    await ImageBlock.init()
    hide('log')
    show('content')
    log("Running app")

if __name__ == '__main__':
    create_task(main())  # noqa: RUF006

print("[python] Loaded app")
