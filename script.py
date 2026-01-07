# ruff: noqa: E402  # pylint: disable=wrong-import-position
# Note: this module is PyScript-only, it won't work outside of browser
from __future__ import annotations

PREFIX = "[python]"
print(f"{PREFIX} Loading app")

from asyncio import create_task, gather, get_running_loop, sleep, AbstractEventLoop
from collections.abc import Callable, Iterable, Iterator, Mapping  # noqa: TC003
from contextlib import suppress
from datetime import datetime
from enum import Enum
# noinspection PyUnresolvedReferences
from gettext import translation, GNUTranslations
from itertools import chain
from re import findall, match, split
import sys
from sys import version as pythonVersion
from traceback import extract_tb
from types import TracebackType  # noqa: TC003
from typing import cast, Any, ClassVar, TYPE_CHECKING

try:
    from beartype import beartype as typechecked
except ImportError:
    print(f"{PREFIX} WARNING: beartype is not available, typing is unchecked")
    def typechecked(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[no-redef]
        return func

try:
    from coolname import generate_slug  # type: ignore[attr-defined]

    @typechecked
    def getDefaultTaskName() -> str:
        return cast(str, generate_slug(2))
except ImportError:
    print(f"{PREFIX} WARNING: coolname is not available, using 'steganography' as a sample name")

    @typechecked
    def getDefaultTaskName() -> str:
        return "steganography"

# noinspection PyUnresolvedReferences
from pyscript import config as pyscriptConfig, document, when, storage, Storage
from pyscript.web import page, Element  # pylint: disable=import-error, no-name-in-module
from pyscript.ffi import to_js  # pylint: disable=import-error, no-name-in-module

try:  # Try to identify PyScript version
    from pyscript import version as pyscriptVersion  # type: ignore[attr-defined]
except ImportError:
    try:
        from pyscript import __version__ as pyscriptVersion  # type: ignore[attr-defined]
    except ImportError:
        urls = tuple(url for url in (element.src for element in page['script']) if url.endswith('core.js'))
        pyscriptVersion = urls[0].split('/')[-2] if urls else "UNKNOWN"

if TYPE_CHECKING:  # This branch is for IDEs and mypy only, beartype ignores it
    from pyscript.events import Event

    # Workarounds for mypy, for things that do not have proper typing stubs (yet?)
    class Blob:  # ToDo: Move these to stubs?
        async def arrayBuffer(self) -> int: ...  # pylint: disable=no-self-use

    class ArrayBuffer:
        def to_bytes(self) -> bytes: ...  # pylint: disable=no-self-use

    class CSSStyleSheet:
        def replaceSync(self, _text: str) -> None: ...  # pylint: disable=no-self-use

    class JsNull: ...

    class JsProxy:
        nodeType: int
        nodeValue: str
        def append(self, *_args: Node | str) -> None: ...  # pylint: disable=no-self-use
        async def arrayBuffer(self) -> ArrayBuffer: ...  # pylint: disable=no-self-use

    class Node:
        TEXT_NODE: ClassVar[int]
        nodeType: int
        nodeValue: str
        def append(self, *_args: Node | str) -> None: ...  # pylint: disable=no-self-use

    class NodeFilter:
        SHOW_TEXT: ClassVar[int]

    class Text(Node): ...

    class TreeWalker:
        currentNode: JsProxy
        def nextNode(self) -> Node: ...  # pylint: disable=no-self-use

    class Uint8Array: ...

    def newCSSStyleSheet() -> CSSStyleSheet: ...
    def newBlob(_blobParts: Iterable[Any], _options: Any) -> Blob: ...
    def newUint8Array(_bytes: bytes) -> Uint8Array: ...
    def newEvent(_name: str) -> Event: ...
    def createObjectURL(_blob: Blob) -> str: ...
    def revokeObjectURL(_url: str) -> None: ...
    def createTreeWalker(_root: Element | JsProxy, _whatToShow: int | None = ..., _filter: Any | None = ...) -> TreeWalker: ...
    def querySelectorAll(_selector: str) -> Iterable[Element]: ...
    def reload() -> None: ...
    adoptedStyleSheets = Any
else:  # beartype follows this branch
    from js import location, Blob, CSSStyleSheet, Event, Node, NodeFilter, Text, TreeWalker, Uint8Array, URL  # type: ignore[attr-defined]
    # noinspection PyUnresolvedReferences
    from pyodide.ffi import JsNull, JsProxy

    # We'll redefine these classes to JsProxy below, so we have to save all needed references
    newCSSStyleSheet = CSSStyleSheet.new
    newBlob = Blob.new
    newUint8Array = Uint8Array.new
    newEvent = Event.new

    # Simplifying addressing to JS functions
    createObjectURL = URL.createObjectURL
    revokeObjectURL = URL.revokeObjectURL
    del URL  # We won't need it anymore
    adoptedStyleSheets = document.adoptedStyleSheets
    createTreeWalker = document.createTreeWalker
    querySelectorAll = document.querySelectorAll
    del document  # We won't need it anymore
    reload = location.reload
    del location  # We won't need it anymore

TEXT_NODE = Node.TEXT_NODE

# JS types that don't work as runtime type annotations
# noinspection PyTypeAliasRedeclaration
type Blob = JsProxy  # type: ignore[no-redef]
# noinspection PyTypeAliasRedeclaration
type Event = JsProxy  # type: ignore[no-redef]
# noinspection PyTypeAliasRedeclaration
type Node = JsProxy  # type: ignore[no-redef]

try:
    from pyodide_js import version as pyodideVersion  # type: ignore[import-not-found]
except ImportError:
    pyodideVersion = "UNKNOWN"

from Steganography import getImageMode, getMimeTypeFromImage, imageToBytes, loadImage, processImage, synthesize, testOverlay, Image

TagAttrValue = str | int | float | bool

# Tag names
INPUT = 'INPUT'
SELECT = 'SELECT'

# <INPUT> types
TEXT = 'text'  # Also is used as a shortcut for innerText attribute
CHECKBOX = 'checkbox'

# Event names
CLICK = 'click'
CHANGE = 'change'

# Class names
HIDDEN = 'hidden'

# Misc
GETTEXT_TEST = 'GETTEXT_TEST'

@typechecked
def _(s: str) -> str:  # Will be replaced by gettext
    return s

@typechecked
def toJsElement(element: Element | JsProxy) -> JsProxy:
    return getattr(element, '_dom_element', element)  # type: ignore[arg-type]

@typechecked
def log(*args: Any) -> None:
    message = ' '.join(str(arg) for arg in args)
    print(f"{PREFIX} {message}")
    logElement = getElementByID('log')
    toJsElement(logElement).append(f"{datetime.now().astimezone().strftime('%H:%M:%S')} {PREFIX} {message}\n")  # ToDo: file it?  # We have to use toJsElement() because PyScript version of append() is not working with strings as of v2025.11.2
    test = message.upper()
    if any(word in test for word in ('ERROR', 'EXCEPTION')):
        logElement.classes.add('error')

@typechecked
def getFileNameFromPath(path: str) -> str:
    # It looks like 'pathlib' and `os` modules fail to parse `C:\fakepath\` paths generated by browser when uploading files
    return split(r'[/\\]', path)[-1]

@typechecked
async def repaint() -> None:
    await sleep(0.1)  # Yield control to the browser so that repaint could happen

@typechecked
def createObjectURLFromBytes(byteArray: bytes, mimeType: str) -> str:
    blob = newBlob([newUint8Array(byteArray),], to_js({'type': mimeType}))  # to_js() converts Python dict into JS object
    return createObjectURL(blob)

@typechecked
async def blobToBytes(blob: Blob) -> bytes:
    return (await blob.arrayBuffer()).to_bytes()

@typechecked
def getElementByID(elementID: str) -> Element:
    try:
        return page['#' + elementID][0]
    except IndexError:
        log("ERROR at getElementByID(): No element ID found:", elementID)
        raise

@typechecked
def hide(element: str | Element) -> None:
    if isinstance(element, str):
        element = getElementByID(element)
    element.classes.add(HIDDEN)

@typechecked
def show(element: str | Element) -> None:
    if isinstance(element, str):
        element = getElementByID(element)
    element.classes.remove(HIDDEN)

@typechecked
def getAttr(element: str | Element, attr: str, default: str | None = None) -> str | None:
    if isinstance(element, str):
        element = getElementByID(element)
    assert isinstance(element, Element), type(element).__name__
    ret = element.textContent if attr == TEXT else element.getAttribute(attr)
    return default if ret is None or isinstance(ret, JsNull) else ret

@typechecked
def setAttr(element: str | Element, attr: str, value: TagAttrValue, onlyIfAbsent: bool = False) -> None:
    if isinstance(element, str):
        element = getElementByID(element)
    if attr == TEXT:
        if not onlyIfAbsent or not element.textContent:
            assert isinstance(value, str), type(value).__name__
            element.textContent = value
    elif not onlyIfAbsent or not element.getAttribute(attr):
        element.setAttribute(attr, value)

@typechecked
def resetInput(element: str | Element) -> None:
    if isinstance(element, str):
        element = getElementByID(element)
    if element.tagName != INPUT:
        return
    if element.type == CHECKBOX:
        element.checked = (getAttr(element, 'checked') == 'true')  # pylint: disable=superfluous-parens
    else:
        element.value = getAttr(element, 'value')
    dispatchEvent(element, CHANGE)

@typechecked
def dispatchEvent(element: str | Element, eventType: str) -> None:
    if isinstance(element, str):
        element = getElementByID(element)
    element.dispatchEvent(newEvent(eventType))

@typechecked
def iterTextNodes(root: Element | JsProxy | None = None) -> Iterator[Node]:
    walker = createTreeWalker(toJsElement(root or page.html), NodeFilter.SHOW_TEXT)
    while node := walker.nextNode():
        assert node.nodeType == TEXT_NODE, node.nodeType
        # noinspection PyTypeChecker
        yield node

@typechecked
class Options(Storage):
    TAG_ATTRIBUTES: ClassVar[Mapping[type[TagAttrValue], Mapping[str, TagAttrValue]]] = {
        int: {
            'inputmode': 'numeric',
            'maxlength': 4,
            'title': "integer",
            'placeholder': "0000",
            'pattern': r'[0-9]+',
        },
        float: {
            'inputmode': 'decimal',
            'maxlength': 4,
            'title': "float",
            'placeholder': "0.00",
            'pattern': r'\.[0-9]{1,2}|[0-9]+\.[0-9]{1,2}|[0-9]+\.?',
        },
    }

    TRANSLATABLE_TAG_ATTRS: ClassVar[Mapping[str, Iterable[str]]] = {
        'HTML'  : ('lang',),
        'META'  : ('content',),
        'A'     : ('title',),
        'BUTTON': ('title',),
         INPUT  : ('title', 'placeholder',),
    }

    LANGUAGES: ClassVar[Mapping[str, str]] = {
        'en_US': "English",
        'ru_RU': "Русский",
    }

    TRANSLATIONS: ClassVar[Mapping[str, GNUTranslations]] = {language: translation('Steganography', './gettext/', (language,)) for language in LANGUAGES}

    @staticmethod
    def translateString(s: str | None) -> str | None:
        if not s or s.isspace():
            return None
        if not (m := match(r'^([\W\d_]*)(.*?)([\W\d_]*)$', s)):
            return None
        (prefix, translatable, suffix) = m.groups()
        if not translatable or (translated := _(translatable)) == translatable:
            return None
        return f"{prefix}{translated}{suffix}"

    @classmethod
    def setLanguage(cls) -> None:
        language = getElementByID('option-language').value
        # noinspection PyGlobalUndefined
        global _  # noqa: PLW0603  # pylint: disable=global-statement
        _ = (tr := cls.TRANSLATIONS[language]).gettext  # type: ignore[assignment]
        if (test := _(GETTEXT_TEST)) != (expected := f'{GETTEXT_TEST}_{language}'):
            log(f"ERROR: gettext.{type(tr).__name__}({language}) is not configured properly, expected {expected}, got {test}")
            return
        log("Language set to", cls.LANGUAGES[language])

        for textNode in chain[Text].from_iterable(iterTextNodes(root)
                for root in page['title, #title, #subtitle, #options, button, .image-title, #footer']):
            if translated := cls.translateString(textNode.nodeValue):
                textNode.nodeValue = translated

        for element in page[', '.join(cls.TRANSLATABLE_TAG_ATTRS)]:
            for attr in cls.TRANSLATABLE_TAG_ATTRS[element.tagName]:
                if translated := cls.translateString(getAttr(element, attr)):
                    setAttr(element, attr, translated)

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # Constructor is only called internally, so we don't know the args and don't care
        super().__init__(*args, **kwargs)

        # These fields define names, types and DEFAULT values for options, actual values are stored in Storage
        self.language = next(iter(self.LANGUAGES))
        self.taskName = ""
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

        elements: list[Element] = []
        for (name, defaultValue) in vars(self).items():
            if isinstance(defaultValue, TagAttrValue):
                elements.append(self.configureElement(name, defaultValue))

        self.styleSheet = newCSSStyleSheet()
        adoptedStyleSheets.push(self.styleSheet)
        self.updateStyle()

        self.setLanguage()

        @when(CLICK, getElementByID('options-reset'))
        @typechecked
        def resetEventHandler(_e: Event) -> None:
            for element in elements:
                resetInput(element)

    def configureElement(self, name: str, defaultValue: TagAttrValue) -> Element:
        valueType = type(defaultValue)
        value = self.get(name, defaultValue)  # Read from database
        assert isinstance(value, valueType), f"Incorrect type for option {name}: {type(value).__name__}, expected {valueType.__name__}"
        elementID = '-'.join(chain(('option',), (word.lower() for word in findall(r'[a-z]+|[A-Z][a-z]*|[A-Z]+', name))))
        element = getElementByID(elementID)
        if name == 'language':  # <SELECT>
            assert element.tagName == SELECT, element.tagName
            assert valueType is str, valueType
            element.innerHTML = ''.join(f'<option value="{lang}">{name}</option>' for (lang, name) in self.LANGUAGES.items())
        else:  # <INPUT>
            assert element.tagName == INPUT, element.tagName
            setAttr(element, 'type', CHECKBOX if valueType is bool else TEXT)
            for (attr, attrValue) in self.TAG_ATTRIBUTES.get(valueType, {}).items():
                setAttr(element, attr, attrValue)  # Set <INPUT> element attributes according to valueType
            if name == 'taskName':  # <INPUT type="text">
                exclude = r'''\/:\\?*'<">&\|'''
                setAttr(element, 'pattern', rf'[^{exclude}\s][^{exclude}]+[^{exclude}\s]')  # Also avoid whitespace at start and end
                title = fr"Do not use {exclude}"  # HTML: "Do not use /:\?*'&lt;&quot;&gt;&amp;|"
                setAttr(element, 'title', title)
                setAttr(element, 'placeholder', title)
                setAttr(element, 'maxlength', 40)
                setAttr(element, 'autocomplete', 'on')
        setAttr(element, 'checked' if valueType is bool else 'value', defaultValue)
        if valueType is bool:
            element.checked = value
        else:
            element.value = value

        @when(CHANGE, element)
        @typechecked
        async def changeEventHandler(_e: Event) -> None:
            newValue: TagAttrValue = element.checked if valueType is bool else element.value.strip()
            if valueType is str:  # taskName or language
                if name == 'taskName' and not newValue:
                    newValue = getDefaultTaskName()  # Provide random task name to save user from extra thinking
            elif valueType is bool:
                pass
            elif newValue:  # int or float as non-empty string
                try:
                    newValue = valueType(newValue)
                except ValueError:
                    newValue = defaultValue
            else:  # empty string
                newValue = defaultValue
            self[name] = newValue  # Save to database
            if valueType is not bool:
                element.value = newValue  # Write processed value back to the input field
            if name in ('maxPreviewWidth', 'maxPreviewHeight'):  # pylint: disable=use-set-for-membership
                self.updateStyle()
            await self.sync()  # Make sure database is really updated
            if name == 'language':
                reload()

        return element

    def updateStyle(self) -> None:
        self.styleSheet.replaceSync(f'''
.image-display {{
    max-width: {self.maxPreviewWidth}px;
    max-height: {self.maxPreviewHeight}px;
}}
        ''')

    async def saveFile(self, name: str, data: bytes | None, fileName: str | None = None) -> None:
        if data:
            assert fileName, repr(fileName)
            self[f'file-{name}'] = bytearray(data)  # Write to database
            self[f'file-{name}-fileName'] = fileName
            await self.sync()
        else:
            await self.delFile(name)

    def loadFile(self, name: str) -> tuple[bytes | None, str | None]:
        data: bytearray | None = self.get(f'file-{name}')  # Read from database
        # noinspection PyRedundantParentheses
        return (bytes(data) if data else None, self.get(f'file-{name}-fileName'))

    async def delFile(self, name: str) -> None:
        with suppress(KeyError):
            del self[f'file-{name}']  # Delete from database
        with suppress(KeyError):
            del self[f'file-{name}-fileName']
        await self.sync()

    def __getattribute__(self, name: str) -> Any:
        defaultValue = super().__getattribute__(name)
        if not isinstance(defaultValue, TagAttrValue):
            return defaultValue  # Not an option field
        ret = self.get(name, None)  # Read from database
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
    TEMPLATE_PREFIX: ClassVar[str] = 'template-'

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
        assert cls.options is None, type(cls.options)  # This method should only be called once
        cls.options = cast(Options, await storage('steganography', storage_class = Options))
        await repaint()
        for stage in Stage:
            cls.ImageBlocks[stage] = ImageBlock(stage)
            await repaint()
        for (stage, block) in cls.ImageBlocks.items():
            block.dependencies = tuple(cls.ImageBlocks[s] for s in cls.DEPENDENCIES[stage])

    @classmethod
    async def process(cls, targetStages: Stage | Iterable[Stage], processFunction: Callable[..., Image | tuple[Image, ...]], sourceStages: Stage | Iterable[Stage]) -> None:
        assert cls.options is not None, type(cls.options)
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
        self.isUpload = (stage.value <= Stage.LOCK.value)  # pylint: disable=superfluous-parens
        self.isProcessed = not self.isUpload and stage.value <= Stage.PROCESSED_KEY.value
        self.isGenerated = not self.isUpload and not self.isProcessed
        self.image: Image | None = None
        self.fileName: str | None = None
        self.dependencies: tuple[ImageBlock, ...] = ()

        # Create DOM element
        block = getElementByID('template').clone(self.getElementID('block'))
        targetID = 'uploaded' if self.isUpload else 'processed' if self.isProcessed else 'generated'
        getElementByID(targetID).append(block)

        # Assign named IDs to all children that have image-* classes
        for element in block.find('*'):
            if element.tagName == 'LABEL' and (for_ := getAttr(element, 'for')) and for_.startswith(self.TEMPLATE_PREFIX):
                setAttr(element, 'for', f'{for_[len(self.TEMPLATE_PREFIX):]}-{self.name}')
            for clas in element.classes:
                if clas.startswith(self.ID_PREFIX):
                    setAttr(element, 'id', f'{clas}-{self.name}')
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
                log("Loading cached image:", fileName)
                image = loadImage(imageBytes, fileName)
            except Exception as ex:  # noqa : BLE001
                self.error(_("loading image"), ex)
                return
            self.setFileName(fileName)
            self.completeOperation(image, imageToBytes(image))
        else:
            self.setFileName()

        if self.isUpload:
            uploadTag = self.getElement('upload')

            @when(CHANGE, uploadTag)
            @typechecked
            async def uploadEventHandler(_e: Event) -> None:
                # noinspection PyShadowingNames
                if fileName := uploadTag.value:
                    await self.upload(getFileNameFromPath(fileName), uploadTag.files.item(0))  # type: ignore[attr-defined]
                else:
                    await self.upload()  # E.g. Esc was pressed at upload dialog

            @when(CLICK, self.getElement('remove'))
            @typechecked
            async def removeEventHandler(_e: Event) -> None:
                uploadTag.value = ''
                await self.remove()

    def getElementID(self, detail: str) -> str:
        return f'{self.ID_PREFIX}{detail}-{self.name}'

    def getElement(self, detail: str) -> Element:
        return getElementByID(self.getElementID(detail))

    def hide(self, name: str) -> None:
        hide(self.getElement(name))

    def show(self, name: str) -> None:
        show(self.getElement(name))

    def getAttr(self, name: str, attr: str, default: str | None = None) -> str | None:
        return getAttr(self.getElement(name), attr, default)

    def setAttr(self, name: str, attr: str, value: str) -> None:
        setAttr(self.getElement(name), attr, value)

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

    async def saveFile(self, byteArray: bytes) -> None:
        assert self.options is not None
        assert self.fileName, repr(self.fileName)
        await self.options.saveFile(self.name, byteArray, self.fileName)

    def loadFile(self) -> tuple[bytes | None, str | None]:
        assert self.options is not None
        return self.options.loadFile(self.name)

    async def delFile(self) -> None:
        assert self.options is not None
        await self.options.delFile(self.name)

    async def remove(self) -> None:
        await self.delFile()
        self.resetBlock()
        await gather(*(block.remove() for block in self.dependencies))

    async def upload(self, fileName: str | None = None, data: Blob | None = None) -> None:
        assert self.isUpload
        if not fileName:  # E.g. Esc was pressed at upload dialog
            return
        self.startOperation(_("Loading image"))
        await repaint()
        try:
            assert data, repr(data)
            # noinspection PyTypeChecker
            imageBytes = await blobToBytes(data)
            image = loadImage(imageBytes)
        except Exception as ex:  # noqa : BLE001
            self.error(_("loading image"), ex)
            return
        self.setFileName(fileName)
        await self.saveFile(imageBytes)
        self.completeOperation(image, imageBytes)
        self.show('remove')
        await self.pipeline()

@typechecked
def exceptionHandler(source: str, exceptionType: type[BaseException | None] | None = None, exception: BaseException | None = None, traceback: TracebackType | None = None) -> None:
    if exceptionType is None:
        exceptionType = type(exception)
    if traceback is None and exception:
        traceback = exception.__traceback__
    # Filter traceback to remove empty lines:
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
    log("Configuring app")
    await ImageBlock.init()
    hide('log')
    show('content')
    log("Running app")

if __name__ == '__main__':
    create_task(main())  # noqa: RUF006

print(f"{PREFIX} Loaded app")
