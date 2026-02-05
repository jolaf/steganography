#
# Note: this app is based on PyScript / Pyodide, it is useless outside the browser
#
# Tested on PyScript 26.2.1 / Pyodide 0.29.3 / Python 3.13.2
#
# ruff: noqa: E402  # pylint: disable=wrong-import-order, wrong-import-position
#
from __future__ import annotations

try:
    from pyscript import document, fetch, when
    from pyscript import storage, Storage
    from pyscript.web import page, Element
    from pyscript.ffi import to_js
except ImportError as ex:
    raise RuntimeError("\n\nThis app can only be run in a browser with PyScript / Pyodide\n") from ex

from sys import version_info
if version_info < (3, 13):  # noqa: UP036
    raise RuntimeError("This app requires Python 3.13+")

PREFIX = "[main]"
print(PREFIX, "Loading app")

from asyncio import create_task, sleep, to_thread
from collections.abc import Buffer, Callable, Coroutine as _Coroutine, Iterable, Iterator, Mapping, Sequence  # `beartype` needs these things in runtime
from contextlib import suppress
from datetime import datetime
from enum import auto, verify, Enum, CONTINUOUS, UNIQUE
from gettext import translation, GNUTranslations
from html import escape
from inspect import iscoroutinefunction, signature
from itertools import chain
from pathlib import Path
from re import findall, match
from sys import stderr  # pylint: disable=ungrouped-imports
from time import time
from traceback import extract_tb
from types import TracebackType  # noqa: TC003
from typing import cast, Any, ClassVar, Final

from js import console, location, Blob, CSSStyleSheet, Event, Node, NodeFilter, Text, Uint8Array, URL
from pyodide.ffi import JsNull, JsProxy  # pylint: disable=import-error, no-name-in-module

# Simplifying addressing to JS functions
newCSSStyleSheet = CSSStyleSheet.new
del CSSStyleSheet  # Delete what we won't need anymore to make sure we really don't need it
newUint8Array = Uint8Array.new
del Uint8Array
createObjectURL = URL.createObjectURL
revokeObjectURL = URL.revokeObjectURL
del URL
adoptedStyleSheets = document.adoptedStyleSheets
createTreeWalker = document.createTreeWalker
del document
reload = location.reload
del location

# We'll redefine these classes to `JsProxy` below, so we have to save all references we actually need
newBlob = Blob.new
newEvent = Event.new
TEXT_NODE = Node.TEXT_NODE

# `beartype` sees these JS types as functions and fails to typecheck calls annotated with them, as of v0.22.9
type Blob = JsProxy  # type: ignore[no-redef]
type Event = JsProxy  # type: ignore[no-redef]
type Node = JsProxy  # type: ignore[no-redef]

type Coroutine[T = object] = _Coroutine[None, None, T]
type CoroutineFunction[T = object] = Callable[..., Coroutine[T]]
type CallableOrCoroutine[T = object] = Callable[..., T | Coroutine[T]]

from workerlib import connectToWorker, diagnostics, elapsedTime, fullName, improveExceptionHandling, systemVersions, typechecked, Worker

from numpy import __version__ as numpyVersion
from PIL import __version__ as pilVersion

try:
    from coolname import generate_slug  # type: ignore[attr-defined]

    @typechecked
    def getDefaultTaskName() -> str:
        return generate_slug(2)
except ImportError:
    console.warn(PREFIX, '`coolname` is not available, using "steganography" as the default task name')

    @typechecked
    def getDefaultTaskName() -> str:
        return "steganography"

from Steganography import getImageMode, getMimeTypeFromImage, imageToBytes, loadImage, OverlayOptions, Image
from Steganography import encrypt, overlay, prepare  # For extracting options only

TagAttrValue = str | int | float | bool

# Tag names
A = 'A'
BUTTON = 'BUTTON'
DIV = 'DIV'
HTML = 'HTML'
INPUT = 'INPUT'
META = 'META'
SELECT = 'SELECT'

# Attribute names
AUTOCOMPLETE = 'autocomplete'
CHECKED = 'checked'
CONTENT = 'content'
INPUTMODE = 'inputmode'
LANG = 'lang'
MAXLENGTH = 'maxlength'
PATTERN = 'pattern'
PLACEHOLDER = 'placeholder'
TITLE = 'title'
VALUE = 'value'

# Special attributes
TEXT_CONTENT = 'textContent'
INNER_HTML = 'innerHTML'

# <INPUT> types
TEXT = 'text'
CHECKBOX = 'checkbox'

# Event names
CLICK = 'click'
CHANGE = 'change'

# Class names
HIDDEN = 'hidden'
TYPE = 'type'

# Misc
GETTEXT_TEST = 'GETTEXT_TEST'

@typechecked
def _(s: str) -> str:  # Will be replaced by gettext
    return s

@typechecked
def toJsElement(element: Element | JsProxy) -> JsProxy:
    return getattr(element, '_dom_element', element)  # type: ignore[arg-type]

@typechecked
def log(*args: Any, showToUser: bool = True) -> None:
    message = ' '.join(str(arg) for arg in args)
    isError = any(word in message.upper() for word in ('ERROR', 'EXCEPTION'))
    print(PREFIX, message, file = stderr if isError else None, flush = True)
    if showToUser:
        logElement = getElementByID('log')
        logElement.append(f"{datetime.now().astimezone().strftime('%H:%M:%S')} {PREFIX} {message}\n")
        if isError:
            logElement.classes.add('error')

@typechecked
async def repaint() -> None:
    await sleep(0.1)  # Yield control to the browser so that repaint could happen

@typechecked
def createObjectURLFromBytes(buffer: Buffer, mimeType: str) -> str:
    blob = newBlob([newUint8Array(buffer),], to_js({TYPE: mimeType}))  # to_js() converts Python dict into JS object
    return createObjectURL(blob)

@typechecked
async def blobToBytes(blob: Blob) -> bytes:
    return (await blob.arrayBuffer()).to_bytes()

@typechecked
def getElementByID(elementID: str) -> Element:  # ToDo: Replace with page[id]
    return page[elementID]

@typechecked
def hide(element: str | Element) -> None:
    if isinstance(element, str):
        element = getElementByID(element)
    element.classes.add(HIDDEN)

@typechecked
def show(element: str | Element) -> None:
    if isinstance(element, str):
        element = getElementByID(element)
    if HIDDEN in element.classes:
        element.classes.remove(HIDDEN)

@typechecked
def getAttr(element: str | Element, attr: str, default: str | None = None) -> str | None:
    if isinstance(element, str):
        element = getElementByID(element)
    assert isinstance(element, Element), type(element)
    ret = getattr(element, attr) if attr in (TEXT_CONTENT, INNER_HTML) else element.getAttribute(attr)
    return default if ret is None or isinstance(ret, JsNull) else ret

@typechecked
def setAttr(element: str | Element, attr: str, value: TagAttrValue, onlyIfAbsent: bool = False) -> None:
    if isinstance(element, str):
        element = getElementByID(element)
    if attr in (TEXT_CONTENT, INNER_HTML):
        if not onlyIfAbsent or not getattr(element, attr):
            assert isinstance(value, str), type(value)
            setattr(element, attr, value)
    elif not onlyIfAbsent or not element.getAttribute(attr):
        element.setAttribute(attr, value)

@typechecked
def resetInput(element: str | Element) -> None:
    if isinstance(element, str):
        element = getElementByID(element)
    if element.tagName != INPUT:
        return
    if element.type == CHECKBOX:
        element.checked = (getAttr(element, CHECKED) == 'true')  # pylint: disable=superfluous-parens
    else:
        element.value = getAttr(element, VALUE)
    #dispatchEvent(element, CHANGE)  # We don't do it to avoid running the pipeline multiple times when resetting all options

@typechecked
def dispatchEvent(element: str | Element, eventType: str) -> None:  # ToDo: remove it when everything is working
    if isinstance(element, str):
        element = getElementByID(element)
    element.dispatchEvent(newEvent(eventType))

@typechecked
def iterTextNodes(root: Element | None = None) -> Iterator[Node]:
    walker = createTreeWalker(toJsElement(root or page.html), NodeFilter.SHOW_TEXT)
    while node := walker.nextNode():
        assert node.nodeType == TEXT_NODE, node.nodeType
        yield node

@typechecked
class Options(Storage):
    UNSET: Final[str] = '-'

    TYPE_DEFAULTS: Final[Mapping[type[TagAttrValue], TagAttrValue]] = {
        int: 0,
        float: 1.0,
    }

    TAG_ATTRIBUTES: Final[Mapping[type[TagAttrValue], Mapping[str, TagAttrValue]]] = {  # pylint: disable=consider-using-namedtuple-or-dataclass
        int: {
            INPUTMODE: 'numeric',
            MAXLENGTH: 4,
            TITLE: "integer",
            PLACEHOLDER: "0000",
            PATTERN: r'\s*(-|[0-9]+)?\s*',
        },
        float: {
            INPUTMODE: 'decimal',
            MAXLENGTH: 4,
            TITLE: "float",
            PLACEHOLDER: "0.00",
            PATTERN: r'\s*(-|\.[0-9]{1,2}|[0-9]+\.[0-9]{1,2}|[0-9]+\.?)?\s*',
        },
    }

    TEXT_NODE_ROOTS: Final[str] = 'title, #title, #subtitle, #options, button, .image-title, #footer'

    TRANSLATABLE_TAG_ATTRS: Final[Mapping[str, Iterable[str]]] = {
        HTML   : (LANG,),
        META   : (CONTENT,),
        A      : (TITLE,),
        BUTTON : (TITLE,),
        DIV    : (TITLE,),
        INPUT  : (TITLE, PLACEHOLDER),
    }

    LANGUAGES: Final[Mapping[str, str]] = {
        'en_US': "English",
        'ru_RU': "Русский",
    }

    TRANSLATIONS: Final[Mapping[str, GNUTranslations]] = {language: translation('Steganography', './gettext/', (language,)) for language in LANGUAGES}

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
            raise ValueError(f"gettext.{type(tr).__name__}({language}) is not configured properly, expected {expected}, got {test}")
        log("Language set to", cls.LANGUAGES[language])

        for textNode in chain[Text].from_iterable(iterTextNodes(root)
                for root in page.find(cls.TEXT_NODE_ROOTS)):
            if translated := cls.translateString(textNode.nodeValue):
                textNode.nodeValue = translated

        for element in page.find(', '.join(cls.TRANSLATABLE_TAG_ATTRS)):
            for attr in cls.TRANSLATABLE_TAG_ATTRS[element.tagName]:
                if translated := cls.translateString(getAttr(element, attr)):
                    setAttr(element, attr, translated)

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # Constructor is only called internally, so we don't know the args and don't care
        super().__init__(*args, **kwargs)

        # These fields define names, types and DEFAULT values for options; actual values are stored in Storage
        self.language = next(iter(self.LANGUAGES))
        self.taskName = ""
        self.maxPreviewWidth = 0
        self.maxPreviewHeight = 100
        self.resizeFactor = 1.0
        self.resizeWidth = 0
        self.resizeHeight = 0
        self.lockFactor = 1.0
        self.lockWidth = 0
        self.lockHeight = 0
        self.randomRotate = False
        self.randomFlip = False
        self.dither = False
        self.smooth = False

        elements: dict[str, Element] = {}
        for (name, defaultValue) in vars(self).items():
            if isinstance(defaultValue, TagAttrValue):
                elements[name] = self.configureElement(name, defaultValue)

        self.styleSheet = newCSSStyleSheet()
        adoptedStyleSheets.push(self.styleSheet)
        self.updateStyle()
        self.setLanguage()

        @when(CLICK, getElementByID('options-reset'))
        @typechecked
        async def resetEventHandler(_e: Event) -> None:
            for (name, element) in elements.items():
                if name == 'taskName':
                    self[name] = element.value = getDefaultTaskName()  # Save to the database
                else:
                    resetInput(element)  # Doesn't reset the language because that is <SELECT>, not <INPUT>
            await ImageBlock.resetUploads()

    def configureElement(self, name: str, defaultValue: TagAttrValue) -> Element:
        valueType = type(defaultValue)
        typeDefaultValue = self.TYPE_DEFAULTS.get(valueType)
        value = self.get(name, defaultValue)  # Read from the database
        assert isinstance(value, valueType), f"Incorrect type for option {name}: {type(value)}, expected {valueType}"
        elementID = '-'.join(chain(('option',), (word.lower() for word in findall(r'[a-z]+|[A-Z][a-z]*|[A-Z]+', name))))
        element = getElementByID(elementID)
        if name == 'language':  # <SELECT>
            assert element.tagName == SELECT, element.tagName
            assert valueType is str, valueType
            for (lang, langName) in self.LANGUAGES.items():
                element.options.add(value = lang, html = langName)  # type: ignore[union-attr]
        else:  # <INPUT>
            assert element.tagName == INPUT, element.tagName
            setAttr(element, TYPE, CHECKBOX if valueType is bool else TEXT)
            for (attr, attrValue) in self.TAG_ATTRIBUTES.get(valueType, {}).items():
                setAttr(element, attr, attrValue)  # Set <INPUT> element attributes according to valueType
            if name == 'taskName':  # <INPUT type="text">
                exclude = r'''\/:\\?*'<">&\|'''
                setAttr(element, PATTERN, rf'[^{exclude}]+')
                for (f, t) in {r'\/': '/', r'\|': '|', r'\\': '\\'}.items():
                    exclude = exclude.replace(f, t)
                title = fr"Do not use {exclude}"  # HTML: "Do not use /:\?*'&lt;&quot;&gt;&amp;|"
                setAttr(element, TITLE, title)
                setAttr(element, PLACEHOLDER, title)
                setAttr(element, MAXLENGTH, 40)
                setAttr(element, AUTOCOMPLETE, 'on')
                if not value:
                    value = getDefaultTaskName()
                    self[name] = value
        setAttr(element, CHECKED if valueType is bool else VALUE, defaultValue)
        if valueType is bool:
            element.checked = value
        elif valueType in (int, float):
            element.value = self.UNSET if value == typeDefaultValue else value
        else:  # str
            element.value = value

        @when(CHANGE, element)
        @typechecked
        async def changeEventHandler(_e: Event) -> None:
            newValue: TagAttrValue = element.checked if valueType is bool else element.value.strip()
            if valueType is str:  # taskName or language
                if name == 'taskName':
                    if not newValue:
                        newValue = getDefaultTaskName()  # Provide random task name to save user from extra thinking
                    elif not element.checkValidity():
                        newValue = self.get(name, getDefaultTaskName())
            elif valueType is bool:  # checkbox
                pass
            elif newValue and newValue != self.UNSET:  # int or float as a non-empty string
                try:
                    if (newValue := valueType(newValue)) <= 0:  # type: ignore[operator]
                        raise ValueError(f"Must be positive, got {newValue}")  # noqa: TRY301
                except ValueError:
                    newValue = self.get(name, defaultValue)  # Read from the database
            else:  # int or float from empty string or UNSET
                assert typeDefaultValue is not None
                newValue = typeDefaultValue if newValue == self.UNSET else defaultValue
            self[name] = newValue  # Save to the database
            if valueType in (int, float):
                element.value = self.UNSET if newValue == typeDefaultValue else newValue
            elif valueType is str:
                element.value = newValue  # Write the processed value back to the input field
            if name in ('maxPreviewWidth', 'maxPreviewHeight', 'randomRotate', 'randomFlip'):  # pylint: disable=use-set-for-membership
                self.updateStyle()
            await self.sync()  # Make sure the database is really updated
            if name == 'language':
                reload()

        return element

    def updateStyle(self) -> None:
        self.styleSheet.replaceSync(f'''
.image-display {{
    max-width: {f'{self.maxPreviewWidth}px' if self.maxPreviewWidth else 'none'};
    max-height: {f'{self.maxPreviewHeight}px' if self.maxPreviewHeight else 'none'};
}}
''' + ('''
#image-display-generated-key {
    border-color: red;
}
''' if self.randomRotate or self.randomFlip else ''))

    async def saveFile(self, name: str, data: Buffer | None, fileName: str | None = None) -> None:
        if data:
            assert fileName, repr(fileName)
            self[f'file-{name}'] = bytearray(data)  # Write to the database
            self[f'file-{name}-fileName'] = fileName
            await self.sync()
        else:
            await self.delFile(name)

    def loadFile(self, name: str) -> tuple[bytearray | None, str | None]:
        data: bytearray | None = self.get(f'file-{name}')  # Read from the database
        return (data, self.get(f'file-{name}-fileName'))

    async def delFile(self, name: str) -> None:
        with suppress(KeyError):
            del self[f'file-{name}']  # Delete from database
        with suppress(KeyError):
            del self[f'file-{name}-fileName']
        await self.sync()

    def fillOptions(self, options: Iterable[str] | Mapping[str, Any] | None) -> Mapping[str, Any]:
        if not options:
            return {}
        if isinstance(options, Mapping):
            ret = dict(options)
            for (option, value) in options.items():
                if value is None and (value := self.get(option)) and value != super().__getattribute__(option):
                    ret[option] = value  # noqa: PERF403
            return ret
        ret = {}
        for option in options:
            if (value := self.get(option)) and value != super().__getattribute__(option):  # ToDo: Maybe instead of this hack we should have proper Option class, encapsulating type, default value, actual value and Element reference
                ret[option] = value  # Only fill options with non-default values
        return ret

    def __getattribute__(self, name: str) -> Any:
        defaultValue = super().__getattribute__(name)
        if not isinstance(defaultValue, TagAttrValue):
            return defaultValue  # Not an option field
        ret = self.get(name, None)  # Read from the database
        return defaultValue if ret is None else ret

    def __setattr__(self, name: str, value: Any) -> None:
        with suppress(AttributeError):
            defaultValue = super().__getattribute__(name)
            if isinstance(defaultValue, TagAttrValue):
                if isinstance(defaultValue, float):
                    assert isinstance(value, float | int), f"Incorrect type for option {name}: {type(value)}, expected int or float"
                else:
                    assert isinstance(value, type(defaultValue)), f"Incorrect type for option {name}: {type(value)}, expected {type(defaultValue)}"
                self[name] = value
                return
        super().__setattr__(name, value)

@typechecked
@verify(UNIQUE, CONTINUOUS)
class Stage(Enum):
    SOURCE = auto()
    LOCK = auto()
    KEY = auto()
    PROCESSED_SOURCE = auto()
    PROCESSED_LOCK = auto()
    PROCESSED_KEY = auto()
    GENERATED_LOCK = auto()
    GENERATED_KEY = auto()
    KEY_OVER_LOCK_TEST = auto()

@typechecked
class ImageBlock:
    ID_PREFIX: Final[str] = 'image-'
    TEMPLATE_PREFIX: Final[str] = 'template-'
    REMOVED: Final[bytes] = b'__REMOVED__'

    BLOCK_NAMES: Final[Mapping[Stage, str]] = dict(zip(Stage, (
        _("Source image"),
        _("Lock mask"),
        _("Key mask"),
        _("Processed source"),
        _("Processed lock mask"),
        _("Processed key mask"),
        _("Generated lock"),
        _("Generated key"),
        _("Overlay test"),
    ), strict = True))

    SOURCES: Final[Mapping[Stage, Stage]] = {
        Stage.PROCESSED_SOURCE: Stage.SOURCE,
        Stage.PROCESSED_LOCK: Stage.LOCK,
        Stage.PROCESSED_KEY: Stage.KEY,
    }

    PRELOADED_FILES: Final[Mapping[Stage, str]] = {
        Stage.LOCK: './images/lock.png',  # These are relative file paths loaded via fetch(), not via `main.toml`
        Stage.KEY: './images/key.png',
    }

    PROCESS_OPTIONS: ClassVar[Mapping[Callable[..., Any], Sequence[str]]] = {}

    imageBlocks: Final[dict[Stage, ImageBlock]] = {}

    options: ClassVar[Options | None] = None
    worker: ClassVar[Worker | None] = None

    @classmethod
    def extractOptions(cls, func: Callable[..., Any]) -> Sequence[str]:
        return tuple(name for (name, param) in signature(func).parameters.items() if param.kind == param.KEYWORD_ONLY)

    @classmethod
    async def init(cls) -> None:
        assert cls.options is None, type(cls.options)  # This method should only be called once
        cls.options = cast(Options, await storage('steganography', storage_class = Options))
        cls.PROCESS_OPTIONS = {func: cls.extractOptions(func) for func in (encrypt, overlay, prepare)}
        for stage in Stage:
            cls.imageBlocks[stage] = ImageBlock(stage)
        getElementByID('template').remove()
        for (stage, block) in cls.imageBlocks.items():
            block.source = cls.imageBlocks.get(cls.SOURCES.get(stage))  # type: ignore[arg-type]
        cls.worker = await connectToWorker()

    @classmethod
    async def loadImages(cls) -> None:
        for block in cls.imageBlocks.values():
            if block.isUpload:
                await block.loadImageFromCache()

    @classmethod
    async def resetUploads(cls) -> None:
        for stage in cls.PRELOADED_FILES:
            await cls.imageBlocks[stage].resetUpload()
        await cls.pipeline()

    @classmethod
    async def process(cls,
                      targetStages: Stage | Iterable[Stage],
                      processFunction: CallableOrCoroutine[Image | tuple[Image, Image, OverlayOptions | None]],
                      sourceStages: Stage | Iterable[Stage],
                      *,
                      optionalSourceStages: Stage | Iterable[Stage] | None = None,
                      affectedStages: Stage | Iterable[Stage] | None = None,
                      options: Iterable[str] | Mapping[str, Any] = ()) -> OverlayOptions | None:
        sources = tuple(cls.imageBlocks[stage] for stage in ((sourceStages,) if isinstance(sourceStages, Stage) else sourceStages))
        targets = tuple(cls.imageBlocks[stage] for stage in ((targetStages,) if isinstance(targetStages, Stage) else targetStages))
        optionalSources = tuple(cls.imageBlocks[stage] for stage in ((optionalSourceStages,) if isinstance(optionalSourceStages, Stage) else optionalSourceStages or ()))
        if not any(source.dirty for source in chain(sources, optionalSources)) and \
               all(target.image for target in targets):
            return None
        affected = tuple(cls.imageBlocks[stage] for stage in ((affectedStages,) if isinstance(affectedStages, Stage) else affectedStages or ()))
        for t in chain(targets, affected):
            t.clean()
            await repaint()
        if not all(source.image for source in sources):
            return None
        for t in targets:
            t.startOperation(_("Processing image"))
        await repaint()
        try:
            sourceImages = tuple(source.image for source in chain(sources, optionalSources))
            assert cls.options
            options = cls.options.fillOptions(options)
            log(f"Processing {', '.join(source.stage.name for source in chain(sources, optionalSources))} => {', '.join(target.stage.name for target in targets)}")
            startTime = time()
            if iscoroutinefunction(processFunction) or isinstance(processFunction, JsProxy):  # pylint: disable=consider-ternary-expression
                ret = await processFunction(*sourceImages, **options)
            else:
                ret = await to_thread(processFunction, *sourceImages, **options)
            log(f"Completed processing {elapsedTime(startTime)}")
            if ret is None:  # No changes were made, use original image
                assert len(sourceImages) == 1 and sourceImages[0], sourceImages  # noqa: PT018
                ret = sourceImages
            elif isinstance(ret, Image):
                ret = (ret,)
            assert ret, repr(ret)
            assert isinstance(ret[0], Image), f"{type(ret)} {type(ret[0])}"
            retImages = tuple(r for r in ret if isinstance(r, Image))
            assert len(retImages) == len(targets), f"{fullName(processFunction)}() returned {len(ret)} images, expected {len(targets)}"
            for (target, image) in zip(targets, retImages, strict = True):
                target.completeOperation(image, await imageToBytes(image))
            await repaint()
            ret = ret[len(targets)] if len(ret) > len(targets) else None
        except Exception as ex:  # noqa : BLE001
            for target in targets:
                target.error(_("processing image"), ex)
            await repaint()
            ret = None
        assert isinstance(ret, Mapping | None), type(ret)
        return cast(OverlayOptions | None, ret)

    @classmethod
    async def pipeline(cls) -> None:  # Called from the upload event handler to generate secondary images
        startTime = time()
        log("Started pipeline")
        assert cls.worker
        ret = await cls.process(Stage.PROCESSED_SOURCE,
                                cls.worker.prepare, Stage.SOURCE,  # type: ignore[attr-defined]
                                options = cls.PROCESS_OPTIONS[prepare])
        ret = await cls.process(Stage.PROCESSED_LOCK,
                                cls.worker.prepare, Stage.LOCK,  # type: ignore[attr-defined]
                                options = {'dither': None})
        ret = await cls.process(Stage.PROCESSED_KEY,
                                cls.worker.prepare, Stage.KEY,  # type: ignore[attr-defined]
                                options = {'dither': None})
        ret = await cls.process((Stage.GENERATED_LOCK, Stage.GENERATED_KEY),
                                cls.worker.encrypt, Stage.PROCESSED_SOURCE,  # type: ignore[attr-defined]
                                optionalSourceStages = (Stage.PROCESSED_LOCK, Stage.PROCESSED_KEY),
                                options = cls.PROCESS_OPTIONS[encrypt])
        options: dict[str, Any] | Sequence[str]
        if ret:
            assert isinstance(ret, Mapping), type(ret)
            options = dict.fromkeys(cls.PROCESS_OPTIONS[overlay])
            options.update(ret)
        else:
            options = cls.PROCESS_OPTIONS[overlay]
        ret = await cls.process(Stage.KEY_OVER_LOCK_TEST,
                                cls.worker.overlay, (Stage.GENERATED_LOCK, Stage.GENERATED_KEY),  # type: ignore[attr-defined]
                                options = options)
        for imageBlock in cls.imageBlocks.values():
            imageBlock.dirty = False

        log(f"Completed pipeline {elapsedTime(startTime)}")

    def __init__(self, stage: Stage) -> None:
        self.stage = stage
        self.name = stage.name.lower().replace('_', '-')
        self.isUpload = (stage.value <= Stage.KEY.value)  # pylint: disable=superfluous-parens
        self.isProcessed = not self.isUpload and stage.value <= Stage.PROCESSED_KEY.value
        self.isGenerated = not self.isUpload and not self.isProcessed
        self.image: Image | None = None
        self.fileName: str | None = None
        self.source: ImageBlock | None = None
        self.dirty = False

        # Create DOM element
        block = getElementByID('template').clone(self.getElementID('block'))
        targetID = 'uploaded' if self.isUpload else 'processed' if self.isProcessed else 'generated'
        getElementByID(targetID).append(block)

        # Assign named IDs to all children that have image-* classes
        for element in block.find('*'):
            if element.tagName == 'LABEL' and (for_ := getAttr(element, 'for')) and for_.startswith(self.TEMPLATE_PREFIX):  # ToDo: Add constants for all repeated strings
                setAttr(element, 'for', f'{for_[len(self.TEMPLATE_PREFIX):]}-{self.name}')
            for clas in element.classes:
                if clas.startswith(self.ID_PREFIX):
                    setAttr(element, 'id', f'{clas}-{self.name}')
                    break

        # Further configure children
        self.setAttr(TITLE, TEXT_CONTENT, _(self.BLOCK_NAMES[self.stage]))
        downloadLink = self.getElement('download-link')

        @when(CLICK, self.getElement('download'))
        @typechecked
        async def downloadEventHandler(_e: Event) -> None:
            downloadLink.click()

        if not self.isUpload:
            self.hide('upload-block')
            self.setFileName()
            return

        # Further configuration for upload blocks only

        self.hide('description')
        self.show('block')
        uploadTag = self.getElement('upload')

        @when(CHANGE, uploadTag)
        @typechecked
        async def uploadEventHandler(_e: Event) -> None:
            if (files := uploadTag.files).length:
                file = files.item(0)  # JS FileList API
                await self.uploadFile(file.name, file)

        @when(CLICK, self.getElement('upload-button'))
        @typechecked
        async def uploadClickEventHandler(_e: Event) -> None:
            uploadTag.click()

        @when(CLICK, self.getElement('remove'))
        @typechecked
        async def removeEventHandler(_e: Event) -> None:
            uploadTag.value = ''
            await self.removeImage()
            await self.pipeline()

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

    def setAttr(self, name: str, attr: str, value: str, onlyIfAbsent: bool = False) -> None:
        setAttr(self.getElement(name), attr, value, onlyIfAbsent)

    def setFileName(self, fileName: str | None = None) -> None:
        assert self.options is not None
        if not fileName:
            fileName = f'{self.options.taskName}-{self.name}.png'
        self.fileName = fileName
        self.setAttr('name', TEXT_CONTENT, fileName)
        self.setAttr('download-link', 'download', fileName)

    def setURL(self, url: str = '') -> None:
        if src := self.getAttr('display', 'src'):
            revokeObjectURL(src)
        self.setAttr('display', 'src', url)
        self.setAttr('display-link', 'href', url)
        self.setAttr('download-link', 'href', url)

    def setDescription(self, message: str, isError: bool = False) -> None:
        self.setAttr('description', INNER_HTML if isError else TEXT_CONTENT, message)
        self.show('description')

    def error(self, message: str, exception: BaseException | None = None) -> None:
        self.setDescription(f"{_("ERROR")} {message}{f": <pre>{escape(str(exception))}</pre>" if exception else ''}", isError = True)
        self.hide('remove')  # ToDo: Should we really print stack here? Or just say "Press F12 to check console"? And what about phones?
        if exception:
            exceptionHandler("Exception at image processing", exception = exception, showToUser = False)

    def clean(self, description: str | None = None) -> None:
        self.setURL()
        self.image = None
        self.dirty = True
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
        self.clean(message + "…")

    def completeOperation(self, image: Image, buffer: Buffer) -> None:
        if self.isUpload:
            self.show('remove')
        if self.source and image == self.source.image:
            self.image = self.source.image  # Make them identical to avoid storing duplicate data
            self.setURL()
            self.hide('block')
        else:
            self.image = image
            self.setURL(createObjectURLFromBytes(buffer, getMimeTypeFromImage(image)))
            self.setDescription(f"{len(buffer)} {_("bytes")} {image.format} {getImageMode(image)} {image.width}x{image.height}")  # type: ignore[arg-type]
            self.show('display-block')
            self.show('block')
        self.dirty = True

    async def resetUpload(self) -> None:
        assert self.isUpload
        assert self.stage in self.PRELOADED_FILES
        assert self.options is not None
        await self.options.delFile(self.name)
        await self.loadImageFromCache()

    async def saveImageToCache(self, buffer: Buffer) -> None:
        assert self.isUpload
        assert self.options is not None
        assert self.fileName, repr(self.fileName)
        await self.options.saveFile(self.name, buffer, self.fileName)

    async def loadImageFromCache(self) -> None:
        assert self.isUpload
        assert self.options is not None
        (imageBytes, fileName) = self.options.loadFile(self.name)
        if imageBytes:
            if imageBytes == self.REMOVED:
                (imageBytes, fileName) = (None, None)
        elif filePath := self.PRELOADED_FILES.get(self.stage):
            log("Fetching the preloaded file:", filePath)
            await repaint()
            (imageBytes, fileName) = (await fetch(filePath).arrayBuffer(), Path(filePath).name)  # type: ignore[attr-defined]
            if imageBytes:
                self.fileName = fileName
                await self.saveImageToCache(imageBytes)
        else:
            (imageBytes, fileName) = (None, None)
        if imageBytes:
            try:
                log("Loading image:", fileName)
                await repaint()
                image = await loadImage(imageBytes, fileName)
            except Exception as ex:  # noqa : BLE001
                self.error(_("loading image"), ex)
                await repaint()
                return
            self.setFileName(fileName)
            self.completeOperation(image, imageBytes)
        else:
            self.setFileName()

    async def removeImage(self) -> None:
        if self.isUpload:
            assert self.options is not None
            if filePath := self.PRELOADED_FILES.get(self.stage):
                await self.options.saveFile(self.name, self.REMOVED, Path(filePath).name)
            else:
                await self.options.delFile(self.name)
        self.clean()

    async def uploadFile(self, fileName: str | None = None, data: Blob | None = None) -> None:
        assert self.isUpload
        if not fileName:  # E.g., Esc was pressed at upload dialog
            return
        self.startOperation(_("Loading image"))
        await repaint()
        try:
            log("Uploading image:", fileName)
            await repaint()
            assert data, repr(data)
            imageBytes = await blobToBytes(data)
            image = await loadImage(imageBytes)
        except Exception as ex:  # noqa : BLE001
            self.error(_("loading image"), ex)
            await repaint()
            return
        self.setFileName(fileName)
        await self.saveImageToCache(imageBytes)
        self.completeOperation(image, imageBytes)
        self.show('remove')
        await self.pipeline()

@typechecked
def exceptionHandler(problem: str,
                     exceptionType: type[BaseException | None] | None = None,
                     exception: BaseException | None = None,
                     traceback: TracebackType | None = None,
                     *,
                     showToUser: bool = True,
                     ) -> None:
    if exceptionType is None:
        exceptionType = type(exception)
    if traceback is None and exception:
        traceback = exception.__traceback__
    # Filter the traceback to remove empty lines:
    tracebackStr = '\n====== Traceback:\n' + '\n'.join(line for line in '\n'.join(extract_tb(traceback).format()).splitlines() if line.strip()) if traceback else ''
    log(f"""
{PREFIX} ERROR: {problem}, type {fullName(exceptionType)}:
{exception}{tracebackStr}

Please make a screenshot and report it to @jolaf at Telegram or VK or to vmzakhar@gmail.com. Thank you!
""", showToUser = showToUser)

@typechecked
async def main() -> None:
    log("Starting app")
    improveExceptionHandling(log)

    for info in diagnostics:
        log(info)

    for (product, version) in systemVersions.items():
        if element := page[product.lower() + '-version']:
            element.textContent = version

    log("Pillow", pilVersion)
    log("NumPy", numpyVersion)

    await repaint()
    await ImageBlock.init()
    hide('log')
    show(CONTENT)
    log("Loading cached images")
    await ImageBlock.loadImages()
    await ImageBlock.pipeline()
    await repaint()
    log("Started app")

if __name__ == '__main__':
    create_task(main())  # `create_task()` is only needed to silence static checkers that don't like `await` in global module code  # noqa: RUF006

print(PREFIX, "Loaded app")
