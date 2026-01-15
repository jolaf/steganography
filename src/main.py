# ruff: noqa: E402  # pylint: disable=wrong-import-order, wrong-import-position
# Note: this module is PyScript-only, it won't work outside of browser
from __future__ import annotations

PREFIX = "[python]"
print(f"{PREFIX} Loading app")

from asyncio import create_task, gather, get_running_loop, sleep, to_thread, AbstractEventLoop
from collections.abc import Awaitable, Buffer, Callable, Iterable, Iterator, Mapping, Sequence  # noqa: TC003  # beartype needs these things in runtime
from contextlib import suppress
from datetime import datetime
from enum import Enum
from gettext import translation, GNUTranslations
from inspect import iscoroutinefunction
from itertools import chain
from pathlib import Path
from re import findall, match
import sys
from sys import version as pythonVersion
from traceback import extract_tb
from types import TracebackType  # noqa: TC003
from typing import cast, Any, ClassVar

try:
    from beartype import beartype as typechecked, __version__ as beartypeVersion
    from beartype.roar import BeartypeException
except ImportError:
    print(f"{PREFIX} WARNING: beartype is not available, running fast with typing unchecked")
    beartypeVersion = None  # type: ignore[assignment]

    def typechecked(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[no-redef]
        return func

try:
    from coolname import generate_slug  # type: ignore[attr-defined]

    @typechecked
    def getDefaultTaskName() -> str:
        return cast(str, generate_slug(2))
except ImportError:
    print(f'{PREFIX} WARNING: coolname is not available, using "steganography" as default task name')

    @typechecked
    def getDefaultTaskName() -> str:
        return "steganography"

from pyscript import document, fetch, when
from pyscript import storage, Storage
from pyscript.web import page, Element  # pylint: disable=import-error, no-name-in-module
from pyscript.ffi import to_js  # pylint: disable=import-error, no-name-in-module

try:  # Try to identify PyScript version
    from pyscript import version as pyscriptVersion  # type: ignore[attr-defined]
except ImportError:
    try:
        from pyscript import __version__ as pyscriptVersion  # type: ignore[attr-defined]
    except ImportError:
        try:
            coreURL = next(element.src for element in page['script'] if element.src.endswith('core.js'))
            pyscriptVersion = next(word for word in coreURL.split('/') if findall(r'\d', word))
        except Exception:  # noqa: BLE001
            pyscriptVersion = "UNKNOWN"

from js import location, Blob, CSSStyleSheet, Event, Node, NodeFilter, Text, Uint8Array, URL
from pyodide.ffi import JsNull, JsProxy  # pylint: disable=import-error, no-name-in-module

# We'll redefine these classes to JsProxy below, so we have to save all references we actually need
newCSSStyleSheet = CSSStyleSheet.new
newBlob = Blob.new
newEvent = Event.new
newUint8Array = Uint8Array.new

# Simplifying addressing to JS functions
createObjectURL = URL.createObjectURL
revokeObjectURL = URL.revokeObjectURL
del URL  # We won't need it anymore
adoptedStyleSheets = document.adoptedStyleSheets
createTreeWalker = document.createTreeWalker
del document  # We won't need it anymore
reload = location.reload
del location  # We won't need it anymore

TEXT_NODE = Node.TEXT_NODE

# JS types that don't work as runtime type annotations
type Blob = JsProxy  # type: ignore[no-redef]  # ToDo: File a bug about this
type Event = JsProxy  # type: ignore[no-redef]
type Node = JsProxy  # type: ignore[no-redef]

try:
    from pyodide_js import version as pyodideVersion  # type: ignore[import-not-found]
except ImportError:
    pyodideVersion = "UNKNOWN"

from numpy import __version__ as numpyVersion
from PIL import __version__ as pilVersion

from workerlib import connectToWorker, Worker

from Steganography import getImageMode, getMimeTypeFromImage, imageToBytes, loadImage, Image

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
    print(f"{PREFIX} {message}")
    if showToUser:
        logElement = getElementByID('log')
        toJsElement(logElement).append(f"{datetime.now().astimezone().strftime('%H:%M:%S')} {PREFIX} {message}\n")  # https://github.com/pyscript/pyscript/issues/2418
        test = message.upper()
        if any(word in test for word in ('ERROR', 'EXCEPTION')):
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
def getElementByID(elementID: str) -> Element:
    return page['#' + elementID][0]

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
    #dispatchEvent(element, CHANGE)  # We don't do it to avoid running pipeline multiple times when resetting all options

@typechecked
def dispatchEvent(element: str | Element, eventType: str) -> None:
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
    UNSET: ClassVar[str] = '-'

    TYPE_DEFAULTS: ClassVar[Mapping[type[TagAttrValue], TagAttrValue]] = {
        int: 0,
        float: 1.0,
    }

    TAG_ATTRIBUTES: ClassVar[Mapping[type[TagAttrValue], Mapping[str, TagAttrValue]]] = {
        int: {
            'inputmode': 'numeric',
            'maxlength': 4,
            'title': "integer",
            'placeholder': "0000",
            'pattern': r'\s*(-|[0-9]+)?\s*',
        },
        float: {
            'inputmode': 'decimal',
            'maxlength': 4,
            'title': "float",
            'placeholder': "0.00",
            'pattern': r'\s*(-|\.[0-9]{1,2}|[0-9]+\.[0-9]{1,2}|[0-9]+\.?)?\s*',
        },
    }

    TRANSLATABLE_TAG_ATTRS: ClassVar[Mapping[str, Iterable[str]]] = {
        'HTML'  : ('lang',),
        'META'  : ('content',),
        'A'     : ('title',),
        'BUTTON': ('title',),
        'DIV'   : ('title',),
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
            raise ValueError(f"gettext.{type(tr).__name__}({language}) is not configured properly, expected {expected}, got {test}")
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
        self.maxPreviewWidth = 0
        self.maxPreviewHeight = 100
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
                    self[name] = element.value = getDefaultTaskName()  # Save to database
                else:
                    resetInput(element)  # Doesn't reset language because that is <SELECT>, not <INPUT>
            await ImageBlock.resetUploads()

    def configureElement(self, name: str, defaultValue: TagAttrValue) -> Element:
        valueType = type(defaultValue)
        typeDefaultValue = self.TYPE_DEFAULTS.get(valueType)
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
            setAttr(element, TYPE, CHECKBOX if valueType is bool else TEXT)
            for (attr, attrValue) in self.TAG_ATTRIBUTES.get(valueType, {}).items():
                setAttr(element, attr, attrValue)  # Set <INPUT> element attributes according to valueType
            if name == 'taskName':  # <INPUT type="text">
                exclude = r'''\/:\\?*'<">&\|'''
                setAttr(element, 'pattern', rf'[^{exclude}]+')
                for (f, t) in {r'\/': '/', r'\|': '|', r'\\': '\\'}.items():
                    exclude = exclude.replace(f, t)
                title = fr"Do not use {exclude}"  # HTML: "Do not use /:\?*'&lt;&quot;&gt;&amp;|"
                setAttr(element, 'title', title)
                setAttr(element, 'placeholder', title)
                setAttr(element, 'maxlength', 40)
                setAttr(element, 'autocomplete', 'on')
                if not value:
                    value = getDefaultTaskName()
                    self[name] = value
        setAttr(element, 'checked' if valueType is bool else 'value', defaultValue)
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
            elif newValue and newValue != self.UNSET:  # int or float as non-empty string
                try:
                    if (newValue := valueType(newValue)) <= 0:  # type: ignore[operator]
                        raise ValueError(f"Must be positive, got {newValue}")  # noqa: TRY301
                except ValueError:
                    newValue = self.get(name, defaultValue)  # Read from database
            else:  # int or float from empty string or UNSET
                assert typeDefaultValue is not None
                newValue = typeDefaultValue if newValue == self.UNSET else defaultValue
            self[name] = newValue  # Save to database
            if valueType in (int, float):
                element.value = self.UNSET if newValue == typeDefaultValue else newValue
            elif valueType is str:
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
    max-width: {f'{self.maxPreviewWidth}px' if self.maxPreviewWidth else 'none'};
    max-height: {f'{self.maxPreviewHeight}px' if self.maxPreviewHeight else 'none'};
}}
        ''')

    async def saveFile(self, name: str, data: Buffer | None, fileName: str | None = None) -> None:
        if data:
            assert fileName, repr(fileName)
            self[f'file-{name}'] = bytearray(data)  # Write to database
            self[f'file-{name}-fileName'] = fileName
            await self.sync()
        else:
            await self.delFile(name)

    def loadFile(self, name: str) -> tuple[bytearray | None, str | None]:
        data: bytearray | None = self.get(f'file-{name}')  # Read from database
        return (data, self.get(f'file-{name}-fileName'))

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
        with suppress(AttributeError):
            defaultValue = super().__getattribute__(name)
            if isinstance(defaultValue, TagAttrValue):
                if isinstance(defaultValue, float):
                    assert isinstance(value, int | float), f"Incorrect type for option {name}: {type(value).__name__}, expected int or float"
                else:
                    assert isinstance(value, type(defaultValue)), f"Incorrect type for option {name}: {type(value).__name__}, expected {type(defaultValue).__name__}"
                self[name] = value
                return
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
    KEY_OVER_LOCK_TEST = 9

@typechecked
class ImageBlock:
    ID_PREFIX: ClassVar[str] = 'image-'
    TEMPLATE_PREFIX: ClassVar[str] = 'template-'
    REMOVED: ClassVar[bytes] = b'__REMOVED__'

    BLOCK_NAMES: ClassVar[Mapping[Stage, str]] = dict(zip(Stage, (
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

    SOURCES: ClassVar[Mapping[Stage, Stage]] = {
        Stage.PROCESSED_SOURCE: Stage.SOURCE,
        Stage.PROCESSED_LOCK: Stage.LOCK,
        Stage.PROCESSED_KEY: Stage.KEY,
    }

    PRELOADED_FILES: ClassVar[Mapping[Stage, str]] = {
        Stage.LOCK: './lock.png',
        Stage.KEY: './key.png',
    }

    ImageBlocks: ClassVar[dict[Stage, ImageBlock]] = {}

    options: ClassVar[Options | None] = None
    worker: ClassVar[Worker | None] = None

    @classmethod
    async def init(cls) -> None:
        assert cls.options is None, type(cls.options)  # This method should only be called once
        cls.options = cast(Options, await storage('steganography', storage_class = Options))
        for stage in Stage:
            cls.ImageBlocks[stage] = ImageBlock(stage)
        for (stage, block) in cls.ImageBlocks.items():
            block.source = cls.ImageBlocks.get(cls.SOURCES.get(stage))  # type: ignore[arg-type]
        cls.worker = await connectToWorker('workerlib')

    @classmethod
    async def loadImages(cls) -> None:
        for block in cls.ImageBlocks.values():
            if block.isUpload:
                await block.loadImageFromCache()

    @classmethod
    async def resetUploads(cls) -> None:
        for stage in cls.PRELOADED_FILES:
            await cls.ImageBlocks[stage].resetUpload()
        await cls.pipeline()

    @classmethod
    async def process(cls,
                      targetStages: Stage | Iterable[Stage],
                      processFunction: Callable[..., Image | Sequence[Image] | None] | Callable[..., Awaitable[Image | Sequence[Image] | None]],
                      sourceStages: Stage | Iterable[Stage],
                      optionalSourceStages: Stage | Iterable[Stage] | None = None,
                      *,
                      options: Iterable[str] | None = None) -> None:
        sources = tuple(cls.ImageBlocks[stage] for stage in ((sourceStages,) if isinstance(sourceStages, Stage) else sourceStages))
        targets = tuple(cls.ImageBlocks[stage] for stage in ((targetStages,) if isinstance(targetStages, Stage) else targetStages))
        if any(not source.image for source in sources):
            for t in targets:
                await t.removeImage()  # ToDo: Make sure images further down the chain get removed too
            await repaint()
            return
        optionalSources = tuple(cls.ImageBlocks[stage] for stage in ((optionalSourceStages,) if isinstance(optionalSourceStages, Stage) else optionalSourceStages or ()))
        if not any(source.dirty for source in chain(sources, optionalSources)):
            return
        log(f"Processing {', '.join(source.stage.name for source in chain(sources, optionalSources))} => {', '.join(target.stage.name for target in targets)}")
        for t in targets:
            t.startOperation(_("Processing image"))
        await repaint()
        try:
            assert cls.options is not None
            sourceImages = tuple(source.image for source in chain(sources, optionalSources))
            options = {option: value for (option, value) in ((option, cls.options.get(option)) for option in (options or ())) if value}  # ToDo: Add method to Options to filter default values
            if iscoroutinefunction(processFunction) or isinstance(processFunction, JsProxy):  # pylint: disable=consider-ternary-expression
                ret = await processFunction(*sourceImages, **options)
            else:
                ret = await to_thread(processFunction, *sourceImages, **options)
            if ret is None:  # No changes were made, use original image
                assert len(sourceImages) == 1 and sourceImages[0], sourceImages  # noqa: PT018
                ret = sourceImages
            elif isinstance(ret, Image):
                ret = (ret,)
            assert ret, repr(ret)
            assert isinstance(ret[0], Image), f"{type(ret)} {type(ret[0])}"
            assert len(ret) == len(targets), f"{processFunction.__name__}() returned {len(ret)} images, expected {len(targets)}"
            for (target, image) in zip(targets, ret, strict = True):
                target.completeOperation(image, imageToBytes(image))
        except Exception as ex:  # noqa : BLE001
            for target in targets:
                target.error(_("processing image"), ex)
            return
        finally:
            await repaint()

    @classmethod
    async def pipeline(cls) -> None:  # Called from upload event handler to generate secondary images
        log("Started pipeline")
        assert cls.worker
        await gather(
            cls.process(Stage.PROCESSED_SOURCE,
                        cls.worker.asyncProcessImage, Stage.SOURCE,  # type: ignore[attr-defined]
                        options = ('resizeFactor', 'resizeWidth', 'resizeHeight', 'randomRotate', 'dither')),
            cls.process(Stage.PROCESSED_LOCK,
                        cls.worker.asyncProcessImage, Stage.LOCK),  # type: ignore[attr-defined]
            cls.process(Stage.PROCESSED_KEY,
                        cls.worker.asyncProcessImage, Stage.KEY),  # type: ignore[attr-defined]
        )
        await cls.process((Stage.GENERATED_LOCK, Stage.GENERATED_KEY),
                          cls.worker.encrypt, Stage.PROCESSED_SOURCE, (Stage.PROCESSED_LOCK, Stage.PROCESSED_KEY),  # type: ignore[attr-defined]
                          options = ('smooth',))
        await cls.process(Stage.KEY_OVER_LOCK_TEST,
                          cls.worker.overlay, (Stage.GENERATED_LOCK, Stage.GENERATED_KEY),  # type: ignore[attr-defined]
                          options = ('border',))  # ToDo: Somehow handle random rotation and location
        # ToDo: Do better job on passing options
        for imageBlock in cls.ImageBlocks.values():
            imageBlock.dirty = False
        log("Completed pipeline")

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
        self.setAttr('title', TEXT, _(self.BLOCK_NAMES[self.stage]))
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
        self.setAttr('name', TEXT, fileName)
        self.setAttr('download-link', 'download', fileName)

    def setURL(self, url: str = '') -> None:
        if src := self.getAttr('display', 'src'):
            revokeObjectURL(src)
        self.setAttr('display', 'src', url)
        self.setAttr('display-link', 'href', url)
        self.setAttr('download-link', 'href', url)

    def setDescription(self, message: str) -> None:
        self.setAttr('description', TEXT, message)
        self.show('description')

    def error(self, message: str, exception: BaseException | None = None) -> None:
        self.setDescription(f"{_("ERROR")} {message}{f": {exception}" if exception else ''}")
        self.hide('remove')
        if exception:
            exceptionHandler("Exception at image processing", exception = exception, showToUser = False)

    def clean(self, description: str | None = None) -> None:
        self.setURL()
        self.dirty = bool(self.image)
        self.image = None
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
            path = Path(filePath)
            fileName = path.name
            log("Fetching preloaded file:", fileName)
            await repaint()
            imageBytes = await fetch(filePath).arrayBuffer()  # type: ignore[attr-defined]
        else:
            (imageBytes, fileName) = (None, None)
        if imageBytes:
            try:
                log("Loading image:", fileName)
                await repaint()
                image = loadImage(imageBytes, fileName)
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
        if not fileName:  # E.g. Esc was pressed at upload dialog
            return
        self.startOperation(_("Loading image"))
        await repaint()
        try:
            log("Uploading image:", fileName)
            await repaint()
            assert data, repr(data)
            imageBytes = await blobToBytes(data)
            image = loadImage(imageBytes)
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
    # Filter traceback to remove empty lines:
    tracebackStr = '\n' + '\n'.join(line for line in '\n'.join(extract_tb(traceback).format()).splitlines() if line.strip()) if traceback else ''
    log(f"""
ERROR {problem}, type {exceptionType.__name__}: {exception}{tracebackStr}

Please make a screenshot and report it to @jolaf at Telegram or VK or to vmzakhar@gmail.com. Thank you!
""", showToUser = showToUser)

@typechecked
def mainExceptionHandler(exceptionType: type[BaseException] | None = None,
                         exception: BaseException | None = None,
                         traceback: TracebackType | None = None) -> None:
    exceptionHandler("Uncaught exception in main thread",
                     exceptionType, exception, traceback)

@typechecked
def loopExceptionHandler(_loop: AbstractEventLoop,
                         context: dict[str, Any]) -> None:
    exceptionHandler("Uncaught exception in async loop",
                     exception = context.get('exception'))

@typechecked
async def main() -> None:
    log("Starting app")
    sys.excepthook = mainExceptionHandler
    get_running_loop().set_exception_handler(loopExceptionHandler)
    log("PyScript v" + pyscriptVersion)
    log("Pyodide v" + pyodideVersion)
    log("Python v" + pythonVersion)
    log("Pillow v" + pilVersion)
    log("NumPy v" + numpyVersion)
    if beartypeVersion:
        try:
            @typechecked
            def test() -> int:
                return 'notInt'  # type: ignore[return-value]
            test()
            raise RuntimeError("Beartype v" + beartypeVersion + " is not operating properly")
        except BeartypeException:
            log("Beartype v" + beartypeVersion + " is up and watching, remove it from PyScript configuration to make things faster")
    await repaint()
    await ImageBlock.init()
    hide('log')
    show('content')
    log("Loading cached images")
    await ImageBlock.loadImages()
    await ImageBlock.pipeline()
    await repaint()
    log("Started app")

if __name__ == '__main__':
    create_task(main())  # `create_task()` is only needed to silence static checkers that don't like `await` in global module code  # noqa: RUF006

print(f"{PREFIX} Loaded app")
