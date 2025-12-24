from __future__ import annotations
from asyncio import sleep
from sys import version
from typing import Any, TYPE_CHECKING

import pyscript
from pyscript import when, window  # type: ignore[attr-defined]  # pylint: disable=no-name-in-module
from pyscript.web import page  # type: ignore[import-not-found]  # pylint: disable=import-error, no-name-in-module
from pyscript.ffi import to_js  # type: ignore[import-not-found]  # pylint: disable=import-error, no-name-in-module

from pyodide_js import version as pyodideVersion  # type: ignore[import-not-found]  # pylint: disable=import-error

from Steganography import getImageMode, imageToBytes, loadImage, processImage

if TYPE_CHECKING:
    Event = Any  # Workarounds for mypy as stuff cannot be imported from PyScript when not in a browser
    Element = Any
    Blob = Any
    Uint8Array = Any
    def createObjectURL(_file: Any) -> str: return ''  # pylint: disable=multiple-statements
    def revokeObjectURL(_url: str) -> None: pass  # pylint: disable=multiple-statements
else:
    Blob = window.Blob  # Simplifying addressing to JS classes and functions
    Uint8Array = window.Uint8Array
    windowOpen = window.open
    createObjectURL = window.URL.createObjectURL
    revokeObjectURL = window.URL.revokeObjectURL
    window = None  # For cleaner code, make sure all used references are mentioned here

# Global error handling
# Task name (default: date_time)
# Input image (upload, regenerate)
# Lock image (upload, remove)
# Key image (upload, remove)
# Lock result (download)
# Key result (download)
# Test (download) - make it with really two images overlaid!
# Move loading log to the bottom
# Add copyright to the bottom
# Check Rigging HTML for ideas

# Options:
# Resize input image to: width, height
# Key size: xN, width, height
# Random key position (check)
# Random key rotation (check)
# Resize key to lock?
# Better background (check)
# Regenerate

def log(*args: str) -> None:
    print(*args)

async def repaint() -> None:
    await sleep(0.1)  # Yield control to the browser so that repaint could happen

def hide(id: str) -> None:
    page['#' + id][0].classes.add('hidden')

def show(id: str) -> None:
    page['#' + id][0].classes.remove('hidden')

def setText(id: str, text: str) -> None:
    page['#' + id][0].innerText = text

async def processUpload(file: Any, fileSize: int) -> tuple[bytes, str]:
    description = f"{fileSize} bytes"
    # noinspection PyRedundantParentheses
    return (Uint8Array.new(await file.arrayBuffer()).to_bytes(), description)

def showImage(name: str, byteArray: bytes | Uint8Array | None, description: str = "") -> None:
    if src := page['#image-display-' + name][0].src:
        revokeObjectURL(src)
    if byteArray is None:
        page['#image-display-' + name][0].src = ''
        hide('image-display-' + name)
        setText('image-description-' + name, description)
    else:
        if isinstance(byteArray, bytes):
            byteArray = Uint8Array.new(byteArray)
        blob = Blob.new([byteArray,], to_js({'type': 'image/png'}))  # to_js() converts Python dict into JS object
        url = createObjectURL(blob)
        page['#image-display-' + name][0].src = url
        show('image-display-' + name)
        try:
            page['#image-display-link-' + name][0].href = url
            page['#image-download-link-' + name][0].href = url
        except:  # ToDo: remove this after unification with renderImage()
            pass
        setText('image-description-' + name, description)

def renderImage(name: str, upload: bool = False) -> None:
    html = f'''
<div id="image-{name}" class="image-block">
  <h2 id="image-title-{name}" class="image-title">{name.capitalize()} file</h2>
  <button id="image-download-{name}" class="image-download" title="Click to download to your computer">Download</button>
  <label id="image-name-{name}" class="image-name">{name}.png</label>
  <a id="image-download-link-{name}" class="image-download-link hidden" download="{name}.png" href=""></a>
  <br>
  <label id="image-description-{name}" class="image-description"{f' for="upload-input-{name}' if upload else ''}>Processing image...</label>
  <br>
  <a id="image-display-link-{name}" class="image-display-link" title="Click for full screen preview" target="_blank" href="">
    <img id="image-display-{name}" class="image-display hidden" src="">
  </a>
</div>'''  # ToDo: Use pythonic DOM for this?
    # ToDo: hide link instead of image
    page['#generated'][0].innerHTML = html

    @when('click', '#image-display-' + name)  # type: ignore[untyped-decorator]
    async def popupEventHandler(_e: Event) -> None:
        page['#image-display-link-' + name][0].click()

    @when('click', '#image-download-' + name)  # type: ignore[untyped-decorator]
    async def downloadEventHandler(_e: Event) -> None:
        page['#image-download-link-' + name][0].click()

def renderUpload(name: str) -> None:  # ToDo: Move this functionality into renderImage()
    html = f'''
<div id="image-{name}" class="image-block">
  <h2 id="image-title-{name}" class="image-title">{name.capitalize()} file</h2>
  <input id="upload-input-{name}" class="upload-input" title="Click to upload a file to process" type="file">
  <br>
  <label id="image-description-{name}" class="image-description" for="upload-input-{name}"></label>
  <br>
  <img id="image-display-{name}" class="image-display hidden" src="">
</div>'''
    page['#uploads'][0].innerHTML = html

    @when('change', '#upload-input-' + name)  # type: ignore[untyped-decorator]
    async def uploadEventHandler(e: Event) -> None:
        showImage(name, None)
        page['#generated'].innerHTML = ''
        if e.target.value:  # file name
            hide('image-display-' + name)
            page['#image-description-' + name][0].innerText = "Loading image..."
            await repaint()
            file = e.target.files.item(0)
            (byteArray, description) = await processUpload(file, file.size)
            try:
                image = loadImage(byteArray)
            except Exception as ex:  # noqa : BLE001
                showImage(name, None, f"Error loading image: {ex}")
                return
            showImage(name, imageToBytes(image), f"{description} {image.format} {getImageMode(image)} {image.width}x{image.height}")
            renderImage('processed')
            await repaint()
            try:
                image = processImage(image)
            except Exception as ex:  # noqa : BLE001
                showImage(name, None, f"Error processing image: {ex}")
                return
            showImage('processed', imageToBytes(image), f"{getImageMode(image)} {image.width}x{image.height}")

def showVersions() -> None:
    page['#versions'][0].innerText = f'''\
PyScript: {getattr(pyscript, 'version', None)
        or getattr(pyscript, '__version__', None)
        or next(tag.src for tag in page['script']).split('/')[-2]
        or "UNKNOWN"}
Pyodide: {pyodideVersion}
Python: {version}'''

def finishLoading() -> None:
    show('content')

def main() -> None:
    showVersions()
    renderUpload('source')
    finishLoading()

if __name__ == '__main__':
    main()
