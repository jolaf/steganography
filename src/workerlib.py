#
# Note: this library is intended for PyScript apps, it's mostly useless outside of browser
#
# This lib wraps all calls from main thread to worker in decorators that unpack `JsProxy`
# objects in function arguments and return values.
#
# To create a worker with this lib, you have two ways:
#
# 1.
# index.html:
# <script type="py" worker name="workerlib" src="./workerlib.py" config="./workerlib.toml"></script>
#
# workerlib.toml:
# [files]
# "./YourModule1.py" = ""
# "./YourModule2.py" = ""
#
# [exports]
# "YourModule1" = ["func1", "func2", ..., "funcK"]
# "YourModule2" = ["func3", "func4", ..., "funcN"]
#
#
# 2.
# index.html:
# <script type="py" worker name="workerlib" src="./YourModule1.py" config="./worker.toml"></script>
#
# worker.toml:
# [files]
# "./workerlib.py" = ""
# "./YourModule2.py" = ""
#
# YourModule1.py:
# from workerlib import export
# from YourModule2 import func3, func4, ..., funcN
# export(func1, func2, ..., funcK, func3, func4, ..., funcN)
## Call `export()` exactly once with ALL the functions you need to export to the main thread!
#
#
# In main PyScript module, in both cases, go like this:
#
# import workerlib
# ret = await workerlib.worker.func1(*args, **kwargs)  # ToDo: update
# OR
# ret = await workerlib.func1(*args, **kwargs)
#
# OR
# from workerlib import *
# ret = await func1(*args, **kwargs)
#

# ToDo: Maybe import decorators from some another module too, the same way, and list (additional) decorators to use in a config option?
# ToDo: create translation table for types and converters to-from bytes?

# ToDo: Create a separate toml config?
# ToDo: Copy logging from main? Or import??
# ToDo: Find section [worker] in config, treat as dict, get name, other entries with keys = modules, values = functions, import them, wrap them, export them?
# ToDo: Maybe import decorators from some another module too, the same way, and list (additional) decorators to use in a config option?

from collections.abc import Coroutine, Buffer, Callable, Iterable, Mapping, Sequence
from functools import wraps
from typing import Any, Never

from pyscript import RUNNING_IN_WORKER

from pyodide.ffi import JsProxy  # pylint: disable=import-error, no-name-in-module

from Steganography import imageToBytes, loadImage, Image  # ToDo: Remove this, add filters through config

PREFIX = ""

CONNECT_REQUEST = b'__REQUEST__'
CONNECT_RESPONSE = b'__RESPONSE__'

__all__: Sequence[str]
__export__: Sequence[str]

def log(*args: Any) -> None:
    print(PREFIX, *args)

def error(message: str) -> Never:
    raise RuntimeError(f"{PREFIX} {message}")

def to_py(obj: Any) -> Any:
    if hasattr(obj, 'to_py'):
        return obj.to_py()
    if hasattr(obj, 'to_bytes'):
        return obj.to_bytes()
    if isinstance(obj, Mapping):
        return {to_py(k): to_py(v) for (k, v) in obj.items()}
    if isinstance(obj, Iterable):
        return tuple(to_py(obj) for obj in obj)
    return obj

if RUNNING_IN_WORKER:

    from importlib import import_module
    from inspect import iscoroutinefunction
    from itertools import chain

    PREFIX = "[worker]"
    __export__ = ()

    log("Starting worker")

    try:  # To turn runtime typechecking on, add "beartype" to `packages` array in your `workerlib.toml`
        from beartype import beartype as typechecked, __version__
        log(f"Beartype v{__version__} is up and watching, remove it from worker configuration to make things faster")
    except ImportError:
        log("WARNING: beartype is not available, running fast with typing unchecked")

        def typechecked(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[no-redef]
            return func

    @typechecked
    def serialized(func: Callable[..., Any | Coroutine[None, None, Any]]) -> Callable[..., Coroutine[None, None, Any]]:
        @wraps(func)
        @typechecked
        async def serializedWrapper(*args: Any, **kwargs: Any) -> Any:
            assert not kwargs  # kwargs get passed to workers as last of args, of type dict
            args = tuple(to_py(arg) for arg in args)
            if args and isinstance(args[-1], dict):
                kwargs = args[-1]
                args = args[:-1]
            if iscoroutinefunction(func):  # pylint: disable=consider-ternary-expression
                ret = await func(*args, **kwargs)
            else:
                ret = func(*args, **kwargs)
            return ret
        return serializedWrapper

    @typechecked
    def images(func: Callable[..., Any | Coroutine[None, None, Any]]) -> Callable[..., Coroutine[None, None, Any]]:
        @wraps(func)
        @typechecked
        async def imagesWrapper(*args: Any, **kwargs: Any) -> Any:
            args = tuple(loadImage(arg) if isinstance(arg, Buffer) else arg for arg in args)
            if iscoroutinefunction(func):  # pylint: disable=consider-ternary-expression
                ret = await func(*args, **kwargs)
            else:
                ret = func(*args, **kwargs)
            if isinstance(ret, Image):
                return imageToBytes(ret)
            if isinstance(ret, Iterable):
                return tuple(imageToBytes(r) if isinstance(r, Image) else r for r in ret)
            return ret
        return imagesWrapper

    @typechecked
    def logged(func: Callable[..., Any | Coroutine[None, None, Any]]) -> Callable[..., Coroutine[None, None, Any]]:
        @wraps(func)
        @typechecked
        async def loggedWrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                if iscoroutinefunction(func):
                    log(f"Awaiting {func.__name__}(): {args} {kwargs}")
                    ret = await func(*args, **kwargs)
                else:
                    log(f"Calling {func.__name__}(): {args} {kwargs}")
                    ret = func(*args, **kwargs)
                log(f"Returned from {func.__name__}(): {ret}")
                return ret  # noqa: TRY300
            except BaseException as ex:
                log(f"Exception at {func.__name__}: {ex}")
                raise
        return loggedWrapper

    @serialized
    @typechecked
    def _connect(data: Buffer) -> tuple[bytes, ...]:
        if data == CONNECT_REQUEST:
            log("Connected to main thread, ready for requests")
            assert __export__, __export__
            return tuple(chain((CONNECT_RESPONSE,), (name.encode() for name in __export__ if name != _connect.__name__)))
        error(f"Connection to main thread is misconfigured, can't continue: {type(data)}({data!r})")

    @typechecked
    def wrap(func: Callable[..., Any | Coroutine[None, None, Any]]) -> Callable[..., Coroutine[None, None, Any]]:
        return serialized(images(logged(typechecked(func))))

    # Must be called by the importing module to actually start providing worker service
    @typechecked
    def export(*functions: Callable[..., Any | Coroutine[None, None, Any]]) -> None:
        from sys import _getframe  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        target = _getframe(1).f_globals  # globals of the calling module
        target[_connect.__name__] = _connect
        exportNames = [_connect.__name__,]

        for func in functions:
            target[func.__name__] = wrap(func)
            exportNames.append(func.__name__)

        exportNamesTuple = tuple(exportNames)
        target['__export__'] = exportNamesTuple  # ToDo: Append, not overwrite
        globals()['__export__'] = exportNamesTuple  # This is only needed to make `_connect()` code universal for both `export()` and `exportFromMapping()`
        log(f"Started worker, providing functions: {', '.join(name for name in exportNamesTuple if name != _connect.__name__)}")

    # Gets called automatically if this module itself is loaded as a worker
    @typechecked
    def exportFromMapping(mapping: Mapping[str, Iterable[str]] | None) -> None:
        exportNames = [_connect.__name__,]
        target = globals()

        if mapping:
            for (moduleName, funcNames) in mapping.items():
                log(f"Importing from module {moduleName}: {', '.join(funcNames)}")
                module = import_module(moduleName)

                for funcName in funcNames:
                    func = getattr(module, funcName)
                    target[funcName] = wrap(func)
                    exportNames.append(funcName)
        else:
            log("WARNING: no functions found to export, check `[exports]` section in the config")

        target['__export__'] = tuple(exportNames)  # ToDo: Append, not overwrite
        log(f"Started worker, providing functions: {', '.join(name for name in exportNames if name != _connect.__name__)}")

    if __name__ == '__main__':
        # If this module itself is used as a worker, it imports modules mentioned in config and exports them automatically
        from pyscript import config  # pylint: disable=ungrouped-imports
        exportFromMapping(config.get('exports'))
        del config
        __all__ = ()
    else:
        # If user is importing this module in a worker, they MUST call `export()` explicitly
        __all__ = (export.__name__,)  # noqa: PLE0604

else:  # Main thread

    from pyscript import workers

    try:  # To turn runtime typechecking on, add "beartype" to `packages` array in your `main.toml`
        from beartype import beartype as typechecked  # pylint: disable=ungrouped-imports
    except ImportError:
        def typechecked(func: Callable[..., Coroutine[None, None, Any]]) -> Callable[..., Coroutine[None, None, Any]]:  # type: ignore[no-redef]
            return func

    PREFIX = "[main]"

    class Worker:
        def __init__(self, worker: JsProxy) -> None:
            self.worker = worker

    @typechecked
    def images(func: Callable[..., Coroutine[None, None, Any | Coroutine[None, None, Any]]]) -> Callable[..., Coroutine[None, None, Any]]:
        @wraps(func)
        @typechecked
        async def imagesWrapper(*args: Any, **kwargs: Any) -> Any:
            args = tuple(imageToBytes(arg) if isinstance(arg, Image) else arg for arg in args)
            ret = await func(*args, **kwargs)
            if isinstance(ret, Buffer):
                return loadImage(ret)
            if isinstance(ret, Iterable):
                return tuple(loadImage(r) if isinstance(r, Buffer) else r for r in ret)
            return ret
        return imagesWrapper

    @typechecked
    def serialized(func: Callable[..., Coroutine[None, None, Any | Coroutine[None, None, Any]]]) -> Callable[..., Coroutine[None, None, Any]]:
        @wraps(func)
        @typechecked
        async def serializedWrapper(*args: Any, **kwargs: Any) -> Any:
            return to_py(await func(*args, **kwargs))
        return serializedWrapper

    @typechecked
    async def connectWorker(workerName: str) -> Worker:
        log(f'Looking for worker named "{workerName}"')
        worker = await workers[workerName]
        log("Got worker, connecting")
        data = await serialized(worker._connect)(CONNECT_REQUEST)  # noqa: SLF001  # pylint: disable=protected-access
        if not data or data[0] != CONNECT_RESPONSE:
            error(f"Connection to worker is misconfigured, can't continue: {type(data)}: {data!r}")
        ret = Worker(worker)
        for b in data[1:]:
            funcName = bytes(b).decode()
            assert funcName != connectWorker.__name__
            if not (func := getattr(worker, funcName, None)):
                error(f"Function {funcName} is not exported from the worker")
            func = images(serialized(typechecked(func)))
            setattr(ret, funcName, func)
        log("Connected to worker")
        return ret

    __all__ = ('Worker', 'connectWorker')
