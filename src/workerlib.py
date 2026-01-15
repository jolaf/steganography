#
# Note: this library is intended for PyScript apps, it's useless outside of browser
#
# This lib wraps all calls from main thread to worker in decorators that unpack `JsProxy`
# objects in function arguments and return values, providing transparent Python-to-Python calls.
#
# Both synchronous and `async` functions can be exported,
# in both cases they're accessed from main thread using `await`.
#
# To create a worker with this lib, you have two ways:
#
# 1.
# index.html:
# <script type="py" worker name="YourWorkerName" src="./workerlib.py" config="./workerlib.toml"></script>
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
# <script type="py" worker name="YourWorkerName" src="./YourModule1.py" config="./worker.toml"></script>
#
# worker.toml:
# [files]
# "./workerlib.py" = ""
# "./YourModule2.py" = ""
#
# YourModule1.py:
# from workerlib import export
# from YourModule2 import func3, func4, ..., funcN
# export(func1, func2, ..., funcK)
# export(func3, func4, ..., funcN)
#
#
# In main PyScript module, in both cases, go like this:
#
# from workerlib import connectToWorker, Worker
# worker: Worker = await connectToWorker("YourWorkerName")
# ret = await worker.func1(*args, **kwargs)
#
# If you need access to actual `worker` object, it's there as `worker.worker`.
#

# ToDo: Maybe import decorators from some another module too, the same way, and list (additional) decorators to use in a config option?
# ToDo: create translation table for types and converters to-from bytes?

from collections.abc import Coroutine, Buffer, Callable, Iterable, Mapping, Sequence
from functools import wraps
from typing import Any, Never

from pyscript import RUNNING_IN_WORKER

try:
    from pyodide.ffi import JsProxy
except ImportError:
    type JsProxy = Any  # type: ignore[no-redef]

from Steganography import imageToBytes, loadImage, Image  # ToDo: Remove this, add filters through config

_PREFIX = ""

_CONNECT_REQUEST = b'__REQUEST__'
_CONNECT_RESPONSE = b'__RESPONSE__'

__all__: Sequence[str]
__export__: Sequence[str]

type _Coroutine = Coroutine[None, None, Any]
type _CoroutineFunction = Callable[..., _Coroutine]
type _FunctionOrCoroutine = Callable[..., Any | _Coroutine]

def _log(*args: Any) -> None:
    print(_PREFIX, *args)

def _error(message: str) -> Never:
    raise RuntimeError(f"{_PREFIX} {message}")

def _to_py(obj: Any) -> Any:
    if hasattr(obj, 'to_py'):
        return obj.to_py()
    if hasattr(obj, 'to_bytes'):
        return obj.to_bytes()
    if isinstance(obj, Mapping):
        return {_to_py(k): _to_py(v) for (k, v) in obj.items()}
    if isinstance(obj, Iterable):
        return tuple(_to_py(obj) for obj in obj)
    return obj

if RUNNING_IN_WORKER:

    from importlib import import_module
    from inspect import iscoroutinefunction
    from itertools import chain

    _PREFIX = "[worker]"  # We name everything starting with underscore to avoid potential conflicts with exported user functions

    _log("Starting worker")

    try:  # To turn runtime typechecking on, add "beartype" to `packages` array in your `workerlib.toml`
        from beartype import beartype as _typechecked, __version__ as _version
        _log(f"Beartype v{_version} is up and watching, remove it from worker configuration to make things faster")
    except ImportError:
        _log("WARNING: beartype is not available, running fast with typing unchecked")

        def _typechecked(func: _FunctionOrCoroutine) -> _FunctionOrCoroutine:  # type: ignore[no-redef]
            return func

    @_typechecked
    def _serialized(func: _FunctionOrCoroutine) -> _CoroutineFunction:
        @wraps(func)
        @_typechecked
        async def workerSerializedWrapper(*args: Any, **kwargs: Any) -> Any:
            assert not kwargs  # kwargs get passed to workers as last of args, of type dict
            args = tuple(_to_py(arg) for arg in args)
            if args and isinstance(args[-1], dict):
                kwargs = args[-1]
                args = args[:-1]
            if iscoroutinefunction(func):  # pylint: disable=consider-ternary-expression
                ret = await func(*args, **kwargs)
            else:
                ret = func(*args, **kwargs)
            return ret
        return workerSerializedWrapper

    @_typechecked
    def _images(func: _FunctionOrCoroutine) -> _CoroutineFunction:
        @wraps(func)
        @_typechecked
        async def workerImagesWrapper(*args: Any, **kwargs: Any) -> Any:
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
        return workerImagesWrapper

    @_typechecked
    def _logged(func: _FunctionOrCoroutine) -> _CoroutineFunction:
        @wraps(func)
        @_typechecked
        async def workerLoggedWrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                if iscoroutinefunction(func):
                    _log(f"Awaiting {func.__name__}(): {args} {kwargs}")
                    ret = await func(*args, **kwargs)
                else:
                    _log(f"Calling {func.__name__}(): {args} {kwargs}")
                    ret = func(*args, **kwargs)
                _log(f"Returned from {func.__name__}(): {ret}")
                return ret  # noqa: TRY300
            except BaseException as ex:
                _log(f"Exception at {func.__name__}: {ex}")
                raise
        return workerLoggedWrapper

    @_serialized
    @_typechecked
    def _connectFromMain(data: Buffer) -> tuple[bytes, ...]:
        if data == _CONNECT_REQUEST:
            _log("Connected to main thread, ready for requests")
            assert __export__, __export__
            return tuple(chain((_CONNECT_RESPONSE,), (name.encode() for name in __export__ if name != _connectFromMain.__name__)))
        _error(f"Connection to main thread is misconfigured, can't continue: {type(data)}({data!r})")

    @_typechecked
    def _wrap(func: _FunctionOrCoroutine) -> _CoroutineFunction:
        return _serialized(_images(_logged(_typechecked(func))))

    # Must be called by the importing module to actually start providing worker service
    @_typechecked
    def export(*functions: _FunctionOrCoroutine) -> None:
        from sys import _getframe  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        target = _getframe(1).f_globals  # globals of the calling module
        target[_connectFromMain.__name__] = _connectFromMain
        exportNames = [_connectFromMain.__name__, ]

        for func in functions:
            target[func.__name__] = _wrap(func)
            exportNames.append(func.__name__)

        exportNamesTuple = tuple(exportNames)
        target['__export__'] = exportNamesTuple  # ToDo: Append, not overwrite
        globals()['__export__'] = exportNamesTuple  # This is only needed to make `_connectFromMain()` code universal for both `export()` and `_exportFromMapping()`
        _log(f"Started worker, providing functions: {', '.join(name for name in exportNamesTuple if name != _connectFromMain.__name__)}")

    # Gets called automatically if this module itself is loaded as a worker
    @_typechecked
    def _exportFromMapping(mapping: Mapping[str, Iterable[str]] | None) -> None:
        exportNames = [_connectFromMain.__name__, ]
        target = globals()

        if mapping:
            for (moduleName, funcNames) in mapping.items():
                _log(f"Importing from module {moduleName}: {', '.join(funcNames)}")
                module = import_module(moduleName)

                for funcName in funcNames:
                    target[funcName] = _wrap(getattr(module, funcName))
                    exportNames.append(funcName)
        else:
            _log("WARNING: no functions found to export, check `[exports]` section in the config")

        target['__export__'] = tuple(exportNames)  # ToDo: Append, not overwrite
        _log(f"Started worker, providing functions: {', '.join(name for name in exportNames if name != _connectFromMain.__name__)}")

    if __name__ == '__main__':
        # If this module itself is used as a worker, it imports modules mentioned in config and exports them automatically
        from pyscript import config  # pylint: disable=ungrouped-imports
        __export__ = ()
        _exportFromMapping(config.get('exports'))
        assert __export__, __export__
        del config  # type: ignore[unreachable]
        del export
        __all__ = ()
    else:
        # If user is importing this module in a worker, they MUST call `export()` explicitly
        __all__ = (export.__name__,)  # noqa: PLE0604
        del __export__
        del _exportFromMapping

       ##
else:  #  MAIN THREAD
       ##

    from pyscript import workers

    try:  # To turn runtime typechecking on, add "beartype" to `packages` array in your `main.toml`
        from beartype import beartype as typechecked  # pylint: disable=ungrouped-imports
    except ImportError:
        def typechecked(func: _FunctionOrCoroutine) -> _FunctionOrCoroutine:  # type: ignore[no-redef]
            return func

    _PREFIX = "[main]"

    @typechecked
    class Worker:
        def __init__(self, worker: JsProxy) -> None:
            self.worker = worker

    @typechecked
    def images(func: _CoroutineFunction) -> _CoroutineFunction:
        @wraps(func)
        @typechecked
        async def mainImagesWrapper(*args: Any, **kwargs: Any) -> Any:
            args = tuple(imageToBytes(arg) if isinstance(arg, Image) else arg for arg in args)
            ret = await func(*args, **kwargs)
            if isinstance(ret, Buffer):
                return loadImage(ret)
            if isinstance(ret, Iterable):
                return tuple(loadImage(r) if isinstance(r, Buffer) else r for r in ret)
            return ret
        return mainImagesWrapper

    @typechecked
    def serialized(func: _CoroutineFunction) -> _CoroutineFunction:
        @wraps(func)
        @typechecked
        async def mainSerializedWrapper(*args: Any, **kwargs: Any) -> Any:
            return _to_py(await func(*args, **kwargs))
        return mainSerializedWrapper

    @typechecked
    def wrap(func: _CoroutineFunction) -> _CoroutineFunction:
        return images(serialized(func))  # Note the reverse call order relative to worker's `_wrap()`

    @typechecked
    async def connectToWorker(workerName: str) -> Worker:
        _log(f'Looking for worker named "{workerName}"')
        worker = await workers[workerName]
        _log("Got worker, connecting")
        data = await serialized(worker._connectFromMain)(_CONNECT_REQUEST)  # noqa: SLF001  # pylint: disable=protected-access
        if not data or data[0] != _CONNECT_RESPONSE:
            _error(f"Connection to worker is misconfigured, can't continue: {type(data)}: {data!r}")
        ret = Worker(worker)  # We can't return `worker`, as it is a `JsProxy` and we can't reassign its fields, `setattr()` is not working, so we have to create a class of our own to store `wrap()`ped functions.
        for b in data[1:]:
            funcName = bytes(b).decode()  # We can't get a list of exported functions from `worker` object to `wrap()` them, so we have to pass it over in our own way.
            assert funcName != connectToWorker.__name__
            if not (func := getattr(worker, funcName, None)):
                _error(f"Function {funcName} is not exported from the worker")
            setattr(ret, funcName, wrap(func))
        _log("Connected to worker")
        return ret

    assert not hasattr(globals(), '__export__'), getattr(globals(), '__export__')  # noqa: B009
    __all__ = ('Worker', 'connectToWorker')
