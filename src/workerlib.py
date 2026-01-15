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

from collections.abc import Coroutine, Buffer, Callable, Iterable, Mapping, Sequence
from functools import wraps
from importlib import import_module
from inspect import isfunction, iscoroutinefunction
from itertools import chain
from typing import Any, NoReturn

from pyscript import config, RUNNING_IN_WORKER

try:
    from pyodide.ffi import JsProxy
except ImportError:
    type JsProxy = Any  # type: ignore[no-redef]

# We name everything starting with underscore to avoid potential conflicts with exported user functions
type _Coroutine = Coroutine[None, None, Any]
type _CoroutineFunction = Callable[..., _Coroutine]
type _FunctionOrCoroutine = Callable[..., Any | _Coroutine]
_TransportSafe = int | float | bool | str | Buffer | Iterable | Sequence | Mapping | None  # type: ignore[type-arg]

try:  # To turn runtime typechecking on, add "beartype" to `packages` array in your `workerlib.toml`
    from beartype import beartype as _typechecked
except ImportError:
    def _typechecked(func: _FunctionOrCoroutine) -> _FunctionOrCoroutine:  # type: ignore[no-redef]
        return func

type _Adapter = tuple[type, Callable[[Any], Any | _Coroutine], Callable[[Any], Any | _Coroutine]]

_PREFIX = ""

_CONNECT_REQUEST = b'__REQUEST__'
_CONNECT_RESPONSE = b'__RESPONSE__'

_ADAPTER_PREFIX = '_workerlib_'

__all__: Sequence[str]
__export__: Sequence[str]
__adapters__: Sequence[_Adapter] = ()

@_typechecked
def _log(*args: Any) -> None:
    print(_PREFIX, *args)

@_typechecked
def _error(message: str) -> NoReturn:
    raise RuntimeError(f"{_PREFIX} {message}")

@_typechecked
async def _from_py(obj: Any) -> Any:
    if encoded := await _adapterEncode(obj):
        return encoded
    if isinstance(obj, str | Buffer):
        return obj
    if isinstance(obj, Mapping):
        return {await _from_py(k): await _from_py(v) for (k, v) in obj.items()}
    if isinstance(obj, Iterable):
        return tuple([await _from_py(v) for v in obj])  # pylint: disable=consider-using-generator
    return obj

@_typechecked
async def _to_py(obj: Any) -> Any:
    if hasattr(obj, 'to_py'):
        return await _to_py(obj.to_py())
    if isinstance(obj, Mapping):
        obj = {await _to_py(k): await _to_py(v) for (k, v) in obj.items()}
    elif not isinstance(obj, str | Buffer) and isinstance(obj, Iterable):
        obj = tuple([await _to_py(v) for v in obj])  # pylint: disable=consider-using-generator
    return await _adapterDecode(obj)

@_typechecked
def _adaptersFrom(mapping: Mapping[str, Sequence[str]] | None) -> None:
    if not mapping:
        return
    ret: list[_Adapter] = []
    for (moduleName, names) in mapping.items():
        if len(names) != 3:
            _error(f'''Bad adapter settings for module {moduleName}, must be '["className", "encoderFunction", "decoderFunction"]', got {names!r} ''')
        _log(f"Importing from module {moduleName}: {', '.join(names)}")
        module = import_module(moduleName)
        (cls, encoder, decoder) = (getattr(module, name) for name in names)
        if not isinstance(cls, type):
            _error(f"Bad adapter class {names[0]} for module {moduleName}, must be type, got {type(cls)}")
        if not isfunction(encoder) and not iscoroutinefunction(encoder):
            _error(f"Bad adapter encoder {names[1]} for module {moduleName}, must be function or coroutine, got {type(encoder)}")
        if not isfunction(decoder) and not iscoroutinefunction(decoder):
            _error(f"Bad adapter encoder {names[2]} for module {moduleName}, must be function or coroutine, got {type(decoder)}")
        ret.append((cls, encoder, decoder))
    global __adapters__  # noqa: PLW0603  # pylint: disable=global-statement
    __adapters__ = tuple(ret)

@_typechecked
async def _adapterEncode(obj: Any) -> tuple[str, Any] | None:
    for (cls, encoder, _decoder) in __adapters__:
        if isinstance(obj, cls):
            if iscoroutinefunction(encoder):
                value = await encoder(obj)
            else:
                value = encoder(obj)
            return (_ADAPTER_PREFIX + cls.__name__, value)  # Encoded class name is NOT the name of type of object being encoded, but name of the adapter to use to decode the object on the other side
    if not isinstance(obj, _TransportSafe):
        _error(f"No adapter found for class {type(obj)}, and transport layer (JavaScript structured clone) would not accept it as is, see https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Structured_clone_algorithm")
    return None

@_typechecked
async def _adapterDecode(obj: Any) -> Any:
    if not isinstance(obj, tuple) or len(obj) != 2:
        return obj
    (name, value) = obj
    if not isinstance(name, str):
        return obj
    tokens = name.split(_ADAPTER_PREFIX)
    if len(tokens) != 2 or tokens[0]:
        return obj
    className = tokens[1]
    for (cls, _encoder, decoder) in __adapters__:
        if className == cls.__name__:
            if iscoroutinefunction(decoder):
                return await decoder(value)
            return decoder(value)
    _error(f"No adapter found to decode class {className}")

@_typechecked
async def _to_transport(obj: Any) -> Any:
    return await _from_py(obj)

@_typechecked
async def _from_transport(obj: Any) -> Any:
    return await _to_py(obj)

if RUNNING_IN_WORKER:

    _PREFIX = "[worker]"

    _log("Starting worker")

    try:  # To turn runtime typechecking on, add "beartype" to `packages` array in your `workerlib.toml`
        from beartype import __version__ as _version
        _log(f"Beartype v{_version} is up and watching, remove it from worker configuration to make things faster")
    except ImportError:
        _log("WARNING: beartype is not available, running fast with typing unchecked")

    @_typechecked
    def _workerSerialized(func: _FunctionOrCoroutine) -> _CoroutineFunction:
        @wraps(func)
        @_typechecked
        async def workerSerializedWrapper(*args: Any, **kwargs: Any) -> Any:
            assert not kwargs  # kwargs get passed to workers as last of args, of type dict
            args = await _from_transport(args)
            if args and isinstance(args[-1], dict):
                kwargs = args[-1]
                args = args[:-1]
            if iscoroutinefunction(func):  # pylint: disable=consider-ternary-expression
                ret = await func(*args, **kwargs)
            else:
                ret = func(*args, **kwargs)
            return await _to_transport(ret)
        return workerSerializedWrapper

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

    @_workerSerialized
    @_typechecked
    def _connectFromMain(data: Buffer) -> tuple[bytes, ...]:
        if data == _CONNECT_REQUEST:
            _log("Connected to main thread, ready for requests")
            assert __export__, __export__
            return tuple(chain((_CONNECT_RESPONSE,), (name.encode() for name in __export__ if name != _connectFromMain.__name__)))
        _error(f"Connection to main thread is misconfigured, can't continue: {type(data)}({data!r})")

    @_typechecked
    def _wrap(func: _FunctionOrCoroutine) -> _CoroutineFunction:
        return _workerSerialized(_logged(_typechecked(func)))

    # Must be called by the importing module to actually start providing worker service
    @_typechecked
    def export(*functions: _FunctionOrCoroutine) -> None:
        from sys import _getframe  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        target = _getframe(1).f_globals  # globals of the calling module
        target[_connectFromMain.__name__] = _connectFromMain
        exportNames: list[str] = []
        if _connectFromMain.__name__ not in target.get('__export__', ()):
            exportNames.append(_connectFromMain.__name__)

        for func in functions:
            target[func.__name__] = _wrap(func)
            exportNames.append(func.__name__)

        exportNamesTuple = tuple(chain(target.get('__export__', ()), exportNames))
        target['__export__'] = exportNamesTuple
        globals()['__export__'] = exportNamesTuple  # This is only needed to make `_connectFromMain()` code universal for both `export()` and `_exportFrom()`
        _log(f"Started worker, providing functions: {', '.join(name for name in exportNamesTuple if name != _connectFromMain.__name__)}")

    # Gets called automatically if this module itself is loaded as a worker
    @_typechecked
    def _exportFrom(mapping: Mapping[str, Iterable[str]] | None) -> None:
        exportNames = [_connectFromMain.__name__,]
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

        target['__export__'] = tuple(exportNames)
        _log(f"Started worker, providing functions: {', '.join(name for name in exportNames if name != _connectFromMain.__name__)}")

    if __name__ == '__main__':
        # If this module itself is used as a worker, it imports modules mentioned in config and exports them automatically
        __export__ = ()
        _exportFrom(config.get('exports'))
        _adaptersFrom(config.get('adapters'))
        assert __export__, __export__
        del export  # type: ignore[unreachable]
        __all__ = ()
    else:
        # If user is importing this module in a worker, they MUST call `export()` explicitly
        __all__ = (export.__name__,)  # noqa: PLE0604
        _adaptersFrom(config.get('adapters'))
        del __export__
        del _exportFrom

       ##
else:  #  MAIN THREAD
       ##

    from pyscript import workers  # pylint: disable=ungrouped-imports

    _PREFIX = "[main]"

    @_typechecked
    class Worker:
        def __init__(self, worker: JsProxy) -> None:
            self.worker = worker

    @_typechecked
    def mainSerialized(func: _CoroutineFunction) -> _CoroutineFunction:
        @wraps(func)
        @_typechecked
        async def mainSerializedWrapper(*args: Any, **kwargs: Any) -> Any:
            args = await _to_transport(args)
            assert isinstance(args, tuple), type(args)
            kwargs = await _to_transport(kwargs)
            assert isinstance(kwargs, dict), type(kwargs)
            assert isinstance(func, JsProxy), type(func)
            ret = await func(*args, **kwargs)  # type: ignore[unreachable]
            return await _from_transport(ret)
        return mainSerializedWrapper

    @_typechecked
    async def connectToWorker(workerName: str) -> Worker:
        _log(f'Looking for worker named "{workerName}"')
        worker = await workers[workerName]
        _log("Got worker, connecting")
        data = await (mainSerialized(worker._connectFromMain)(_CONNECT_REQUEST))  # noqa: SLF001  # pylint: disable=protected-access
        if not data or data[0] != _CONNECT_RESPONSE:
            _error(f"Connection to worker is misconfigured, can't continue: {type(data)}: {data!r}")
        ret = Worker(worker)  # We can't return `worker`, as it is a `JsProxy` and we can't reassign its fields, `setattr()` is not working, so we have to create a class of our own to store `wrap()`ped functions.
        for b in data[1:]:
            funcName = bytes(b).decode()  # We can't get a list of exported functions from `worker` object to `wrap()` them, so we have to pass it over in our own way.
            assert funcName != connectToWorker.__name__
            if not (func := getattr(worker, funcName, None)):
                _error(f"Function {funcName} is not exported from the worker")
            setattr(ret, funcName, mainSerialized(func))
        _log("Connected to worker")
        return ret

    assert not hasattr(globals(), '__export__'), getattr(globals(), '__export__')  # noqa: B009
    _adaptersFrom(config.get('adapters'))
    __all__ = ('Worker', 'connectToWorker')
