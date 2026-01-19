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

# ToDo: [adapters]
# "module" = ["type", "encoder", "decoder" ]
# # All three would be imported from the specified module, without eval()!
# # type - name of a class, will be used with isinstance() check to identify that your encoder has to be employed
# # encoder - name of a function or coroutine that converts an instance of your class to some type that's suitable for JS structural clone (link).
# # decoder - will be used on a JsProxy object, should identify if this is an encoder()-generated representation of type and return its instance, or return None, if that JsProxy object is not a representation of type.

from collections.abc import Coroutine, Buffer, Callable, Iterable, Iterator, Mapping, Sequence
from functools import wraps
from importlib import import_module
from inspect import isfunction, iscoroutinefunction, signature
from itertools import chain
from sys import _getframe
from types import ModuleType
from typing import cast, Any, Final, NoReturn

from pyscript import config, RUNNING_IN_WORKER
from pyscript.ffi import to_js  # pylint: disable=import-error, no-name-in-module

try:
    from pyodide.ffi import JsProxy
except ImportError:
    type JsProxy = Any  # type: ignore[no-redef]

# We name everything starting with underscore to avoid potential conflicts with exported user functions
type _Coroutine = Coroutine[None, None, Any]
type _CoroutineFunction = Callable[..., _Coroutine]
type _FunctionOrCoroutine = Callable[..., Any | _Coroutine]
type _Adapter = tuple[type, Callable[[Any], Any | _Coroutine] | None, Callable[[Any], Any | _Coroutine] | None]
_TransportSafe = int | float | bool | str | Buffer | Iterable | Sequence | Mapping | None  # type: ignore[type-arg]

try:  # To turn runtime typechecking on, add "beartype" to `packages` array in your `workerlib.toml`
    from beartype import beartype as _typechecked
except ImportError:
    def _typechecked(func: _FunctionOrCoroutine) -> _FunctionOrCoroutine:  # type: ignore[no-redef]
        return func

_PREFIX = ''

_CONNECT_REQUEST: Final[str] = '__REQUEST__'
_CONNECT_RESPONSE: Final[str] = '__RESPONSE__'

_ADAPTER_PREFIX: Final[str] = '_workerlib_'

__EXPORT__: Final[str] = '__export__'

_ADAPTERS_SECTION: Final[str] = 'adapters'
_EXPORTS_SECTION: Final[str] = 'exports'

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
def _adaptersFromSequence(module: ModuleType, names: Sequence[str | Sequence[str]]) -> Iterator[_Adapter]:
    if not isinstance(names, Sequence):
        _error(f"""Bad adapter tuple type {type(names)} for module {module.__name__}""")
    if not names:
        _error(f"""Empty adapter tuple for module {module.__name__}""")
    if isinstance(names[0], str):
        if len(names) != 3:
            _error(f"""Bad adapter settings for module {module.__name__}, must be '["className", "encoderFunction", "decoderFunction"]', got {names!r}""")
        _log(f"Importing from module {module.__name__}: {', '.join(cast(str, name) for name in names if name != 'None')}")
        (cls, encoder, decoder) = (None if name == 'None' else getattr(module, cast(str, name)) for name in names)
        if not isinstance(cls, type):
            _error(f"Bad adapter class {names[0]} for module {module.__name__}, must be type, got {type(cls)}")
        if encoder is not None and not isfunction(encoder) and not iscoroutinefunction(encoder):
            _error(f'Bad adapter encoder {names[1]} for module {module.__name__}, must be function or coroutine or "None", got {type(encoder)}')
        if decoder is not None and not isfunction(decoder) and not iscoroutinefunction(decoder):
            _error(f'Bad adapter encoder {names[2]} for module {module.__name__}, must be function or coroutine or "None", got {type(decoder)}')
        _log(f"Adapter created: {decoder} => {cls} => {encoder}")
        yield (cls, encoder, decoder)
    else:
        for subSequence in names:
            yield from _adaptersFromSequence(module, subSequence)

@_typechecked
def _adaptersFrom(mapping: Mapping[str, Sequence[str | Sequence[str]]] | None) -> None:
    if not mapping:
        return
    adapters: list[_Adapter] = []
    for (moduleName, names) in mapping.items():
        if (module := globals().get(moduleName)) is None:
            _log(f"Importing module {moduleName}")
            module = import_module(moduleName)
        adapters.extend(_adaptersFromSequence(module, names))
    global __adapters__  # noqa: PLW0603  # pylint: disable=global-statement
    __adapters__ = tuple(adapters)

@_typechecked
async def _to_js(obj: Any) -> Any:
    for (cls, encoder, _decoder) in __adapters__:
        if isinstance(obj, cls):
            if encoder:  # noqa: SIM108
                value = await encoder(obj) if iscoroutinefunction(encoder) else encoder(obj)
            else:
                value = obj  # Trying to pass object as is, hoping it would work, like for Enums
            return to_js((_ADAPTER_PREFIX + cls.__name__, value))  # Encoded class name is NOT the name of type of object being encoded, but name of the adapter to use to decode the object on the other side
    if not isinstance(obj, _TransportSafe):
        _error(f"No adapter found for class {type(obj)}, and transport layer (JavaScript structured clone) would not accept it as is, see https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Structured_clone_algorithm")
    if isinstance(obj, str | Buffer):
        return to_js(obj)
    if isinstance(obj, Mapping):
        return to_js({await _to_js(k): await _to_js(v) for (k, v) in obj.items()})
    if isinstance(obj, Iterable):
        return to_js(tuple([await _to_js(v) for v in obj]))  # pylint: disable=consider-using-generator
    return to_js(obj)

@_typechecked
async def _to_py(obj: Any) -> Any:
    if hasattr(obj, 'to_py'):
        return await _to_py(obj.to_py())
    if isinstance(obj, Mapping):
        obj = {await _to_py(k): await _to_py(v) for (k, v) in obj.items()}
    elif not isinstance(obj, str | Buffer) and isinstance(obj, Iterable):
        obj = tuple([await _to_py(v) for v in obj])  # pylint: disable=consider-using-generator
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
        if className == cls.__name__:  # ToDo: Can't we import class by full name and use `isinstance()` ?
            if decoder is None:
                return cls(value)  # e.g. Enum
            return await decoder(value) if iscoroutinefunction(decoder) else decoder(value)
    _error(f"No adapter found to decode class {className}")

                       ##
if RUNNING_IN_WORKER:  ##
                       ##

    _PREFIX = "[worker]"

    _log("Starting worker, sync_main_only =", config.get('sync_main_only', False))

    try:
        from beartype import __version__ as _version
        _log(f"Beartype v{_version} is up and watching, remove it from worker configuration to make things faster")
    except ImportError:
        _log("WARNING: beartype is not available, running fast with typing unchecked")

    try:
        assert str()  # noqa: UP018
        _log("Assertions are DISABLED")
    except AssertionError:
        _log("Assertions are enabled")

    @_typechecked
    def _workerSerialized(func: _FunctionOrCoroutine) -> _CoroutineFunction:
        @wraps(func)
        @_typechecked
        async def workerSerializedWrapper(*args: Any, **kwargs: Any) -> Any:
            assert not kwargs  # kwargs get passed to workers as last of args, of type `dict`
            args = await _to_py(args)
            if any(param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY, param.VAR_KEYWORD)
                        for param in signature(func).parameters.values()) \
                    and args and isinstance(args[-1], dict):  # If `func` accepts keyword arguments, extract them from last of `args`
                kwargs = args[-1]
                args = args[:-1]
            ret = await func(*args, **kwargs) if iscoroutinefunction(func) else func(*args, **kwargs)
            return await _to_js(ret)
        return workerSerializedWrapper

    @_typechecked
    def _workerLogged(func: _FunctionOrCoroutine) -> _CoroutineFunction:
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

    @_typechecked
    def _wrap(func: _FunctionOrCoroutine) -> _CoroutineFunction:
        return _workerSerialized(_workerLogged(_typechecked(func)))

    @_workerSerialized
    @_typechecked
    def _connectFromMain(data: str) -> tuple[str, ...]:
        if data == _CONNECT_REQUEST:
            _log("Connected to main thread, ready for requests")
            assert __export__, __export__
            return tuple(chain((_CONNECT_RESPONSE,), (name for name in __export__ if name != _connectFromMain.__name__)))
        _error(f"Connection to main thread is misconfigured, can't continue: {type(data)}({data!r})")

    # Must be called by the importing module to actually start providing worker service
    @_typechecked
    def export(*functions: _FunctionOrCoroutine) -> None:
        target = _getframe(1).f_globals  # globals of the calling module
        target[_connectFromMain.__name__] = _connectFromMain
        exportNames: list[str] = []
        if _connectFromMain.__name__ not in target.get(__EXPORT__, ()):
            exportNames.append(_connectFromMain.__name__)

        for func in functions:
            target[func.__name__] = _wrap(func)
            exportNames.append(func.__name__)

        exportNamesTuple = tuple(chain(target.get(__EXPORT__, ()), exportNames))
        target[__EXPORT__] = exportNamesTuple
        globals()[__EXPORT__] = exportNamesTuple  # This is only needed to make `_connectFromMain()` code universal for both `export()` and `_exportFrom()`
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

        target[__EXPORT__] = tuple(exportNames)
        _log(f"Started worker, providing functions: {', '.join(name for name in exportNames if name != _connectFromMain.__name__)}")

    _adaptersFrom(config.get(_ADAPTERS_SECTION))

    if __name__ == '__main__':
        # If this module itself is used as a worker, it imports modules mentioned in config and exports them automatically
        __export__ = ()
        _exportFrom(config.get(_EXPORTS_SECTION))
        assert __export__, __export__
        del export  # type: ignore[unreachable]
        __all__ = ()
    else:
        # If user is importing this module in a worker, they MUST call `export()` explicitly
        del __export__
        del _exportFrom
        __all__ = (export.__name__,)  # noqa: PLE0604

       ##
else:  ##  MAIN THREAD
       ##

    from pyscript import workers  # pylint: disable=ungrouped-imports

    _PREFIX = "[main]"

    @_typechecked
    class Worker:
        def __init__(self, worker: JsProxy) -> None:
            self.worker = worker

    @_typechecked
    def _mainSerialized(func: _CoroutineFunction, looksLike: _FunctionOrCoroutine | str | None = None) -> _CoroutineFunction:
        @_typechecked
        async def mainSerializedWrapper(*args: Any, **kwargs: Any) -> Any:
            assert isinstance(func, JsProxy), type(func)
            args = tuple([await _to_js(arg) for arg in args])  # type: ignore[unreachable]  # pylint: disable=consider-using-generator
            kwargs = {key: await _to_js(value) for (key, value) in kwargs.items()}
            return await _to_py(await func(*args, **kwargs))
        if looksLike and (isfunction(looksLike) or iscoroutinefunction(looksLike)):
            return wraps(looksLike)(mainSerializedWrapper)
        ret = wraps(func)(mainSerializedWrapper)
        if looksLike and isinstance(looksLike, str):
            ret.__name__ = looksLike
        return ret

    @_typechecked
    async def connectToWorker(workerName: str) -> Worker:
        _log(f'Looking for worker named "{workerName}"')
        worker = await workers[workerName]
        _log("Got worker, connecting")
        data = await _mainSerialized(worker._connectFromMain, '_connectFromMain')(_CONNECT_REQUEST)  # noqa: SLF001  # pylint: disable=protected-access
        if not data or data[0] != _CONNECT_RESPONSE:
            _error(f"Connection to worker is misconfigured, can't continue: {type(data)}: {data!r}")
        ret = Worker(worker)  # We can't return `worker`, as it is a `JsProxy` and we can't reassign its fields, `setattr()` is not working, so we have to create a class of our own to store `wrap()`ped functions.
        for funcName in data[1:]:  # We can't get a list of exported functions from `worker` object to `wrap()` them, so we have to pass it over in our own way.
            assert funcName != connectToWorker.__name__
            if not (func := getattr(worker, funcName, None)):
                _error(f"Function {funcName} is not exported from the worker")
            setattr(ret, funcName, _mainSerialized(func, funcName))
        _log("Connected to worker")
        return ret

    assert not hasattr(globals(), __EXPORT__), getattr(globals(), __EXPORT__)
    _adaptersFrom(config.get(_ADAPTERS_SECTION))
    __all__ = (Worker.__name__, connectToWorker.__name__)  # noqa: PLE0604
