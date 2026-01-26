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
from inspect import isclass, isfunction, iscoroutinefunction, signature
from itertools import chain
from re import findall
from sys import platform, version as _pythonVersion, _getframe
from time import time
from types import ModuleType
from typing import cast, Any, Final, NoReturn, TypeAlias

try:
    from os import process_cpu_count  # type: ignore[attr-defined]
    _cpus: Any = process_cpu_count()
except ImportError:
    try:
        from os import cpu_count
        _cpus = cpu_count()
    except ImportError:
        _cpus = "UNKNOWN"

try:
    from sys import _emscripten_info  # type: ignore[attr-defined]  # pylint: disable=ungrouped-imports
    assert platform == 'emscripten'
    _emscriptenVersion: str | None = '.'.join(str(v) for v in _emscripten_info.emscripten_version)
    _runtime = _emscripten_info.runtime
    _pthreads = _emscripten_info.pthreads
    _sharedMemory = _emscripten_info.shared_memory
except ImportError:
    _emscriptenVersion = _runtime = _sharedMemory = None
    try:
        from os import sysconf  # pylint: disable=ungrouped-imports
        _pthreads = sysconf('SC_THREADS') > 0
    except (ImportError, ValueError, AttributeError):
        _pthreads = False

from pyscript import config, RUNNING_IN_WORKER
from pyscript.web import page  # pylint: disable=import-error, no-name-in-module

try:  # Try to identify PyScript version
    from pyscript import version as _pyscriptVersion  # type: ignore[attr-defined]
except ImportError:
    try:
        from pyscript import __version__ as _pyscriptVersion  # type: ignore[attr-defined]
    except ImportError:
        try:
            coreURL = next(element.src for element in page.find('script') if element.src.endswith('core.js'))
            _pyscriptVersion = next(word for word in coreURL.split('/') if findall(r'\d', word))
        except Exception:  # noqa: BLE001
            _pyscriptVersion = "UNKNOWN"

try:
    from pyodide_js import version as _pyodideVersion  # type: ignore[import-not-found]
except ImportError:
    _pyodideVersion = "UNKNOWN"

try:
    from pyodide.ffi import JsProxy
except ImportError:
    type JsProxy = Any  # type: ignore[no-redef]

# We name everything starting with underscore to minimize the chance of a conflict with exported user functions
type _Coroutine[T] = Coroutine[None, None, T]
type _CoroutineFunction[T] = Callable[..., _Coroutine[T]]
type _FunctionOrCoroutine[T] = Callable[..., T | _Coroutine[T]]
type _Adapter[T] = tuple[type[T], Callable[[T], Any | _Coroutine[Any]] | None, Callable[[Any], T | _Coroutine[T]] | None]
type _Timed[T] = tuple[str, tuple[float, T]]
_TransportSafe: TypeAlias = int | float | bool | str | Buffer | Iterable | Sequence | Mapping | None  # type: ignore[type-arg]  # Using `type` or adding `[Any]` breaks `isinstance()`  # noqa: UP040

try:  # To turn runtime typechecking on, add "beartype" to `packages` array in your `workerlib.toml`
    from beartype import beartype as typechecked, __version__ as _beartypeVersion
    from beartype.roar import BeartypeException
except ImportError:
    _beartypeVersion = None  # type: ignore[assignment]

    def typechecked(func: _FunctionOrCoroutine[Any]) -> _FunctionOrCoroutine[Any]:  # type: ignore[no-redef]
        return func

_PREFIX = ''

_CONNECT_REQUEST: Final[str] = '__REQUEST__'
_CONNECT_RESPONSE: Final[str] = '__RESPONSE__'

_ADAPTER_PREFIX: Final[str] = '_workerlib_'
_START_TIME: Final[str] = _ADAPTER_PREFIX + 'startTime'

__EXPORT__: Final[str] = '__export__'

_ADAPTERS_SECTION: Final[str] = 'adapters'
_EXPORTS_SECTION: Final[str] = 'exports'

__all__: Sequence[str] = (  # Will be reduced below
    'Worker',
    '__export__',
    '__info__',
    '__short_info__',
    '_elapsedTime',
    'connectToWorker',
    'export',
    'typechecked',
)

__export__: Sequence[str]
__adapters__: Sequence[_Adapter[Any]] = ()

@typechecked
def _log(*args: Any) -> None:
    print(_PREFIX, *args)

@typechecked
def _error(message: str) -> NoReturn:
    raise RuntimeError(f"{_PREFIX} {message}")

@typechecked
def _elapsedTime(startTime: float) -> str:
    dt = time() - startTime
    return f"{round(dt)}s" if dt >= 1 else f"{round(dt * 1000)}ms"

@typechecked
def _importModule(moduleName: str) -> ModuleType:
    if (module := globals().get(moduleName)) is None:
        _log("Importing module", moduleName)
        globals()[moduleName] = module = import_module(moduleName)
    return module

@typechecked
def _importFromModule(module: str | ModuleType, qualNames: str | Iterable[str]) -> Iterator[Any]:
    if isinstance(module, str):
        module = _importModule(module)
    if isinstance(qualNames, str):
        qualNames = (qualNames,)
    builtins = cast(dict[str, Any], __builtins__)
    if not isinstance(builtins, dict):
        builtins = builtins.__dict__  # type: ignore[unreachable]
    _log(f"Importing from module {module.__name__}: {', '.join(name for name in qualNames if name not in builtins)}")
    for qualName in qualNames:
        try:
            yield builtins[qualName]
            continue
        except KeyError:
            obj = module
            for name in qualName.split('.'):
                obj = getattr(obj, name)
            yield obj

@typechecked
def _adaptersFromSequence(module: ModuleType, names: Sequence[str | Sequence[str]], allowSubSequences: bool = True) -> Iterator[_Adapter[Any]]:
    if not isinstance(names, Sequence):
        _error(f"""Bad adapter tuple type {type(names)} for module {module.__name__}""")
    if not names:
        _error(f"""Empty adapter tuple for module {module.__name__}""")
    if isinstance(names[0], str):
        if len(names) != 3:
            _error(f"""Bad adapter settings for module {module.__name__}, must be '["className", "encoderFunction", "decoderFunction"]', got {names!r}""")
        (cls, encoder, decoder) = _importFromModule(module, cast(Sequence[str], names))
        if not isinstance(cls, type):
            _error(f"Bad adapter class {names[0]} for module {module.__name__}, must be type, got {type(cls)}")
        if encoder is not None and not isfunction(encoder) and not iscoroutinefunction(encoder) and not isclass(encoder):
            _error(f'Bad adapter encoder {names[1]} for module {module.__name__}, must be function or coroutine or class or "None", got {type(encoder)}')
        if decoder is not None and not isfunction(decoder) and not iscoroutinefunction(decoder):
            _error(f'Bad adapter encoder {names[2]} for module {module.__name__}, must be function or coroutine or "None", got {type(decoder)}')
        _log(f"Adapter created: {decoder} => {cls} => {encoder}")
        yield (cls, encoder, decoder)
    elif allowSubSequences:
        for subSequence in names:
            yield from _adaptersFromSequence(module, subSequence, allowSubSequences = False)
    else:
        _error("""Adapter specification should be either [strings] or [[strings], ...], third level of inclusion is not needed'""")

@typechecked
def _adaptersFrom(mapping: Mapping[str, Sequence[str | Sequence[str]]] | None) -> Sequence[_Adapter[Any]]:
    if not mapping:
        return ()
    adapters: list[_Adapter[Any]] = []
    for (moduleName, names) in mapping.items():
        module = _importModule(moduleName)
        adapters.extend(_adaptersFromSequence(module, names))
    return tuple(adapters)

@typechecked
async def _to_js(obj: Any) -> Any:
    for (cls, encoder, _decoder) in __adapters__:
        if isinstance(obj, cls):
            if encoder:  # noqa: SIM108
                value = await encoder(obj) if iscoroutinefunction(encoder) else encoder(obj)
            else:
                value = obj  # Trying to pass object as is, hoping it would work, like for Enums
            return (_ADAPTER_PREFIX + cls.__name__, value)  # Encoded class name is NOT the name of type of object being encoded, but the name of the adapter to use to decode the object on the other side
    if not isinstance(obj, _TransportSafe):
        _error(f"No adapter found for class {type(obj)}, and transport layer (JavaScript structured clone) would not accept it as is, see https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Structured_clone_algorithm")
    if isinstance(obj, str | Buffer):
        return obj
    if isinstance(obj, Mapping):
        return {await _to_js(k): await _to_js(v) for (k, v) in obj.items()}
    if isinstance(obj, Iterable):
        return tuple([await _to_js(v) for v in obj])  # pylint: disable=consider-using-generator
    return obj

@typechecked
async def _to_py(obj: Any) -> Any:
    if hasattr(obj, 'to_py'):
        return await _to_py(obj.to_py())
    if isinstance(obj, str | Buffer):
        return obj
    if isinstance(obj, Mapping):
        return {await _to_py(k): await _to_py(v) for (k, v) in obj.items()}
    if isinstance(obj, Iterable):
        obj = tuple([await _to_py(v) for v in obj])  # pylint: disable=consider-using-generator
    if not isinstance(obj, tuple) or len(obj) != 2:
        return obj
    (name, value) = obj
    if not isinstance(name, str):
        return obj
    if name == _START_TIME:
        return obj
    tokens = name.split(_ADAPTER_PREFIX)
    if len(tokens) != 2 or tokens[0]:
        return obj
    className = tokens[1]
    for (cls, _encoder, decoder) in __adapters__:
        if cls.__name__ == className:
            if decoder is None:
                return cls(value)  # e.g. Enum
            return await decoder(value) if iscoroutinefunction(decoder) else decoder(value)
    _error(f"No adapter found to decode class {className}")

@typechecked
def _info() -> Sequence[str]:
    ret: list[str] = []
    ret.append(f"PyScript {_pyscriptVersion}")
    ret.append(f"Pyodide {_pyodideVersion}")

    if platform == 'emscripten':
        ret.append(f"Emscripten {_emscriptenVersion}")
        ret.append(f"Runtime: {_runtime}")
        ret.append(f"CPUs: {_cpus}  pthreads: {_pthreads}  SharedMemory: {_sharedMemory}")
    else:
        ret.append(f"Platform: {platform}")
        ret.append(f"CPUs: {_cpus}  pthreads: {_pthreads}")

    ret.append(f"Python {_pythonVersion}")

    if _beartypeVersion:
        try:
            @typechecked
            def test() -> int:
                return 'notInt'  # type: ignore[return-value]
            test()
            raise RuntimeError("Beartype v" + _beartypeVersion + " is not operating properly")
        except BeartypeException:
            ret.append(f"Beartype {_beartypeVersion} is up and watching, remove it from PyScript configuration to make things faster")
    else:
        ret.append("Runtime type checking is off")

    try:
        assert str()  # noqa: UP018
        ret.append("Assertions are DISABLED")
    except AssertionError:
        ret.append("Assertions are enabled")

    return tuple(ret)

__info__ = _info()
__short_info__ = f'PyScript {_pyscriptVersion} / Pyodide {_pyodideVersion} / Python {_pythonVersion}'

                       ##
if RUNNING_IN_WORKER:  ##
                       ##

    _PREFIX = "[worker]"

    _log("Starting worker, sync_main_only =", config.get('sync_main_only', False))

    for info in __info__:
        _log(info)

    @typechecked
    def _workerSerialized[T](func: _FunctionOrCoroutine[T]) -> _CoroutineFunction[Any]:  # pylint: disable=redefined-outer-name
        @wraps(func)
        @typechecked
        async def workerSerializedWrapper(*args: Any, **kwargs: Any) -> Any:
            @typechecked
            def hasKwargs(func: _FunctionOrCoroutine[T], args: Any) -> bool:  # ToDo: Remove it as it's always true?
                if not args:
                    return False
                lastArg = args[-1]
                if not isinstance(lastArg, dict):
                    return False
                if _START_TIME in lastArg:
                    return True
                if any(not isinstance(key, str) for key in lastArg):
                    return False
                funcParams = signature(func).parameters
                if any(param.kind == param.VAR_KEYWORD for param in funcParams.values()):
                    return True  # If `func` has a `**kwargs` argument, any str-keyed dictionary would fit
                paramNames = {paramName for (paramName, param) in funcParams.items() if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)}
                return all(argName in paramNames for argName in lastArg)  # If `func` accepts keyword arguments, make sure all keys in `args[-1]` are among them

            assert not kwargs  # `kwargs` get passed to workers as the last of `args`, of type `dict`
            args = await _to_py(args)
            if hasKwargs(func, args):
                (args, kwargs) = (args[:-1], args[-1])  # If `func` accepts keyword arguments, extract them from the last of `args`
            ret = await func(*args, **kwargs) if iscoroutinefunction(func) else func(*args, **kwargs)
            return await _to_js(ret)
        return workerSerializedWrapper

    @typechecked
    def _workerLogged[T](func: _FunctionOrCoroutine[T]) -> _CoroutineFunction[_Timed[T]]:  # pylint: disable=redefined-outer-name
        @wraps(func)
        @typechecked
        async def workerLoggedWrapper(*args: Any, **kwargs: Any) -> _Timed[T]:
            try:
                elapsed = _elapsedTime(startTime) if (startTime := kwargs.pop(_START_TIME, None)) else ''
                startTime = time()
                if iscoroutinefunction(func):
                    _log(f"{f"Passing arguments {elapsed}, awaiting" if elapsed else "Awaiting"} {func.__name__}(): {args} {kwargs}")
                    ret: T = await func(*args, **kwargs)
                else:
                    assert isfunction(func), func
                    _log(f"{f"Passing arguments {elapsed}, calling" if elapsed else "Calling"} {func.__name__}(): {args} {kwargs}")
                    ret = func(*args, **kwargs)
                _log(f"Returned in {_elapsedTime(startTime)} from {func.__name__}(): {ret}")
                return (_START_TIME, (time(), ret))  # Starting calculating time it would take to return the data to main thread
            except BaseException as ex:
                _log(f"Exception at {func.__name__}: {ex}")
                raise
        return workerLoggedWrapper

    @typechecked
    def _wrap[T](func: _FunctionOrCoroutine[T]) -> _CoroutineFunction[Any]:  # pylint: disable=redefined-outer-name
        return _workerSerialized(_workerLogged(typechecked(func)))  # type: ignore[arg-type]  # It looks there's some bug in mypy in this matter

    @_workerSerialized
    @typechecked
    def _connectFromMain(data: str, **_kwargs: Any) -> tuple[str, ...]:
        if data == _CONNECT_REQUEST:
            _log("Connected to main thread, ready for requests")
            assert __export__, __export__
            return tuple(chain((_CONNECT_RESPONSE,), (name for name in __export__ if name != _connectFromMain.__name__)))
        _error(f"Connection to main thread is misconfigured, can't continue: {type(data)}({data!r})")

    # Must be called by the importing module to actually start providing worker service
    @typechecked
    def export(*functions: _FunctionOrCoroutine[Any]) -> None:
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
        _log("Started worker, providing functions:", ', '.join(name for name in exportNamesTuple if name != _connectFromMain.__name__))

    # Gets called automatically if this module itself is loaded as a worker
    @typechecked
    def _exportFrom(mapping: Mapping[str, Iterable[str]] | None) -> None:
        exportNames = [_connectFromMain.__name__,]
        target = globals()

        if mapping:
            for (moduleName, funcNames) in mapping.items():
                for (funcName, func) in zip(funcNames, _importFromModule(moduleName, funcNames), strict = True):
                    target[funcName] = _wrap(func)
                    exportNames.append(funcName)
        else:
            _log("WARNING: no functions found to export, check `[exports]` section in the config")

        target[__EXPORT__] = tuple(exportNames)
        _log("Started worker, providing functions:", ', '.join(name for name in exportNames if name != _connectFromMain.__name__))

    __adapters__ = _adaptersFrom(config.get(_ADAPTERS_SECTION))

    if __name__ == '__main__':
        # If this module itself is used as a worker, it imports modules mentioned in config and exports them automatically
        del export
        __all__ = ()
        __export__ = ()
        _exportFrom(config.get(_EXPORTS_SECTION))
        assert __export__, __export__
    else:
        # If user is importing this module in a worker, they MUST call `export()` explicitly
        del _exportFrom
        __all__ = ('__info__', '__short_info__', '_elapsedTime', 'export', 'typechecked')

       ##
else:  ##  MAIN THREAD
       ##

    from pyscript import workers  # pylint: disable=ungrouped-imports

    _PREFIX = "[main]"

    @typechecked
    class Worker:
        def __init__(self, worker: JsProxy) -> None:
            self.worker = worker

    @typechecked
    def _mainSerialized[T](func: JsProxy, looksLike: _FunctionOrCoroutine[T] | str | None = None) -> _CoroutineFunction[T]:  # pylint: disable=redefined-outer-name

        @typechecked
        async def mainSerializedWrapper(*args: Any, **kwargs: Any) -> T:
            assert isinstance(func, JsProxy), type(func)
            kwargs[_START_TIME] = time()
            args = await _to_js(args)
            kwargs = await _to_js(kwargs)
            ret = await cast(_CoroutineFunction[T], func)(*args, kwargs)  # Passing `kwargs` as positional arguments because `**kwargs` don't get serialized properly
            ret = await _to_py(ret)
            if isinstance(ret, tuple) and ret and ret[0] == _START_TIME:  # ToDo: Rewrite it to dict
                (startTime, ret) = ret[1]
                _log(f"Passing return value {_elapsedTime(startTime)}")
            return ret

        if looksLike and (isfunction(looksLike) or iscoroutinefunction(looksLike)):
            return wraps(looksLike)(mainSerializedWrapper)
        ret: _CoroutineFunction[T] = wraps(cast(_CoroutineFunction[T], func))(mainSerializedWrapper)
        if looksLike and isinstance(looksLike, str):
            ret.__name__ = looksLike

        return ret

    @typechecked
    async def connectToWorker(workerName: str | None = None) -> Worker:
        global __adapters__  # noqa: PLW0603  # pylint: disable=global-statement
        if not __adapters__:
            __adapters__ = _adaptersFrom(config.get(_ADAPTERS_SECTION))

        if not workerName:
            if not (names := tuple(element.getAttribute('name') for element in page.find('script[type="py"][worker]'))):
                _error("Could not find worker name in DOM")
            if len(names) > 1:
                _error(f"Found the following worker names in DOM: ({', '.join(names)}), which one to connect to?")
            workerName = names[0]

        _log(f'Looking for worker named "{workerName}"')
        worker = await workers[workerName]
        _log("Got worker, connecting")
        data: Sequence[str] = await _mainSerialized(worker._connectFromMain, '_connectFromMain')(_CONNECT_REQUEST)  # noqa: SLF001  # pylint: disable=protected-access
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
    __all__ = ('Worker', '__info__', '__short_info__', '_elapsedTime', 'connectToWorker', 'typechecked')
