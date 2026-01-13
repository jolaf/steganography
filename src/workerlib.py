#
# Note: this library is intended for PyScript apps, it's mostly useless outside of browser
#
# To create a worker, you have two ways:
#
# 1.
# index.html:
# <script type="py" worker name="workerlib" src="./workerlib.py" config="./worker.toml"></script>
#
# worker.toml:
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
# from workerlib import workerCall
# ret = await workerCall('func1', *args, **kwargs)
#

# ToDo: Maybe import decorators from some another module too, the same way, and list (additional) decorators to use in a config option?
# ToDo: create translation table for types and converters to-from bytes?

# ToDo: Create a separate toml config?
# ToDo: Copy logging from main? Or import??
# ToDo: Find section [worker] in config, treat as dict, get name, other entries with keys = modules, values = functions, import them, wrap them, export them?
# ToDo: Maybe import decorators from some another module too, the same way, and list (additional) decorators to use in a config option?

from collections.abc import Awaitable, Buffer, Callable, Iterable, Mapping
from functools import wraps
from typing import Any, Never

from pyscript import RUNNING_IN_WORKER

from Steganography import imageToBytes, loadImage, Image

PREFIX = ""

CONNECT_REQUEST = b'__REQUEST__'
CONNECT_RESPONSE = b'__RESPONSE__'

def log(*args: Any) -> None:
    print(PREFIX, *args)

def error(message: str) -> Never:
    raise RuntimeError(f"{PREFIX} {message}")

def to_py(obj: Any) -> Any:
    if hasattr(obj, 'to_py'):
        return obj.to_py()
    if hasattr(obj, 'to_bytes'):
        return obj.to_bytes()
    if isinstance(obj, Iterable):
        return tuple(to_py(obj) for obj in obj)
    return obj

if RUNNING_IN_WORKER:

    from importlib import import_module

    PREFIX = "[worker]"

    log("Starting worker")

    try:  # To turn runtime typechecking on, add "beartype" to `packages` array in your `worker.toml`
        from beartype import beartype as typechecked, __version__
        log(f"Beartype v{__version__} is up and watching, remove it from worker configuration to make things faster")
    except ImportError:
        log("WARNING: beartype is not available, running fast with typing unchecked")

        def typechecked(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[no-redef]
            return func

    def serialized(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def serializedWrapper(*args: Any, **kwargs: Any) -> Any:
            assert not kwargs  # kwargs get passed to workers as last of args, of type dict
            args = tuple(to_py(arg) for arg in args)
            if args and isinstance(args[-1], dict):
                kwargs = args[-1]
                args = args[:-1]
            return func(*args, **kwargs)
        return serializedWrapper

    def images(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def imagesWrapper(*args: Any, **kwargs: Any) -> Any:
            args = tuple(loadImage(arg) if isinstance(arg, Buffer) else arg for arg in args)
            ret = func(*args, **kwargs)
            if isinstance(ret, Image):
                return imageToBytes(ret)
            if isinstance(ret, Iterable):
                return tuple(imageToBytes(r) if isinstance(r, Image) else r for r in ret)
            return ret
        return imagesWrapper

    def logged(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def loggedWrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                log(f"Calling {func.__name__}(): {args} {kwargs}")
                ret = func(*args, **kwargs)
                log(f"Returned from {func.__name__}(): {ret}")
                return ret  # noqa: TRY300
            except BaseException as ex:
                log(f"Exception at {func.__name__}: {ex}")
                raise
        return loggedWrapper

    @serialized
    def connect(data: bytes) -> bytes:
        if data == CONNECT_REQUEST:
            log("Connected to main thread, ready for requests")
            return CONNECT_RESPONSE
        error(f"Connection to main thread is misconfigured, can't continue: {type(data)}({data!r})")

    @typechecked
    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        return serialized(images(logged(typechecked(func))))

    # Must be called by the importing module to actually start providing worker service
    @typechecked
    def export(*functions: Callable[..., Any]) -> None:
        from sys import _getframe  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        target = _getframe(1).f_globals  # globals of the calling module
        target[connect.__name__] = connect
        exportNames = [connect.__name__,]

        for func in functions:
            target[func.__name__] = wrap(func)
            exportNames.append(func.__name__)

        target['__export__'] = tuple(exportNames)
        log(f"Started worker, providing functions: {', '.join(exportNames)}")

    # Gets called automatically if `[exports]` section is found in config
    @typechecked
    def exportFromMapping(mapping: Mapping[str, Iterable[str]] | None) -> None:
        exportNames = [connect.__name__,]
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

        target['__export__'] = tuple(exportNames)
        log(f"Started worker, providing functions: {', '.join(exportNames)}")

    if __name__ == '__main__':
        from pyscript import config  # pylint: disable=ungrouped-imports
        exportFromMapping(config.get('exports'))
    else:
        __all__ = (export.__name__,)  # noqa: PLE0604

else:  # Main thread

    from asyncio import run

    from pyscript import workers

    PREFIX = "[main]"

    try:  # To turn runtime typechecking on, add "beartype" to `packages` array in your `main.toml`
        from beartype import beartype as typechecked  # pylint: disable=ungrouped-imports
    except ImportError:
        def typechecked(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:  # type: ignore[no-redef]
            return func

    def images(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:  # type: ignore[misc]
        @wraps(func)
        async def imagesWrapper(*args: Any, **kwargs: Any) -> Any:
            args = tuple(imageToBytes(arg) if isinstance(arg, Image) else arg for arg in args)
            ret = await func(*args, **kwargs)
            if isinstance(ret, Buffer):
                return loadImage(ret)
            if isinstance(ret, Iterable):
                return tuple(loadImage(r) if isinstance(r, Buffer) else r for r in ret)
            return ret
        return imagesWrapper

    def serialized(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:  # type: ignore[misc]
        @wraps(func)
        async def serializedWrapper(*args: Any, **kwargs: Any) -> Any:
            return to_py(await func(*args, **kwargs))
        return serializedWrapper

    async def connect(workerName: str) -> Any:
        print(f'{PREFIX} Looking for worker named "{workerName}"')
        worker = await workers[workerName]  # pylint: disable=redefined-outer-name
        log("Got worker, connecting")
        data = await serialized(worker.connect)(CONNECT_REQUEST)
        if data == CONNECT_RESPONSE:  # pylint: disable=consider-using-assignment-expr
            log("Connected to worker")
            return worker
        error(f"Connection to worker is misconfigured, can't continue: {type(data)}({data!r})")

    # Should be called by the importing module to call a worker function
    async def workerCall(funcName: str, *args: Any, **kwargs: Any) -> Any:
        return await images(serialized(typechecked(getattr(worker, funcName))))(*args, **kwargs)

    worker = run(connect('workerlib'))

    __all__ = (workerCall.__name__,)  # noqa: PLE0604
