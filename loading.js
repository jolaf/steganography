window.localLoadingIsWorking= true;  // Used to check if local scripts can be loaded

const formatter = new Intl.DateTimeFormat('default', {
    hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
});

const target = document.getElementById('log');

console.log("[javascript] Started JavaScript");
target.append(formatter.format(new Date()), " [javascript] Started JavaScript\n");

function pyscriptLog(message) {
    console.log("[pyscript] " + message);
    target.append("" + formatter.format(new Date()) + " [pyscript] " + message + '\n');
}

addEventListener('py:progress', ({ detail }) => {
    pyscriptLog(detail);
});

addEventListener('py:ready', () => {
    pyscriptLog("Starting PyScript");
});

addEventListener('py:done', () => {
    pyscriptLog("Started PyScript");
});

const consoleWarn = console.warn
console.warn = function (...args) {
    if (args.length > 0 && args[0].startsWith('Pyodide') && args[0].includes('might not support')) {
        return;  // Avoiding stupid warnings during loading
    }
    consoleWarn(...args);
}
