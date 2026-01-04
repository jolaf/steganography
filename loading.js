const target = document.getElementById('log');

console.log("[javascript] Started JavaScript");
target.append("Started JavaScript\n");

function log(message) {
    console.log("[pyscript] " + message);
    target.append(message + '\n');
}

addEventListener('py:progress', ({ detail }) => {
    log(detail);
});

addEventListener('py:ready', () => {
    log("Starting PyScript");
});

addEventListener('py:done', () => {
    log("Started PyScript");
});

const consoleWarn = console.warn
console.warn = function (...args) {
    if (args.length > 0 && args[0].startsWith('Pyodide') && args[0].includes('might not support')) {
        return;  // Avoiding stupid warnings during loading
    }
    consoleWarn(...args);
}
