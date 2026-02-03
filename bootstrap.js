"use strict";

/* jshint browser: true */
/* jshint esversion: 6 */
/* global console */

window.bootstrapLoadedOK = true;  // Used to check if local scripts can be loaded

const timeFormatter = new Intl.DateTimeFormat('default', {
      hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
});

function timestamp() {
    return timeFormatter.format(new Date());
}

const logElement = document.getElementById('log');

function log(message, prefix = 'javascript') {
    message = "[" + prefix + "] " + message;
    console.log(message);
    logElement.append(timestamp() + " " + message + "\n");
}
window.log = log;

log("Bootstrapping");

function pyscriptLog(message) {
    log(message, 'pyscript');
}

addEventListener('py:progress', ({ detail }) => {
    pyscriptLog(detail);
    window.pyodidePresent = true; // Used to check if Pyodide is available
});

addEventListener('py:ready', () => {
    pyscriptLog("Starting PyScript");
    window.pyscriptPresent = true; // Used to check if PyScript is available
});

addEventListener('py:done', () => {
    pyscriptLog("Started PyScript");
    window.pyscriptDone = true; // Used to check if PyScript app has started
});

const consoleWarn = console.warn;
console.warn = function (...args) {
    if (args.length > 0 && /^Pyodide \S+ might not support (beartype|coolname)/.test(args[0])) {
        return;  // Avoiding annoying warnings about well-behaving packages (main thread only, unfortunately)
    }
    consoleWarn(...args);
};

let locationName;
if (location.protocol === 'file:') {
    locationName = 'files';
} else if (location.hostname === 'localhost') {
    locationName = location.port.startsWith('633') ? 'PyCharm' : 'localhost';
} else if (location.hostname.endsWith('github.io')) {
    locationName = 'GitHub Pages';
} else {
    locationName = location.hostname;
}
document.getElementById('location').textContent = '/' + locationName;

setTimeout(() => {
    if (typeof pyodidePresent === 'undefined') {
        const logElement = document.getElementById('log');
        logElement.textContent = "ERROR: Pyodide failed to load, do you have Internet access? Press F12 and check the browser console for details.";
        logElement.classList.add('error');
    } else {
        setTimeout(() => {
            if (typeof pyscriptPresent === 'undefined') {
                const logElement = document.getElementById('log');
                logElement.textContent = "ERROR: PyScript failed to load, do you have Internet access? Press F12 and check the browser console for details.";
                logElement.classList.add('error');
            }
        }, 30000); // Wait 30 more seconds before deciding that PyScript failed to load
    }
}, 30000); // Wait 30 seconds before deciding that Pyodide failed to load
