"use strict";

/* global console */
/* jshint browser: true */
/* jshint esversion: 6 */

window.localLoadingIsWorking = true;  // Used to check if local scripts can be loaded

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

const message = "[javascript] Started JavaScript";
console.log(message);
logElement.append(timestamp() + " " + message + "\n");

function pyscriptLog(message) {
    message = "[pyscript] " + message;
    console.log(message);
    logElement.append(timestamp() + " " + message + "\n");
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

const consoleWarn = console.warn;
console.warn = function (...args) {
    if (args.length > 0 && args[0].startsWith('Pyodide') && args[0].includes('might not support')) {
        return;  // Avoiding stupid useless warnings during PyScript booting
    }
    consoleWarn(...args);
};

let locationName;
if (location.protocol === 'file:') {
    locationName = 'files';
} else if (location.hostname === 'localhost') {
    locationName = (location.port === '63342') ? 'PyCharm' : 'localhost';
} else if (location.hostname.endsWith('github.io')) {
    locationName = 'GitHub Pages';
} else {
    locationName = location.hostname;
}
document.getElementById('location').textContent = '/' + locationName;
