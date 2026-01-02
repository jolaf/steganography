const target = document.getElementById('log');

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
