
//
// Config for gettext-extract to create .pot from HTML files
//
// Usage:
// $ npm install -g gettext-extract gettext-extractor
// $ gettext-extract -c gettext.js  # Creates html.pot, cleanup manually
// $ pygettext.py -a *.py  # Creates messages.pot, cleanup manually
// # Now merge html.pot with messages.pot manually
// # and put the result into Steganography.pot
//

const { GettextExtractor, HtmlExtractors } = require('gettext-extractor');

const extractor = new GettextExtractor();

extractor.createHtmlParser([
    HtmlExtractors.elementContent('a'),
    HtmlExtractors.elementContent('button'),
    HtmlExtractors.elementContent('div'),
    HtmlExtractors.elementContent('h1'),
    HtmlExtractors.elementContent('h2'),
    HtmlExtractors.elementContent('label'),
    HtmlExtractors.elementContent('p'),
    HtmlExtractors.elementContent('span'),
    HtmlExtractors.elementContent('title'),

    HtmlExtractors.elementContent('#log'),

    HtmlExtractors.elementAttribute('a', 'title'),
    HtmlExtractors.elementAttribute('html', 'lang'),
    HtmlExtractors.elementAttribute('img', 'alt'),
    HtmlExtractors.elementAttribute('input', 'placeholder'),
    HtmlExtractors.elementAttribute('input', 'title'),
    HtmlExtractors.elementAttribute('meta', 'content'),
]).parseFilesGlob('../*.html');

extractor.savePotFile('html.pot');
extractor.printStats();
