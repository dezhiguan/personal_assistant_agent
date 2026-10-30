// Render untrusted evaluation text with the UI's parser; never execute its HTML.
import { marked } from '../src/personal_assistant_agent/static/vendor/marked.esm.js';
let source = '';
for await (const chunk of process.stdin) source += chunk;
process.stdout.write(marked.parse(source, { gfm: true, breaks: true, async: false }));
