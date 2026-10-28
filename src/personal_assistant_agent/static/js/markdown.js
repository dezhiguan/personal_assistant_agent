import { marked } from '../vendor/marked.esm.js';
import DOMPurify from '../vendor/purify.es.mjs';

const OPTIONS = { gfm: true, breaks: true, async: false };
const SANITIZE = {
  ALLOWED_TAGS: [
    'p', 'br', 'strong', 'em', 'del', 's', 'blockquote', 'ul', 'ol', 'li',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'code', 'a', 'hr',
    'table', 'thead', 'tbody', 'tr', 'th', 'td',
  ],
  ALLOWED_ATTR: ['href', 'title', 'start', 'align'],
  RETURN_DOM_FRAGMENT: true,
};

/** Use the same safe renderer for streaming, finished and restored replies. */
export function renderMarkdown(element, source) {
  const html = marked.parse(source, OPTIONS);
  const fragment = DOMPurify.sanitize(html, SANITIZE);

  for (const link of fragment.querySelectorAll('a[href]')) {
    const href = link.getAttribute('href');
    try {
      const url = new URL(href, window.location.origin);
      if (!['http:', 'https:', 'mailto:'].includes(url.protocol)) {
        link.removeAttribute('href');
        continue;
      }
      if (!href.startsWith('#')) {
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
      }
    } catch { link.removeAttribute('href'); }
  }

  // Wide tables scroll inside the answer instead of expanding the whole page.
  for (const table of fragment.querySelectorAll('table')) {
    const wrapper = document.createElement('div');
    wrapper.className = 'markdown-table';
    wrapper.tabIndex = 0;
    wrapper.setAttribute('role', 'region');
    wrapper.setAttribute('aria-label', '表格，可左右滚动');
    table.replaceWith(wrapper);
    wrapper.append(table);
  }
  element.replaceChildren(fragment);
}
