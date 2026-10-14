# Vendorized Frontend Libraries

Para garantizar el principio **100% Local & Private** sin llamadas externas a CDNs en runtime,
estas librerías se encuentran vendorizadas localmente y son servidas por FastAPI desde /static/vendor/.

| Librería | Versión | Licencia | Tamaño | SHA-256 | Origen |
|---|---|---|---|---|---|
| marked | 12.0.2 | MIT | 35479 bytes | 15fabce5b65898b32b03f5ed25e9f891a729ad4c0d6d877110a7744aa847a894 | [CDN URL](https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js) |
| highlight.js | 11.9.0 | BSD-3-Clause | 121727 bytes | 837a6fa5b0c736b52bbde2b2b6190f305da3fc9ed41681db5321507057b5c846 | [CDN URL](https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js) |
| highlight.js (theme github-dark) | 11.9.0 | BSD-3-Clause | 1315 bytes | 9f208d022102b1d0c7aebfecd8e42ca7997d5de636649d2b31ea63093d809019 | [CDN URL](https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github-dark.min.css) |
| DOMPurify | 3.1.6 | MPL-2.0 / Apache-2.0 | 21496 bytes | c0845096a7c4a6741f362ac506c94c1c7d27dc603bcc1bf64a587f76f2dbe3a1 | [CDN URL](https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js) |

## Licencias
- **marked.js**: [MIT License](https://github.com/markedjs/marked/blob/master/LICENSE.md)
- **highlight.js**: [BSD-3-Clause License](https://github.com/highlightjs/highlight.js/blob/main/LICENSE)
- **DOMPurify**: [MPL-2.0 / Apache-2.0](https://github.com/cure53/DOMPurify/blob/main/LICENSE)

Fecha de descarga: 2026-10-14