import {createHash} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const root = process.argv[2];
if (!root) throw new Error('runtime root is required');

function patch(relative, needle, replacement) {
  const target = path.join(root, relative);
  const source = fs.readFileSync(target, 'utf8');
  const matches = source.split(needle).length - 1;
  if (matches !== 1 && !source.includes(replacement)) {
    throw new Error(`${relative}: pinned patch target mismatch (${matches})`);
  }
  fs.writeFileSync(target, source.replace(needle, replacement));
  return createHash('sha256').update(source).digest('hex');
}

const hashes = {};
for (const name of ['spanBase.js', 'spanBase.mjs']) {
  hashes[name] = patch(
    `node_modules/weave/dist/genai/${name}`,
    'this.span = span;',
    "this.span = span; try { this.span.setAttributes(JSON.parse(process.env.FUGUE_TRACE_ATTRIBUTES_JSON || '{}')); } catch {}",
  );
}
fs.writeFileSync(
  path.join(root, 'openclaw-patch-lock.json'),
  `${JSON.stringify(hashes, null, 2)}\n`,
);
