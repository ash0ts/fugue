import {createHash} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const root = process.argv[2];
if (!root) throw new Error('runtime root is required');
const plugin = path.join(root, 'node_modules/weave-claude-code');

function mutate(relative, transform) {
  const target = path.join(plugin, relative);
  const source = fs.readFileSync(target, 'utf8');
  const output = transform(source);
  if (output === source) throw new Error(`${relative}: pinned patch target mismatch`);
  fs.writeFileSync(target, output);
  return createHash('sha256').update(source).digest('hex');
}

const hashes = {};
hashes.marketplace = mutate('.claude-plugin/marketplace.json', source => {
  const value = JSON.parse(source);
  value.plugins = (value.plugins || []).map(plugin => ({...plugin, source: './'}));
  return `${JSON.stringify(value, null, 2)}\n`;
});
hashes.transcript = mutate('dist/transcriptFile.js', source => source.replace(
  'isPathWithinBase(resolved, os.homedir())',
  "[os.homedir(), process.env.CLAUDE_CONFIG_DIR].filter(Boolean).some(b => isPathWithinBase(resolved, b))",
));
hashes.spans = mutate('dist/genaiSpans.js', source => source
  .replace(
    "entries[`${WEAVE_INTEGRATION_META_PREFIX}${key}`] = { value };",
    "entries[(key.startsWith('fugue.') || key.startsWith('weave.eval.')) ? key : `${WEAVE_INTEGRATION_META_PREFIX}${key}`] = { value: String(value) };",
  )
  .replace(
    'if (key.startsWith(WEAVE_INTEGRATION_PREFIX)) {',
    "if (key.startsWith(WEAVE_INTEGRATION_PREFIX) || key.startsWith('fugue.') || key.startsWith('weave.eval.')) {",
  ));
hashes.daemon = mutate('dist/daemon.js', source => source
  .replace(
    'meta: { claude_code_app_version: claudeCodeAppVersion },',
    "meta: { claude_code_app_version: claudeCodeAppVersion, ...JSON.parse(process.env.FUGUE_TRACE_ATTRIBUTES_JSON || '{}') },",
  )
  .replace(
    `        const group = callsForResponseKey(calls, key);\n        const model = group.map(c => c.model).find(Boolean);`,
    `        const group = callsForResponseKey(calls, key);\n        if (group.length === 0) {\n            existingSpan?.end();\n            session.emittedChatSpanResponseKeys.add(key);\n            return;\n        }\n        const model = group.map(c => c.model).find(Boolean);`,
  ));
fs.writeFileSync(
  path.join(root, 'claude-code-patch-lock.json'),
  `${JSON.stringify(hashes, null, 2)}\n`,
);
