import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const pagePath = new URL('../app/projects/page.jsx', import.meta.url);
const clientPath = new URL('../app/projects/projects-client.jsx', import.meta.url);

test('server passes one canonical project collection across the RSC boundary', async () => {
  const source = await readFile(pagePath, 'utf8');
  assert.match(source, /<ProjectsClient\s+projects=\{projects\}\s*\/>/);
  assert.doesNotMatch(source, /projectNames=/);
  assert.doesNotMatch(source, /const\s+projectNames\s*=/);
});

test('client derives the presentation-only names from the canonical collection', async () => {
  const source = await readFile(clientPath, 'utf8');
  assert.match(source, /ProjectsClient\(\{\s*projects\s*\}\)/);
  assert.match(source, /projects\.map\(\(project\)\s*=>\s*project\.name\)/);
  assert.doesNotMatch(source, /projectNames\s*[,}]/);
});

test('existing accessible structure and project list remain present', async () => {
  const source = await readFile(clientPath, 'utf8');
  assert.match(source, /aria-labelledby="projects-title"/);
  assert.match(source, /projects\.map/);
  assert.match(source, /key=\{project\.id\}/);
});
