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
hashes['weave-openclaw/run-start'] = patch(
  'node_modules/weave-openclaw/dist/src/handlers/diagnostic/run.js',
  `            const session = getOrCreateSession(deps, event.sessionKey, agentName);
            const turn = runIsolated(() => session
                ? session.startTurn({ agentName, model: event.model })
                : startTurn({ agentName, model: event.model }));
            if (resolved.agentVersion)
                turn.setAttribute("weave.agent.version", resolved.agentVersion);
            if (resolved.agentDescription)
                turn.setAttribute("gen_ai.agent.description", resolved.agentDescription);
            deps.registries.turns.set(event.runId, turn);`,
  `            const stableTurnKey = process.env.FUGUE_WEAVE_SINGLE_TURN_KEY;
            const existing = stableTurnKey
                ? deps.registries.turns.get(stableTurnKey)
                : undefined;
            const session = getOrCreateSession(deps, event.sessionKey, agentName);
            const turn = existing ?? runIsolated(() => session
                ? session.startTurn({ agentName, model: event.model })
                : startTurn({ agentName, model: event.model }));
            if (!existing && resolved.agentVersion)
                turn.setAttribute("weave.agent.version", resolved.agentVersion);
            if (!existing && resolved.agentDescription)
                turn.setAttribute("gen_ai.agent.description", resolved.agentDescription);
            if (stableTurnKey && !existing)
                deps.registries.turns.set(stableTurnKey, turn);
            deps.registries.turns.set(event.runId, turn);`,
);
hashes['weave-openclaw/run-finalize'] = patch(
  'node_modules/weave-openclaw/dist/src/handlers/diagnostic/run.js',
  `            turn.setAttribute("weave.outcome", event.outcome);
            if (isErrorOutcome(event.outcome)) {
                turn.end({ error: new Error(event.outcome) });
            }
            else {
                turn.end();
            }
            deps.registries.turns.delete(event.runId);`,
  `            turn.setAttribute("weave.outcome", event.outcome);
            const stableTurnKey = process.env.FUGUE_WEAVE_SINGLE_TURN_KEY;
            if (stableTurnKey) {
                turn.addEvent("run_segment_finalized", {
                    "weave.run.segment_outcome": event.outcome,
                });
            }
            else if (isErrorOutcome(event.outcome)) {
                turn.end({ error: new Error(event.outcome) });
            }
            else {
                turn.end();
            }
            deps.registries.turns.delete(event.runId);`,
);
hashes['weave-openclaw/service-stop'] = patch(
  'node_modules/weave-openclaw/dist/src/plugin.js',
  `        async stop(ctx) {
            lifecycle = "stopped";
            try {
                await flushOTel();`,
  `        async stop(ctx) {
            lifecycle = "stopped";
            const stableTurnKey = process.env.FUGUE_WEAVE_SINGLE_TURN_KEY;
            const stableTurn = stableTurnKey
                ? registries.turns.get(stableTurnKey)
                : undefined;
            if (stableTurn) {
                stableTurn.end();
                registries.turns.delete(stableTurnKey);
            }
            try {
                await flushOTel();`,
);
fs.writeFileSync(
  path.join(root, 'openclaw-patch-lock.json'),
  `${JSON.stringify(hashes, null, 2)}\n`,
);
