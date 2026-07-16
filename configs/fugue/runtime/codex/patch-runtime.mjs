import {createHash} from "node:crypto";
import {readFileSync, writeFileSync} from "node:fs";
import {join} from "node:path";

const root = process.argv[2];
if (!root) throw new Error("usage: patch-runtime.mjs WEAVE_CODEX_ROOT");
const path = join(root, "dist/spans/emit.js");
let source = readFileSync(path, "utf8");
const actual = createHash("sha256").update(source).digest("hex");
const expected = "256848e0da221e204ae471aeb45879b4a5bbc14ab0f0a5cd2a0a7060a830eabc";
if (actual !== expected) {
  throw new Error(`weave-codex emit.js digest ${actual} != ${expected}`);
}

const agentNeedle = "const AGENT_NAME = 'codex';";
if (source.split(agentNeedle).length - 1 !== 1) {
  throw new Error("weave-codex agent identity patch target changed");
}
source = source.replace(
  agentNeedle,
  "const AGENT_NAME = process.env.WEAVE_CODEX_AGENT_NAME || 'codex';\n" +
    "const FUGUE_TRACE_ATTRIBUTES = (() => { try { return JSON.parse(process.env.FUGUE_TRACE_ATTRIBUTES_JSON || '{}'); } catch { return {}; } })();",
);

const attributesNeedle = "const spanAttributes = {";
if (source.split(attributesNeedle).length - 1 !== 3) {
  throw new Error("weave-codex span attribute patch targets changed");
}
source = source.split(attributesNeedle).join(
  "const spanAttributes = { ...FUGUE_TRACE_ATTRIBUTES,",
);
writeFileSync(path, source);

const parserPath = join(root, "dist/rollout/parser.js");
let parser = readFileSync(parserPath, "utf8");
const parserActual = createHash("sha256").update(parser).digest("hex");
const parserExpected = "7c5c83f0b79d9505c3501b70fc90c96e0bf40156ca1ccd10d8442c3700e05869";
if (parserActual !== parserExpected) {
  throw new Error(
    `weave-codex parser.js digest ${parserActual} != ${parserExpected}`,
  );
}
const resultNeedle =
  "            if (tool) {\n" +
  "                tool.result = renderOutput(functionOutput.output);\n" +
  "                tool.endTime = timestamp;\n" +
  "            }";
if (parser.split(resultNeedle).length - 1 !== 1) {
  throw new Error("weave-codex MCP result patch target changed");
}
parser = parser.replace(
  resultNeedle,
  "            if (tool) {\n" +
    "                // mcp_tool_call_end carries gateway correlation that the later function output omits.\n" +
    "                if (tool.kind !== 'mcp')\n" +
    "                    tool.result = renderOutput(functionOutput.output);\n" +
    "                tool.endTime = timestamp;\n" +
    "            }",
);
writeFileSync(parserPath, parser);
