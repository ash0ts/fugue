#!/usr/bin/env node
"use strict";

// Trusted, no-network package-structure validator. This checks only the public
// Agent Skill envelope; task-specific compatibility and instruction truth stay
// in the private-label-backed deterministic scorer.

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function digest(value) {
  const input = Buffer.isBuffer(value) ? value : Buffer.from(String(value));
  return crypto.createHash("sha256").update(input).digest("hex");
}

function exactKeys(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
  if (canonical(Object.keys(value).sort()) !== canonical([...keys].sort())) throw new Error(`${label} fields changed`);
}

function safeRelative(value) {
  if (typeof value !== "string" || !value || value.includes("\\") || path.posix.isAbsolute(value)) return false;
  const normalized = path.posix.normalize(value);
  return normalized === value && !value.split("/").some((part) => !part || part === "." || part === "..");
}

function inputFile(root, value, label) {
  if (typeof value !== "string" || !path.isAbsolute(value)) throw new Error(`${label} path must be absolute`);
  const selected = path.resolve(value);
  if (path.dirname(selected) !== root) throw new Error(`${label} must stay directly under /input`);
  const stat = fs.lstatSync(selected);
  if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`${label} must be a regular file`);
  return selected;
}

function cString(buffer) {
  const end = buffer.indexOf(0);
  return buffer.subarray(0, end < 0 ? buffer.length : end).toString("utf8");
}

function octal(buffer, label) {
  const value = cString(buffer).trim();
  if (!/^[0-7]+$/.test(value)) throw new Error(`invalid tar ${label}`);
  return Number.parseInt(value, 8);
}

function extractUstar(archivePath, workspace) {
  const payload = fs.readFileSync(archivePath);
  const folded = new Set();
  let offset = 0;
  let files = 0;
  while (offset + 512 <= payload.length) {
    const header = payload.subarray(offset, offset + 512);
    if (header.every((value) => value === 0)) break;
    const recordedChecksum = octal(header.subarray(148, 156), "checksum");
    let computedChecksum = 0;
    for (let index = 0; index < header.length; index += 1) computedChecksum += index >= 148 && index < 156 ? 32 : header[index];
    if (recordedChecksum !== computedChecksum) throw new Error("tar checksum changed");
    const name = cString(header.subarray(0, 100));
    const prefix = cString(header.subarray(345, 500));
    const archived = prefix ? `${prefix}/${name}` : name;
    const type = header[156];
    const size = octal(header.subarray(124, 136), "size");
    if (type !== 0 && type !== 48) throw new Error(`non-file tar entry rejected: ${archived}`);
    if (!safeRelative(archived)) throw new Error(`unsafe tar entry rejected: ${archived}`);
    const parts = archived.split("/");
    if (parts.length < 2 || !["repo", "workspace"].includes(parts[0])) throw new Error(`tar entry lacks reviewed root: ${archived}`);
    const relative = parts.slice(1).join("/");
    const collision = relative.toLocaleLowerCase("en-US");
    if (folded.has(collision)) throw new Error(`case-colliding tar entry: ${relative}`);
    folded.add(collision);
    const start = offset + 512;
    const end = start + size;
    if (end > payload.length) throw new Error(`truncated tar entry: ${relative}`);
    const destination = path.resolve(workspace, relative);
    if (!destination.startsWith(`${workspace}${path.sep}`)) throw new Error(`tar traversal rejected: ${relative}`);
    fs.mkdirSync(path.dirname(destination), {recursive: true, mode: 0o755});
    fs.writeFileSync(destination, payload.subarray(start, end), {mode: 0o444});
    files += 1;
    offset = start + Math.ceil(size / 512) * 512;
  }
  if (!files) throw new Error("task archive is empty");
}

function parseFrontmatter(text) {
  if (typeof text !== "string" || text.includes("\0") || text.includes("\r") || !text.startsWith("---\n")) throw new Error("SKILL.md frontmatter is missing or noncanonical");
  const marker = text.indexOf("\n---\n", 4);
  if (marker < 0) throw new Error("SKILL.md frontmatter is unterminated");
  const metadata = {};
  for (const line of text.slice(4, marker).split("\n")) {
    if (!line || /^\s/.test(line) || !line.includes(":")) throw new Error("SKILL.md frontmatter must use flat key-value fields");
    const index = line.indexOf(":");
    const key = line.slice(0, index).trim();
    let value = line.slice(index + 1).trim();
    if (!/^[a-z][a-z0-9-]*$/.test(key) || Object.hasOwn(metadata, key)) throw new Error("SKILL.md frontmatter key is invalid or duplicated");
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) value = value.slice(1, -1);
    metadata[key] = value;
  }
  const body = text.slice(marker + 5);
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(metadata.name || "") || metadata.name.length > 64) throw new Error("Skill name is invalid");
  if (!metadata.description || metadata.description.length > 1024) throw new Error("Skill description is missing or too long");
  if (!body.trim()) throw new Error("Skill instructions are empty");
  if (metadata.compatibility !== undefined && (!metadata.compatibility || metadata.compatibility.length > 500)) throw new Error("Skill compatibility is invalid");
}

function validateOutput(output, config, workspace) {
  exactKeys(output, ["schema_version", "task_id", "files", "summary"], "Agent output");
  if (output.schema_version !== 1 || output.task_id !== config.task_id || typeof output.summary !== "string" || !output.summary.trim()) throw new Error("Agent output identity or summary is invalid");
  if (!output.files || typeof output.files !== "object" || Array.isArray(output.files) || !Object.keys(output.files).length) throw new Error("Agent output files are invalid");
  const allowed = new Set(config.allowed_paths);
  const skillFiles = [];
  for (const [relative, content] of Object.entries(output.files)) {
    if (!safeRelative(relative) || !allowed.has(relative) || typeof content !== "string") throw new Error(`undeclared output file: ${relative}`);
    if (path.posix.basename(relative) === "SKILL.md") skillFiles.push(relative);
    const materialized = path.resolve(workspace, relative);
    if (!materialized.startsWith(`${workspace}${path.sep}`)) throw new Error(`output traversal rejected: ${relative}`);
    fs.mkdirSync(path.dirname(materialized), {recursive: true, mode: 0o755});
    if (fs.existsSync(materialized)) {
      const existing = fs.lstatSync(materialized);
      if (!existing.isFile() || existing.isSymbolicLink()) throw new Error(`output target is not a regular file: ${relative}`);
      fs.chmodSync(materialized, 0o644);
    }
    fs.writeFileSync(materialized, content, {encoding: "utf8", mode: 0o444});
    fs.chmodSync(materialized, 0o444);
  }
  if (skillFiles.length !== 1) throw new Error("Agent output must contain exactly one SKILL.md");
  parseFrontmatter(output.files[skillFiles[0]]);
  const combined = `${output.summary}\n${Object.values(output.files).join("\n")}`;
  if (/(?:api[_-]?key|token|password)\s*[:=]\s*[A-Za-z0-9_-]{12,}/i.test(combined) || /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/.test(combined)) throw new Error("secret-shaped content is forbidden");
}

const inputPath = path.resolve(process.argv[2] || "");
if (!process.argv[2]) throw new Error("usage: node host_skill_package_verifier.cjs /input/input.json");
const inputStat = fs.lstatSync(inputPath);
if (!inputStat.isFile() || inputStat.isSymbolicLink()) throw new Error("input.json must be a regular file");
const inputRoot = path.dirname(inputPath);
const config = JSON.parse(fs.readFileSync(inputPath, "utf8"));
exactKeys(config, ["schema_version", "task_id", "task_archive", "agent_output", "runtime_lock_digest", "workspace", "allowed_paths"], "input contract");
exactKeys(config.task_archive, ["path", "sha256"], "task archive");
exactKeys(config.agent_output, ["path", "sha256"], "Agent output");
if (config.schema_version !== 1 || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(config.task_id || "")) throw new Error("input identity is invalid");
if (!/^[0-9a-f]{64}$/.test(config.runtime_lock_digest || "")) throw new Error("runtime lock digest is invalid");
if (!Array.isArray(config.allowed_paths) || !config.allowed_paths.length || !config.allowed_paths.every(safeRelative) || new Set(config.allowed_paths).size !== config.allowed_paths.length) throw new Error("allowed paths are invalid");
const archivePath = inputFile(inputRoot, config.task_archive.path, "task archive");
const outputPath = inputFile(inputRoot, config.agent_output.path, "Agent output");
if (digest(fs.readFileSync(archivePath)) !== config.task_archive.sha256) throw new Error("task archive digest changed");
if (digest(fs.readFileSync(outputPath)) !== config.agent_output.sha256) throw new Error("Agent output digest changed");
const workspace = path.resolve(config.workspace || "");
const workspaceStat = fs.lstatSync(workspace);
if (!workspaceStat.isDirectory() || workspaceStat.isSymbolicLink() || fs.readdirSync(workspace).length) throw new Error("workspace must be an empty regular directory");
extractUstar(archivePath, workspace);

const output = JSON.parse(fs.readFileSync(outputPath, "utf8"));
let status = "passed";
let exitCode = 0;
let message = "Skill package contract passed";
try {
  validateOutput(output, config, workspace);
} catch (error) {
  status = "failed";
  exitCode = 1;
  message = error instanceof Error ? error.message : "Skill package contract failed";
}
const outputFileDigests = output && output.files && typeof output.files === "object" && !Array.isArray(output.files)
  ? Object.fromEntries(Object.entries(output.files).filter(([, content]) => typeof content === "string").map(([relative, content]) => [relative, digest(content)]))
  : {};
const unsigned = {
  schema_version: 1,
  verifier_id: "fugue-skill-package-validator-v1",
  task_id: config.task_id,
  task_archive_sha256: config.task_archive.sha256,
  agent_output_sha256: config.agent_output.sha256,
  output_files_sha256: digest(canonical(outputFileDigests)),
  allowed_paths_digest: digest(canonical([...config.allowed_paths].sort())),
  runtime_lock_digest: config.runtime_lock_digest,
  observed_node_version: process.version,
  command: ["node", "skill-package-validate"],
  status,
  exit_code: exitCode,
  stdout_sha256: digest(message),
  stderr_sha256: digest(""),
};
const receipt = {...unsigned, receipt_digest: digest(canonical(unsigned))};
process.stdout.write(`${JSON.stringify(receipt)}\n`);
process.exit(exitCode);
