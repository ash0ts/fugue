#!/usr/bin/env node
"use strict";

// Generic trusted post-trial verifier. Production invokes this program with
// one host-authored /input/input.json. The immutable task archive and exact
// Agent output are separate read-only files; /work is an empty writable mount.

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const {spawnSync} = require("node:child_process");

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
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (canonical(actual) !== canonical(expected)) throw new Error(`${label} fields changed`);
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
    for (let index = 0; index < header.length; index += 1) {
      computedChecksum += index >= 148 && index < 156 ? 32 : header[index];
    }
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
  return files;
}

const inputPath = path.resolve(process.argv[2] || "");
if (!process.argv[2]) throw new Error("usage: node host_node_verifier.cjs /input/input.json");
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
if (output.schema_version !== 1 || output.task_id !== config.task_id || !output.files || typeof output.files !== "object" || Array.isArray(output.files) || !Object.keys(output.files).length) throw new Error("Agent output identity or files are invalid");
const allowed = new Set(config.allowed_paths);
for (const [relative, content] of Object.entries(output.files)) {
  if (!safeRelative(relative) || !allowed.has(relative) || typeof content !== "string") throw new Error(`undeclared output file: ${relative}`);
  const materialized = path.resolve(workspace, relative);
  if (!materialized.startsWith(`${workspace}${path.sep}`)) throw new Error(`output traversal rejected: ${relative}`);
  const existing = fs.lstatSync(materialized);
  if (!existing.isFile() || existing.isSymbolicLink()) throw new Error(`output target is not a base file: ${relative}`);
  fs.chmodSync(materialized, 0o644);
  fs.writeFileSync(materialized, content, {encoding: "utf8", mode: 0o444});
  fs.chmodSync(materialized, 0o444);
}

const run = spawnSync(process.execPath, ["--test"], {
  cwd: workspace,
  encoding: "utf8",
  timeout: 30_000,
  env: Object.fromEntries(
    ["HOME", "PATH", "TMPDIR"].filter((name) => process.env[name]).map((name) => [name, process.env[name]])
  ),
});
const outputFileDigests = Object.fromEntries(
  Object.entries(output.files).map(([relative, content]) => [relative, digest(content)])
);
const unsigned = {
  schema_version: 1,
  verifier_id: "fugue-node-test-v1",
  task_id: config.task_id,
  task_archive_sha256: config.task_archive.sha256,
  agent_output_sha256: config.agent_output.sha256,
  output_files_sha256: digest(canonical(outputFileDigests)),
  allowed_paths_digest: digest(canonical([...config.allowed_paths].sort())),
  runtime_lock_digest: config.runtime_lock_digest,
  observed_node_version: process.version,
  command: ["node", "--test"],
  status: run.status === 0 && !run.error ? "passed" : "failed",
  exit_code: Number.isInteger(run.status) ? run.status : null,
  stdout_sha256: digest(run.stdout || ""),
  stderr_sha256: digest(run.stderr || ""),
};
const receipt = {...unsigned, receipt_digest: digest(canonical(unsigned))};
process.stdout.write(`${JSON.stringify(receipt)}\n`);
process.exit(receipt.status === "passed" ? 0 : 1);
