const { spawnSync } = require("node:child_process");
const { createHash } = require("node:crypto");
const {
  chmodSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} = require("node:fs");
const { dirname, join, posix } = require("node:path");
const { tmpdir } = require("node:os");

const MAX_ARCHIVE_BYTES = 1_000_000;
const MAX_BASE64_BYTES = Math.ceil((MAX_ARCHIVE_BYTES * 4) / 3) + 4;
const MAX_FILE_BYTES = 100_000;
const PUBLIC_TEST_PATH = "tests/task.test.mjs";

const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const canonical = (value) => {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonical(value[key])]),
    );
  }
  return value;
};
const canonicalJson = (value) =>
  JSON.stringify(canonical(value)).replace(/[\u007f-\uffff]/g, (character) =>
    `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
const stableDigest = (value) => sha256(canonicalJson(value));
const treeDigest = (files) =>
  stableDigest(
    [...files.entries()]
      .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
      .map(([path, body]) => [path, body.length, sha256(body)]),
  );
const infrastructureFailure = (message) => {
  throw new Error(message);
};

const safePath = (value) => {
  if (
    typeof value !== "string" ||
    value.length === 0 ||
    value.length > 240 ||
    value.startsWith("/") ||
    value.startsWith("\\") ||
    value.endsWith("/") ||
    value.includes("\\") ||
    value.includes("//") ||
    value.includes("\0") ||
    !/^[A-Za-z0-9._/-]+$/.test(value)
  ) {
    return false;
  }
  const normalized = posix.normalize(value);
  return (
    normalized === value &&
    normalized !== "." &&
    !normalized.startsWith("../") &&
    normalized.split("/").every((part) => part !== "." && part !== "..")
  );
};

const details = ({
  passed,
  failureKind,
  detail,
  exitCode,
  archiveContract,
  submittedArtifact,
  finalFiles,
}) => ({
  schema_version: 2,
  status: passed ? "passed" : "failed",
  failure_kind: passed ? null : failureKind,
  runtime: `node-${process.version}`,
  command: ["node", "--test", PUBLIC_TEST_PATH],
  exit_code: exitCode,
  test_count: 1,
  pass_count: passed ? 1 : 0,
  fail_count: passed ? 0 : 1,
  output_sha256: sha256(detail),
  base_archive_sha256: archiveContract.sha256,
  public_test_sha256: archiveContract.public_test_sha256,
  submitted_artifact_sha256: stableDigest(submittedArtifact),
  final_tree_sha256: treeDigest(finalFiles),
});

const candidateFailure = (code, context) => ({
  score: 0,
  reason: `candidate_failure:${code}`,
  details: details({
    passed: false,
    failureKind: code,
    detail: context.detail || code,
    exitCode: 1,
    archiveContract: context.archiveContract,
    submittedArtifact: context.submittedArtifact,
    finalFiles: context.finalFiles,
  }),
});

const readString = (header, start, length) => {
  const field = header.subarray(start, start + length);
  const end = field.indexOf(0);
  return field.subarray(0, end === -1 ? field.length : end).toString("utf8");
};

const readOctal = (header, start, length, label) => {
  const raw = readString(header, start, length).trim();
  if (!/^[0-7]+$/.test(raw)) {
    infrastructureFailure(`frozen archive has an invalid ${label}`);
  }
  const value = Number.parseInt(raw, 8);
  if (!Number.isSafeInteger(value) || value < 0) {
    infrastructureFailure(`frozen archive has an unsafe ${label}`);
  }
  return value;
};

const parseArchive = (archive) => {
  if (
    !Buffer.isBuffer(archive) ||
    archive.length === 0 ||
    archive.length > MAX_ARCHIVE_BYTES ||
    archive.length % 512 !== 0
  ) {
    infrastructureFailure("frozen archive has an invalid bounded size");
  }
  const files = new Map();
  let offset = 0;
  let zeroBlocks = 0;
  while (offset + 512 <= archive.length) {
    const header = archive.subarray(offset, offset + 512);
    offset += 512;
    if (header.every((value) => value === 0)) {
      zeroBlocks += 1;
      if (zeroBlocks === 2) {
        if (!archive.subarray(offset).every((value) => value === 0)) {
          infrastructureFailure("frozen archive has data after its terminator");
        }
        break;
      }
      continue;
    }
    if (zeroBlocks !== 0) {
      infrastructureFailure("frozen archive has a partial terminator");
    }
    const suppliedChecksum = readOctal(header, 148, 8, "header checksum");
    let computedChecksum = 0;
    for (let index = 0; index < header.length; index += 1) {
      computedChecksum += index >= 148 && index < 156 ? 32 : header[index];
    }
    if (suppliedChecksum !== computedChecksum) {
      infrastructureFailure("frozen archive header checksum does not match");
    }
    const magic = header.subarray(257, 263).toString("binary");
    if (magic !== "ustar\0") {
      infrastructureFailure("frozen archive is not strict USTAR");
    }
    const type = header[156];
    if (type !== 0 && type !== 48) {
      infrastructureFailure("frozen archive contains a non-regular entry");
    }
    const name = readString(header, 0, 100);
    const prefix = readString(header, 345, 155);
    const archivedPath = prefix ? `${prefix}/${name}` : name;
    if (!archivedPath.startsWith("repo/")) {
      infrastructureFailure("frozen archive entry is outside repo/");
    }
    const relative = archivedPath.slice("repo/".length);
    if (!safePath(relative) || files.has(relative)) {
      infrastructureFailure("frozen archive contains an unsafe or duplicate path");
    }
    const size = readOctal(header, 124, 12, "entry size");
    if (size > MAX_FILE_BYTES || offset + size > archive.length) {
      infrastructureFailure("frozen archive entry exceeds its bound");
    }
    const bodyEnd = offset + size;
    const paddedEnd = offset + Math.ceil(size / 512) * 512;
    if (!archive.subarray(bodyEnd, paddedEnd).every((value) => value === 0)) {
      infrastructureFailure("frozen archive entry has nonzero padding");
    }
    files.set(relative, Buffer.from(archive.subarray(offset, bodyEnd)));
    offset = paddedEnd;
  }
  if (zeroBlocks < 2 || files.size === 0) {
    infrastructureFailure("frozen archive is incomplete");
  }
  return files;
};

const validateArchiveContract = (contract, taskId) => {
  const contractKeys = [
    "content_base64",
    "encoding",
    "file_count",
    "files",
    "format",
    "manifest_digest",
    "public_test_path",
    "public_test_sha256",
    "sha256",
    "size",
    "task_id",
  ];
  if (
    !contract ||
    typeof contract !== "object" ||
    Array.isArray(contract) ||
    Object.keys(contract).sort().join("\n") !== contractKeys.join("\n") ||
    contract.task_id !== taskId ||
    contract.format !== "ustar" ||
    contract.encoding !== "base64" ||
    typeof contract.content_base64 !== "string" ||
    contract.content_base64.length > MAX_BASE64_BYTES ||
    !/^[A-Za-z0-9+/]+={0,2}$/.test(contract.content_base64) ||
    typeof contract.sha256 !== "string" ||
    !/^[0-9a-f]{64}$/.test(contract.sha256) ||
    !Number.isInteger(contract.size) ||
    !Number.isInteger(contract.file_count) ||
    !Array.isArray(contract.files) ||
    typeof contract.manifest_digest !== "string" ||
    !/^[0-9a-f]{64}$/.test(contract.manifest_digest) ||
    contract.public_test_path !== PUBLIC_TEST_PATH ||
    typeof contract.public_test_sha256 !== "string" ||
    !/^[0-9a-f]{64}$/.test(contract.public_test_sha256)
  ) {
    infrastructureFailure("frozen archive contract is malformed");
  }
  const archive = Buffer.from(contract.content_base64, "base64");
  if (
    archive.toString("base64") !== contract.content_base64 ||
    archive.length !== contract.size ||
    sha256(archive) !== contract.sha256
  ) {
    infrastructureFailure("frozen archive bytes do not match their lock");
  }
  if (stableDigest(contract.files) !== contract.manifest_digest) {
    infrastructureFailure("frozen archive manifest digest does not match");
  }
  const files = parseArchive(archive);
  if (files.size !== contract.file_count || files.size !== contract.files.length) {
    infrastructureFailure("frozen archive file count does not match");
  }
  const expectedPaths = new Set();
  let previousPath = "";
  for (const record of contract.files) {
    if (
      !record ||
      typeof record !== "object" ||
      Array.isArray(record) ||
      Object.keys(record).sort().join("\n") !== "path\nsha256\nsize" ||
      !safePath(record.path) ||
      (previousPath && record.path <= previousPath) ||
      expectedPaths.has(record.path) ||
      !Number.isInteger(record.size) ||
      record.size < 0 ||
      typeof record.sha256 !== "string" ||
      !/^[0-9a-f]{64}$/.test(record.sha256)
    ) {
      infrastructureFailure("frozen archive manifest is malformed");
    }
    previousPath = record.path;
    expectedPaths.add(record.path);
    const body = files.get(record.path);
    if (!body || body.length !== record.size || sha256(body) !== record.sha256) {
      infrastructureFailure("frozen archive file does not match its manifest");
    }
  }
  const publicTest = files.get(PUBLIC_TEST_PATH);
  if (!publicTest || sha256(publicTest) !== contract.public_test_sha256) {
    infrastructureFailure("frozen public test does not match its lock");
  }
  return files;
};

const writeLocked = (root, relative, body) => {
  const target = join(root, relative);
  mkdirSync(dirname(target), { recursive: true, mode: 0o755 });
  writeFileSync(target, body, { mode: 0o444 });
  chmodSync(target, 0o444);
};

const tapCount = (text, label) => {
  const match = text.match(new RegExp(`^# ${label}\\s+(\\d+)\\s*$`, "m"));
  return match ? Number.parseInt(match[1], 10) : null;
};

const verify = () => {
  const inputPath = process.argv[2];
  if (!inputPath) infrastructureFailure("host verifier requires one input path");
  if ((statSync(dirname(inputPath)).mode & 0o077) !== 0) {
    infrastructureFailure("private verifier input mount has unsafe permissions");
  }
  const payload = JSON.parse(readFileSync(inputPath, "utf8"));
  const reference = payload && payload.reference;
  const task = reference && reference.task;
  const taskId = task && task.id;
  const expected = reference && reference.expected;
  if (
    typeof taskId !== "string" ||
    taskId.length === 0 ||
    !expected ||
    typeof expected !== "object" ||
    Array.isArray(expected) ||
    expected.task_id !== taskId
  ) {
    infrastructureFailure("host verifier expected contract is unavailable");
  }
  const required = expected.required_file_paths;
  const allowed = expected.allowed_file_paths;
  if (
    !Array.isArray(required) ||
    required.length === 0 ||
    !Array.isArray(allowed) ||
    required.length !== allowed.length ||
    new Set(required).size !== required.length ||
    new Set(allowed).size !== allowed.length ||
    required.some((value) => !safePath(value)) ||
    allowed.some((value) => !safePath(value)) ||
    [...required].sort().join("\n") !== [...allowed].sort().join("\n")
  ) {
    infrastructureFailure("host verifier allowlist is malformed");
  }
  const archiveContract = {
    task_id: expected.task_id,
    format: expected.base_archive_format,
    encoding: "base64",
    content_base64: expected.base_archive_base64,
    sha256: expected.base_archive_sha256,
    size: expected.base_archive_size,
    file_count: expected.base_archive_file_count,
    files: expected.base_archive_files,
    manifest_digest: expected.base_archive_manifest_digest,
    public_test_path: expected.public_test_path,
    public_test_sha256: expected.public_test_sha256,
  };
  const archiveFiles = validateArchiveContract(archiveContract, taskId);
  const finalFiles = new Map(archiveFiles);
  const requiredArchivePaths = new Set([
    "package.json",
    "README.md",
    PUBLIC_TEST_PATH,
    ...required,
  ]);
  if (
    requiredArchivePaths.size !== archiveFiles.size ||
    [...requiredArchivePaths].some((path) => !archiveFiles.has(path))
  ) {
    infrastructureFailure("frozen archive does not match its public file contract");
  }

  const output = reference.output;
  const context = {
    archiveContract,
    submittedArtifact: null,
    finalFiles,
  };
  if (!output || typeof output !== "object" || Array.isArray(output)) {
    return candidateFailure("missing_structured_output", context);
  }
  context.submittedArtifact = output;
  if (
    output.schema_version !== 1 ||
    output.task_id !== taskId ||
    output.status !== "completed"
  ) {
    return candidateFailure("output_identity_mismatch", {
      ...context,
      submittedArtifact: output,
    });
  }
  const returned = output.files;
  if (!returned || typeof returned !== "object" || Array.isArray(returned)) {
    return candidateFailure("missing_files_object", context);
  }
  if (
    Object.keys(returned).sort().join("\n") !== [...required].sort().join("\n")
  ) {
    return candidateFailure("submitted_files_do_not_match_allowlist", context);
  }
  for (const [relative, body] of Object.entries(returned)) {
    if (
      !safePath(relative) ||
      !allowed.includes(relative) ||
      typeof body !== "string" ||
      body.length === 0 ||
      Buffer.byteLength(body) > MAX_FILE_BYTES ||
      Buffer.from(body, "utf8").toString("utf8") !== body ||
      body.includes("\0")
    ) {
      return candidateFailure("submitted_file_is_not_bounded_utf8_text", context);
    }
    if (
      /\bprocess\s*\./.test(body) ||
      /node:(?:test|child_process|worker_threads|vm|module)/.test(body)
    ) {
      return candidateFailure("submitted_file_can_interfere_with_test_runner", context);
    }
    finalFiles.set(relative, Buffer.from(body, "utf8"));
  }

  const root = mkdtempSync(join(tmpdir(), "fugue-node-verifier-"));
  chmodSync(root, 0o755);
  try {
    for (const [relative, body] of archiveFiles.entries()) {
      writeLocked(root, relative, body);
    }
    for (const [relative, body] of Object.entries(returned)) {
      const target = join(root, relative);
      chmodSync(target, 0o644);
      writeFileSync(target, body, { encoding: "utf8", mode: 0o444 });
      chmodSync(target, 0o444);
    }
    const completed = spawnSync(
      process.execPath,
      [
        "--permission",
        `--allow-fs-read=${root}`,
        "--experimental-test-isolation=none",
        "--test",
        PUBLIC_TEST_PATH,
      ],
      {
        cwd: root,
        encoding: "utf8",
        timeout: 20_000,
        maxBuffer: 2_000_000,
        env: {
          HOME: tmpdir(),
          PATH: dirname(process.execPath),
          NODE_NO_WARNINGS: "1",
        },
      },
    );
    if (
      completed.error &&
      completed.error.code !== "ETIMEDOUT" &&
      completed.error.code !== "ENOBUFS"
    ) {
      infrastructureFailure(
        `Node verifier infrastructure failed: ${completed.error.code}`,
      );
    }
    const detail = `${completed.stdout || ""}${completed.stderr || ""}`;
    if (completed.error && completed.error.code === "ETIMEDOUT") {
      return candidateFailure("public_test_timeout", { ...context, detail });
    }
    if (completed.error && completed.error.code === "ENOBUFS") {
      return candidateFailure("public_test_output_limit_exceeded", {
        ...context,
        detail,
      });
    }
    const testCount = tapCount(detail, "tests");
    const passCount = tapCount(detail, "pass");
    const failCount = tapCount(detail, "fail");
    const namedTest =
      typeof expected.public_test_name === "string" &&
      expected.public_test_name.length > 0 &&
      detail.includes(expected.public_test_name);
    const passed =
      completed.status === 0 &&
      namedTest &&
      testCount === 1 &&
      passCount === 1 &&
      failCount === 0;
    if (passed) {
      return {
        score: 1,
        reason: "public_test_passed_on_frozen_repository",
        details: details({
          passed: true,
          failureKind: null,
          detail,
          exitCode: 0,
          archiveContract,
          submittedArtifact: output,
          finalFiles,
        }),
      };
    }
    return {
      score: 0,
      reason: "candidate_failure:public_test_failed",
      details: details({
        passed: false,
        failureKind: "public_test_failed",
        detail,
        exitCode: 1,
        archiveContract,
        submittedArtifact: output,
        finalFiles,
      }),
    };
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
};

process.stdout.write(`${JSON.stringify(verify())}\n`);
