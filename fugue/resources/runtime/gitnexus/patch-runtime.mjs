import {createHash} from "node:crypto";
import {readFileSync, writeFileSync} from "node:fs";
import {join} from "node:path";

const root = process.argv[2];
if (!root) throw new Error("usage: patch-runtime.mjs GITNEXUS_ROOT");

function patch(relative, expected, replacements) {
  const path = join(root, relative);
  let source = readFileSync(path, "utf8");
  const actual = createHash("sha256").update(source).digest("hex");
  if (actual !== expected) {
    throw new Error(`${relative}: upstream digest ${actual} != ${expected}`);
  }
  for (const [needle, replacement, expectedCount = 1] of replacements) {
    const count = source.split(needle).length - 1;
    if (count !== expectedCount) {
      throw new Error(
        `${relative}: expected ${expectedCount} patch targets, found ${count}`,
      );
    }
    source = source.split(needle).join(replacement);
  }
  writeFileSync(path, source);
}

const localModels = [
  "            env.allowLocalModels = false;",
  "            env.localModelPath = '/opt/gitnexus-models';\n" +
    "            env.allowLocalModels = true;\n" +
    "            env.allowRemoteModels = false;",
];

patch(
  "dist/core/embeddings/embedder.js",
  "c20ba0f01a586bf33bbe5f3892d987f60e5d00f86f594e176fd0af52a12ce83f",
  [
    localModels,
    [
      "            const devicesToTry = requestedDevice === 'dml' || requestedDevice === 'cuda'\n" +
        "                ? [requestedDevice, 'cpu']\n" +
        "                : [requestedDevice];",
      "            const devicesToTry = ['cpu'];",
    ],
  ],
);

patch(
  "dist/mcp/core/embedder.js",
  "472efa00aae8e820423643c7f614677965cb910b97bb16721118fb1de2a80b50",
  [
    localModels,
    [
      "            const isWindows = process.platform === 'win32';\n" +
        "            const gpuDevice = isWindows ? 'dml' : 'cuda';\n" +
        "            const devicesToTry = [gpuDevice, 'cpu'];",
      "            const devicesToTry = ['cpu'];",
    ],
  ],
);

patch(
  "dist/mcp/local/local-backend.js",
  "3bb70ce73625ce0f751d599d63ecee6edee473cd10b1680f9204b65402459857",
  [
    [
      "            const { embedQuery, getEmbeddingDims } = await import('../core/embedder.js');",
      "            const fugueVectorStartedAt = performance.now();\n" +
        "            console.error('FUGUE_GITNEXUS_VECTOR ' + JSON.stringify({vector_search_attempted:true,model_digest:process.env.FUGUE_GITNEXUS_MODEL_DIGEST ?? null}));\n" +
        "            const { embedQuery, getEmbeddingDims } = await import('../core/embedder.js');",
    ],
    [
      "            return results;\n        }\n        catch {\n" +
      "            // Expected when embeddings are disabled — silently fall back to BM25-only\n" +
      "            return [];\n" +
      "        }",
      "            console.error('FUGUE_GITNEXUS_VECTOR ' + JSON.stringify({vector_search_succeeded:true,semantic_result_count:results.length,model_digest:process.env.FUGUE_GITNEXUS_MODEL_DIGEST ?? null,query_latency_ms:Math.round((performance.now()-fugueVectorStartedAt)*1000)/1000}));\n" +
        "            return results;\n        }\n        catch (error) {\n" +
      "            if (process.env.FUGUE_GITNEXUS_VECTOR_REQUIRED === '1') {\n" +
      "                throw error;\n" +
      "            }\n" +
      "            return [];\n" +
      "        }",
    ],
    [
      "        const bm25Results = bm25SearchResult.results;\n" +
        "        const ftsUsed = bm25SearchResult.ftsUsed;",
      "        const bm25Results = bm25SearchResult.results;\n" +
        "        const ftsUsed = bm25SearchResult.ftsUsed;\n" +
        "        if (process.env.FUGUE_GITNEXUS_VECTOR_REQUIRED === '1') {\n" +
        "            console.error('FUGUE_GITNEXUS_VECTOR ' + JSON.stringify({vector_search_attempted:true,vector_search_succeeded:true,semantic_result_count:semanticResults.length,bm25_result_count:bm25Results.length,model_digest:process.env.FUGUE_GITNEXUS_MODEL_DIGEST ?? null}));\n" +
        "        }",
    ],
    [
      "            const dims = getEmbeddingDims();\n" +
        "            const queryVecStr = `[${queryVec.join(',')}]`;\n" +
        "            const bestChunks = await collectBestChunks(limit, async (fetchLimit) => {\n" +
        "                const vectorQuery = `\n" +
        "          CALL QUERY_VECTOR_INDEX('${EMBEDDING_TABLE_NAME}', '${EMBEDDING_INDEX_NAME}',\n" +
        "            CAST(${queryVecStr} AS FLOAT[${dims}]), ${fetchLimit})\n" +
        "          YIELD node AS emb, distance\n" +
        "          WITH emb, distance\n" +
        "          WHERE distance < 0.6\n" +
        "          RETURN emb.nodeId AS nodeId, emb.chunkIndex AS chunkIndex,\n" +
        "                 emb.startLine AS startLine, emb.endLine AS endLine, distance\n" +
        "          ORDER BY distance\n" +
        "        `;\n" +
        "                const embResults = await executeQuery(repo.id, vectorQuery);\n" +
        "                return embResults.map((row) => ({\n" +
        "                    nodeId: row.nodeId ?? row[0],\n" +
        "                    chunkIndex: row.chunkIndex ?? row[1] ?? 0,\n" +
        "                    startLine: row.startLine ?? row[2] ?? 0,\n" +
        "                    endLine: row.endLine ?? row[3] ?? 0,\n" +
        "                    distance: row.distance ?? row[4],\n" +
        "                }));\n" +
        "            });",
      "            getEmbeddingDims();\n" +
        "            const embeddingRows = await executeQuery(repo.id, `\n" +
        "              MATCH (e:${EMBEDDING_TABLE_NAME})\n" +
        "              RETURN e.nodeId AS nodeId, e.chunkIndex AS chunkIndex,\n" +
        "                     e.startLine AS startLine, e.endLine AS endLine,\n" +
        "                     e.embedding AS embedding\n" +
        "            `);\n" +
        "            const ranked = embeddingRows.map((row) => {\n" +
        "                const embedding = row.embedding ?? row[4];\n" +
        "                if (!Array.isArray(embedding) || embedding.length !== queryVec.length) {\n" +
        "                    throw new Error('GitNexus flat vector index has invalid dimensions');\n" +
        "                }\n" +
        "                let similarity = 0;\n" +
        "                for (let i = 0; i < queryVec.length; i += 1) {\n" +
        "                    similarity += queryVec[i] * embedding[i];\n" +
        "                }\n" +
        "                return {\n" +
        "                    nodeId: row.nodeId ?? row[0],\n" +
        "                    chunkIndex: row.chunkIndex ?? row[1] ?? 0,\n" +
        "                    startLine: row.startLine ?? row[2] ?? 0,\n" +
        "                    endLine: row.endLine ?? row[3] ?? 0,\n" +
        "                    distance: 1 - similarity,\n" +
        "                };\n" +
        "            }).filter((row) => row.distance < 0.6)\n" +
        "                .sort((left, right) => left.distance - right.distance);\n" +
        "            const bestChunks = new Map();\n" +
        "            for (const row of ranked) {\n" +
        "                if (!bestChunks.has(row.nodeId)) bestChunks.set(row.nodeId, row);\n" +
        "                if (bestChunks.size >= limit) break;\n" +
        "            }",
    ],
    [
      "        const { searchFTSFromLbug } = await import('../../core/search/bm25-index.js');\n" +
        "        let bm25Results;\n" +
        "        try {\n" +
        "            bm25Results = await searchFTSFromLbug(query, limit, repo.id);\n" +
        "        }\n" +
        "        catch (err) {\n" +
        "            console.error('GitNexus: BM25/FTS search failed (FTS indexes may not exist) -', err.message);\n" +
        "            return { results: [], ftsUsed: false };\n" +
        "        }\n" +
        "        const ftsUsed = bm25Results.length === 0 || bm25Results[0]?.ftsUsed !== false;",
      "        const rows = await executeQuery(repo.id, `\n" +
        "          MATCH (n)\n" +
        "          RETURN n.id AS id, n.name AS name, labels(n)[0] AS type,\n" +
        "                 n.filePath AS filePath, n.content AS content\n" +
        "        `);\n" +
        "        const tokenize = (value) => String(value ?? '').toLowerCase().match(/[a-z0-9_]+/g) ?? [];\n" +
        "        const queryTerms = [...new Set(tokenize(query))];\n" +
        "        const documents = rows.map((row) => ({\n" +
        "            id: row.id ?? row[0],\n" +
        "            filePath: row.filePath ?? row[3] ?? '',\n" +
        "            terms: tokenize(`${row.name ?? row[1] ?? ''} ${row.content ?? row[4] ?? ''}`),\n" +
        "        })).filter((row) => row.id && row.filePath && row.terms.length);\n" +
        "        const averageLength = documents.reduce((sum, row) => sum + row.terms.length, 0) / Math.max(documents.length, 1);\n" +
        "        const documentFrequency = new Map(queryTerms.map((term) => [term, documents.filter((row) => row.terms.includes(term)).length]));\n" +
        "        const bm25Results = documents.map((row) => {\n" +
        "            const frequencies = new Map();\n" +
        "            for (const term of row.terms) frequencies.set(term, (frequencies.get(term) ?? 0) + 1);\n" +
        "            let score = 0;\n" +
        "            for (const term of queryTerms) {\n" +
        "                const frequency = frequencies.get(term) ?? 0;\n" +
        "                if (!frequency) continue;\n" +
        "                const seen = documentFrequency.get(term) ?? 0;\n" +
        "                const inverseFrequency = Math.log(1 + (documents.length - seen + 0.5) / (seen + 0.5));\n" +
        "                const denominator = frequency + 1.2 * (0.25 + 0.75 * row.terms.length / Math.max(averageLength, 1));\n" +
        "                score += inverseFrequency * frequency * 2.2 / denominator;\n" +
        "            }\n" +
        "            return { filePath: row.filePath, nodeIds: [row.id], score };\n" +
        "        }).filter((row) => row.score > 0).sort((left, right) => right.score - left.score).slice(0, limit);\n" +
        "        const ftsUsed = true;",
    ],
  ],
);

patch(
  "dist/core/lbug/lbug-adapter.js",
  "13f98b12997af1424a6f87758663f31cf89de321946507ddb00c5446ac03dfc5",
  [
    [
      "    try {\n" +
        "        // Try loading locally first (no network required)\n" +
        "        await c.query('LOAD EXTENSION fts');\n" +
        "        return markLoaded();\n" +
        "    }\n" +
        "    catch {\n" +
        "        // Fall back to install + load (requires network)\n" +
        "        try {\n" +
        "            await c.query('INSTALL fts');\n" +
        "            await c.query('LOAD EXTENSION fts');\n" +
        "            return markLoaded();\n" +
        "        }\n" +
        "        catch (err) {\n" +
        "            const msg = err?.message || '';\n" +
        "            if (msg.includes('already loaded') ||\n" +
        "                msg.includes('already installed') ||\n" +
        "                msg.includes('already exists')) {\n" +
        "                return markLoaded();\n" +
        "            }\n" +
        "            console.error('GitNexus: FTS extension load failed:', msg);\n" +
        "            return false;\n" +
        "        }\n" +
        "    }",
      "    return markLoaded();",
    ],
    [
      "        await conn.query('INSTALL VECTOR');\n        await conn.query('LOAD EXTENSION VECTOR');",
      "        vectorExtensionLoaded = true;",
    ],
  ],
);

patch(
  "dist/core/lbug/pool-adapter.js",
  "d6f7c3fdce8f14525c814d8bec9aae04d048181a1ebd2dec9e41569951ef3f3d",
  [
    [
      "            await available[0].query('INSTALL VECTOR');\n            await available[0].query('LOAD EXTENSION VECTOR');",
      "            shared.vectorLoaded = true;",
      2,
    ],
  ],
);

patch(
  "dist/core/embeddings/embedding-pipeline.js",
  "f7db89b42553509ba4087cb31029fec84c6286aa79b0ef099005dc4a331ca658",
  [
    [
      "    // Delegate to the adapter which tracks loaded state and handles DB reconnect resets\n" +
        "    await loadVectorExtension();\n" +
        "    try {\n" +
        "        await executeQuery(CREATE_VECTOR_INDEX_QUERY);\n" +
        "    }\n" +
        "    catch (error) {\n" +
        "        if (isDev) {\n" +
        "            console.warn('Vector index creation warning:', error);\n" +
        "        }\n" +
        "    }",
      "    const rows = await executeQuery(`MATCH (e:${EMBEDDING_TABLE_NAME}) RETURN COUNT(*) AS count`);\n" +
        "    const count = rows[0]?.count ?? rows[0]?.[0] ?? 0;\n" +
        "    if (count <= 0) throw new Error('GitNexus flat vector index is empty');",
    ],
  ],
);
