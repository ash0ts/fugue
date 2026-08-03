"""Frozen source catalog for the Vercel confirmatory Skill study.

This module is consumed only during trusted preparation and host-side tests.  The
Agent receives the generated base repository archive, never this catalog or its
gold sources.
"""

from __future__ import annotations

from typing import Any


def _auth_fixture(
    *,
    fixture_id: str,
    split: str,
    title: str,
    symptom: str,
    source_name: str,
    function_name: str,
    base_source: str,
    gold_source: str,
    test_source: str,
    verifier: dict[str, Any],
) -> dict[str, Any]:
    path = f"app/{source_name}.mjs"
    return {
        "id": fixture_id,
        "split": split,
        "family": "server-action-security",
        "title": title,
        "symptom": symptom,
        "target_files": {path: {"base": base_source, "gold": gold_source}},
        "public_test_name": f"{fixture_id} reproduces and repairs the reported behavior",
        "public_test_source": test_source,
        "required_inspected_paths": [path, "README.md", "tests/task.test.mjs"],
        "verifier": {
            "kind": "server_action",
            "source_path": path,
            "export_name": function_name,
            **verifier,
        },
    }


def _rsc_fixture(
    *,
    fixture_id: str,
    split: str,
    title: str,
    symptom: str,
    server_source: str,
    server_gold: str,
    client_source: str,
    client_gold: str,
    test_source: str,
    verifier: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": fixture_id,
        "split": split,
        "family": "rsc-serialization",
        "title": title,
        "symptom": symptom,
        "target_files": {
            "app/server.mjs": {"base": server_source, "gold": server_gold},
            "app/client.mjs": {"base": client_source, "gold": client_gold},
        },
        "public_test_name": f"{fixture_id} reproduces and repairs the reported behavior",
        "public_test_source": test_source,
        "required_inspected_paths": [
            "app/server.mjs",
            "app/client.mjs",
            "README.md",
            "tests/task.test.mjs",
        ],
        "verifier": {
            "kind": "rsc_props",
            "server_path": "app/server.mjs",
            "client_path": "app/client.mjs",
            **verifier,
        },
    }


def _single_fixture(
    *,
    fixture_id: str,
    split: str,
    family: str,
    title: str,
    symptom: str,
    source_path: str,
    base_source: str,
    gold_source: str,
    test_source: str,
    verifier: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": fixture_id,
        "split": split,
        "family": family,
        "title": title,
        "symptom": symptom,
        "target_files": {source_path: {"base": base_source, "gold": gold_source}},
        "public_test_name": f"{fixture_id} reproduces and repairs the reported behavior",
        "public_test_source": test_source,
        "required_inspected_paths": [source_path, "README.md", "tests/task.test.mjs"],
        "verifier": {"source_path": source_path, **verifier},
    }


FIXTURES: tuple[dict[str, Any], ...] = (
    _auth_fixture(
        fixture_id="vr-dev-signed-in-nonmember-action",
        split="discovery",
        title="Prevent a signed-in non-member from renaming a team",
        symptom="A direct action call by a signed-in non-member renames a team.",
        source_name="teams",
        function_name="renameTeam",
        base_source="""'use server';

export async function renameTeam(input, { auth, db }) {
  const session = await auth();
  if (!session?.user?.id) throw new Error('Authentication required');
  const teamId = String(input?.teamId || '').trim();
  const name = String(input?.name || '').trim();
  if (!teamId || name.length < 2 || name.length > 80) throw new Error('Invalid team update');
  return db.team.update({ where: { id: teamId }, data: { name } });
}
""",
        gold_source="""'use server';

export async function renameTeam(input, { auth, db }) {
  const session = await auth();
  if (!session?.user?.id) throw new Error('Authentication required');
  const teamId = String(input?.teamId || '').trim();
  const name = String(input?.name || '').trim();
  if (!teamId || name.length < 2 || name.length > 80) throw new Error('Invalid team update');
  const membership = await db.membership.findFirst({
    where: { teamId, userId: session.user.id, role: { in: ['OWNER', 'ADMIN'] } },
  });
  if (!membership) throw new Error('Not authorized');
  return db.team.update({ where: { id: teamId }, data: { name } });
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { renameTeam } from '../app/teams.mjs';

test('vr-dev-signed-in-nonmember-action reproduces and repairs the reported behavior', async () => {
  let writes = 0;
  const services = {
    auth: async () => ({ user: { id: 'outsider' } }),
    db: {
      membership: { findFirst: async () => null },
      team: { update: async () => { writes += 1; } },
    },
  };
  await assert.rejects(() => renameTeam({ teamId: 'team-1', name: 'Safe name' }, services));
  assert.equal(writes, 0);
});
""",
        verifier={
            "mode": "authorize_mutation",
            "auth_call": "await auth(",
            "authorization_call": "db.membership.findFirst",
            "mutation_call": "db.team.update",
            "authorization_terms": ["teamId", "session.user.id", "OWNER", "ADMIN"],
            "validation_terms": ["name.length < 2", "name.length > 80"],
        },
    ),
    _auth_fixture(
        fixture_id="vr-dev-unauthenticated-direct-action",
        split="discovery",
        title="Protect a directly invoked publish action",
        symptom="A caller can publish another author's draft without a session.",
        source_name="posts",
        function_name="publishPost",
        base_source="""'use server';

export async function publishPost(input, { db }) {
  const postId = String(input?.postId || '').trim();
  if (!postId) throw new Error('Post required');
  return db.post.update({ where: { id: postId }, data: { published: true } });
}
""",
        gold_source="""'use server';

export async function publishPost(input, { auth, db }) {
  const postId = String(input?.postId || '').trim();
  if (!postId) throw new Error('Post required');
  const session = await auth();
  if (!session?.user?.id) throw new Error('Authentication required');
  const post = await db.post.findFirst({ where: { id: postId, authorId: session.user.id } });
  if (!post) throw new Error('Not authorized');
  return db.post.update({ where: { id: postId }, data: { published: true } });
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { publishPost } from '../app/posts.mjs';

test('vr-dev-unauthenticated-direct-action reproduces and repairs the reported behavior', async () => {
  let writes = 0;
  const services = {
    auth: async () => null,
    db: {
      post: {
        findFirst: async () => null,
        update: async () => { writes += 1; },
      },
    },
  };
  await assert.rejects(() => publishPost({ postId: 'post-1' }, services));
  assert.equal(writes, 0);
});
""",
        verifier={
            "mode": "authorize_mutation",
            "auth_call": "await auth(",
            "authorization_call": "db.post.findFirst",
            "mutation_call": "db.post.update",
            "authorization_terms": ["postId", "session.user.id", "authorId"],
            "validation_terms": ["if (!postId)"],
        },
    ),
    _rsc_fixture(
        fixture_id="vr-dev-rsc-primitive-derived-array",
        split="discovery",
        title="Stop serializing a duplicate primitive names array",
        symptom="The server sends both users and a names array derived from the same users.",
        server_source="""export function buildUserProps(users) {
  const names = users.map((user) => user.name);
  return { users, names };
}
""",
        server_gold="""export function buildUserProps(users) {
  return { users };
}
""",
        client_source="""export function renderUserNames({ names }) {
  return names.join(', ');
}
""",
        client_gold="""export function renderUserNames({ users }) {
  const names = users.map((user) => user.name);
  return names.join(', ');
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { buildUserProps } from '../app/server.mjs';
import { renderUserNames } from '../app/client.mjs';

test('vr-dev-rsc-primitive-derived-array reproduces and repairs the reported behavior', () => {
  const users = [{ id: 'u1', name: 'Ada' }, { id: 'u2', name: 'Grace' }];
  const props = buildUserProps(users);
  assert.deepEqual(Object.keys(props), ['users']);
  assert.equal(renderUserNames(props), 'Ada, Grace');
});
""",
        verifier={
            "mode": "canonical_only",
            "canonical_prop": "users",
            "derived_props": ["names"],
            "client_terms": ["users.map", "user.name"],
        },
    ),
    _rsc_fixture(
        fixture_id="vr-dev-rsc-object-filtered-array",
        split="discovery",
        title="Move a filtered object collection behind the client boundary",
        symptom="The RSC payload contains projects and a second filtered array of the same objects.",
        server_source="""export function buildProjectProps(projects) {
  const activeProjects = projects.filter((project) => project.active);
  return { projects, activeProjects };
}
""",
        server_gold="""export function buildProjectProps(projects) {
  return { projects };
}
""",
        client_source="""export function activeProjectNames({ activeProjects }) {
  return activeProjects.map((project) => project.name);
}
""",
        client_gold="""export function activeProjectNames({ projects }) {
  const activeProjects = projects.filter((project) => project.active);
  return activeProjects.map((project) => project.name);
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { buildProjectProps } from '../app/server.mjs';
import { activeProjectNames } from '../app/client.mjs';

test('vr-dev-rsc-object-filtered-array reproduces and repairs the reported behavior', () => {
  const projects = [{ name: 'A', active: true }, { name: 'B', active: false }];
  const props = buildProjectProps(projects);
  assert.deepEqual(Object.keys(props), ['projects']);
  assert.deepEqual(activeProjectNames(props), ['A']);
});
""",
        verifier={
            "mode": "canonical_only",
            "canonical_prop": "projects",
            "derived_props": ["activeProjects"],
            "client_terms": ["projects.filter", "project.active"],
        },
    ),
    _single_fixture(
        fixture_id="vr-dev-layout-interleaved-measurement",
        split="discovery",
        family="dom-batching-control",
        title="Remove repeated forced layout from panel sizing",
        symptom="Panel sizing alternates style writes and layout reads, causing two forced layouts.",
        source_path="lib/panel.mjs",
        base_source="""export function sizePanel(element) {
  element.style.width = '320px';
  const width = element.offsetWidth;
  element.style.height = '180px';
  const height = element.offsetHeight;
  return { width, height };
}
""",
        gold_source="""export function sizePanel(element) {
  element.style.width = '320px';
  element.style.height = '180px';
  const { width, height } = element.getBoundingClientRect();
  return { width, height };
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { sizePanel } from '../lib/panel.mjs';

test('vr-dev-layout-interleaved-measurement reproduces and repairs the reported behavior', () => {
  const operations = [];
  const style = {};
  Object.defineProperties(style, {
    width: { set: (value) => operations.push(['write', 'width', value]) },
    height: { set: (value) => operations.push(['write', 'height', value]) },
  });
  const element = {
    style,
    get offsetWidth() { operations.push(['read', 'width']); return 320; },
    get offsetHeight() { operations.push(['read', 'height']); return 180; },
    getBoundingClientRect() { operations.push(['read', 'rect']); return { width: 320, height: 180 }; },
  };
  assert.deepEqual(sizePanel(element), { width: 320, height: 180 });
  assert.deepEqual(operations.map((item) => item[0]), ['write', 'write', 'read']);
});
""",
        verifier={
            "kind": "dom_batch",
            "write_terms": ["style.width", "style.height"],
            "read_terms": ["getBoundingClientRect"],
            "forbidden_read_terms": ["offsetWidth", "offsetHeight"],
        },
    ),
    _single_fixture(
        fixture_id="vr-dev-large-array-max-rangeerror",
        split="discovery",
        family="large-array-control",
        title="Make maximum latency safe for production-size arrays",
        symptom="Maximum latency throws for a production array that exceeds the engine argument limit.",
        source_path="lib/latency.mjs",
        base_source="""export function maxLatency(values) {
  if (values.length === 0) return null;
  return Math.max(...values);
}
""",
        gold_source="""export function maxLatency(values) {
  if (values.length === 0) return null;
  let maximum = values[0];
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > maximum) maximum = values[index];
  }
  return maximum;
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { maxLatency } from '../lib/latency.mjs';

test('vr-dev-large-array-max-rangeerror reproduces and repairs the reported behavior', () => {
  const values = Array.from({ length: 300000 }, (_, index) => index % 10007);
  assert.equal(maxLatency(values), 10006);
  assert.equal(maxLatency([]), null);
});
""",
        verifier={"kind": "array_extreme", "mode": "max"},
    ),
    _single_fixture(
        fixture_id="vr-dev-use-latest-layout-read",
        split="discovery",
        family="hook-timing-control",
        title="Expose the latest callback before child layout effects run",
        symptom="A child layout effect observes the previous callback for one commit.",
        source_path="hooks/use-latest.mjs",
        base_source="""export function useLatest(value, hooks) {
  const ref = hooks.useRef(value);
  hooks.useEffect(() => { ref.current = value; }, [value]);
  return ref;
}
""",
        gold_source="""export function useLatest(value, hooks) {
  const ref = hooks.useRef(value);
  hooks.useLayoutEffect(() => { ref.current = value; }, [value]);
  return ref;
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { useLatest } from '../hooks/use-latest.mjs';

test('vr-dev-use-latest-layout-read reproduces and repairs the reported behavior', () => {
  const ref = { current: 'old' };
  const layout = [];
  const passive = [];
  const hooks = {
    useRef: () => ref,
    useLayoutEffect: (callback) => layout.push(callback),
    useEffect: (callback) => passive.push(callback),
  };
  useLatest('new', hooks);
  layout.forEach((callback) => callback());
  assert.equal(ref.current, 'new');
});
""",
        verifier={"kind": "hook_timing", "effect": "useLayoutEffect"},
    ),
    _single_fixture(
        fixture_id="vr-dev-window-event-handler-typing",
        split="discovery",
        family="event-signature-control",
        title="Use the concrete custom-event contract in a window handler",
        symptom="The window listener loses the payload because its handler assumes a generic event shape.",
        source_path="lib/window-filter.mjs",
        base_source="""/** @param {Event} event */
export function readFilterChange(event) {
  return event.value ?? null;
}
""",
        gold_source="""/** @param {CustomEvent} event */
export function readFilterChange(event) {
  return event.detail?.value ?? null;
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { readFilterChange } from '../lib/window-filter.mjs';

test('vr-dev-window-event-handler-typing reproduces and repairs the reported behavior', () => {
  assert.equal(readFilterChange({ detail: { value: 'critical' }, value: 'wrong' }), 'critical');
});
""",
        verifier={"kind": "event_signature", "event_type": "CustomEvent", "property": "detail"},
    ),
    _auth_fixture(
        fixture_id="vr-holdout-cross-tenant-delete-action",
        split="holdout",
        title="Block cross-tenant document deletion",
        symptom="A signed-in user can delete a document from a different tenant by calling the action directly.",
        source_name="documents",
        function_name="deleteDocument",
        base_source="""'use server';

export async function deleteDocument(input, { auth, db }) {
  const session = await auth();
  if (!session?.user?.id) throw new Error('Authentication required');
  const documentId = String(input?.documentId || '').trim();
  if (!documentId) throw new Error('Document required');
  return db.document.delete({ where: { id: documentId } });
}
""",
        gold_source="""'use server';

export async function deleteDocument(input, { auth, db }) {
  const session = await auth();
  if (!session?.user?.id) throw new Error('Authentication required');
  const documentId = String(input?.documentId || '').trim();
  if (!documentId) throw new Error('Document required');
  const document = await db.document.findFirst({
    where: { id: documentId, tenant: { members: { some: { userId: session.user.id } } } },
  });
  if (!document) throw new Error('Not authorized');
  return db.document.delete({ where: { id: documentId } });
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { deleteDocument } from '../app/documents.mjs';

test('vr-holdout-cross-tenant-delete-action reproduces and repairs the reported behavior', async () => {
  let deletes = 0;
  const services = {
    auth: async () => ({ user: { id: 'user-1' } }),
    db: {
      document: {
        findFirst: async () => null,
        delete: async () => { deletes += 1; },
      },
    },
  };
  await assert.rejects(() => deleteDocument({ documentId: 'foreign' }, services));
  assert.equal(deletes, 0);
});
""",
        verifier={
            "mode": "authorize_mutation",
            "auth_call": "await auth(",
            "authorization_call": "db.document.findFirst",
            "mutation_call": "db.document.delete",
            "authorization_terms": ["documentId", "session.user.id", "tenant", "members"],
            "validation_terms": ["if (!documentId)"],
        },
    ),
    _auth_fixture(
        fixture_id="vr-holdout-admin-or-owner-action",
        split="holdout",
        title="Retain both administrator and owner access to invitations",
        symptom="Team administrators receive an authorization error even though policy permits admins and owners.",
        source_name="invitations",
        function_name="inviteMember",
        base_source="""'use server';

export async function inviteMember(input, { auth, db }) {
  const session = await auth();
  if (!session?.user?.id) throw new Error('Authentication required');
  const membership = await db.membership.findFirst({ where: { teamId: input.teamId, userId: session.user.id } });
  if (!membership || membership.role !== 'OWNER') throw new Error('Not authorized');
  return db.invitation.create({ data: { teamId: input.teamId, email: input.email } });
}
""",
        gold_source="""'use server';

export async function inviteMember(input, { auth, db }) {
  const session = await auth();
  if (!session?.user?.id) throw new Error('Authentication required');
  const teamId = String(input?.teamId || '').trim();
  const email = String(input?.email || '').trim();
  if (!teamId || !email.includes('@')) throw new Error('Invalid invitation');
  const membership = await db.membership.findFirst({ where: { teamId, userId: session.user.id } });
  if (!membership || !['OWNER', 'ADMIN'].includes(membership.role)) throw new Error('Not authorized');
  return db.invitation.create({ data: { teamId, email } });
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { inviteMember } from '../app/invitations.mjs';

test('vr-holdout-admin-or-owner-action reproduces and repairs the reported behavior', async () => {
  let writes = 0;
  const services = {
    auth: async () => ({ user: { id: 'admin' } }),
    db: {
      membership: { findFirst: async () => ({ role: 'ADMIN' }) },
      invitation: { create: async () => { writes += 1; return { id: 'invite-1' }; } },
    },
  };
  await inviteMember({ teamId: 'team-1', email: 'new@example.com' }, services);
  assert.equal(writes, 1);
});
""",
        verifier={
            "mode": "authorize_mutation",
            "auth_call": "await auth(",
            "authorization_call": "db.membership.findFirst",
            "mutation_call": "db.invitation.create",
            "authorization_terms": ["teamId", "session.user.id", "OWNER", "ADMIN"],
            "validation_terms": ["email.includes", "Invalid invitation"],
        },
    ),
    _auth_fixture(
        fixture_id="vr-holdout-action-validation-order",
        split="holdout",
        title="Validate and authorize a refund before mutating it",
        symptom="An invalid refund request records a mutation before validation rejects the request.",
        source_name="refunds",
        function_name="approveRefund",
        base_source="""'use server';

export async function approveRefund(input, { auth, db }) {
  const session = await auth();
  if (!session?.user?.id) throw new Error('Authentication required');
  const updated = await db.refund.update({ where: { id: input.refundId }, data: { approved: true } });
  const amount = Number(input?.amount);
  if (!Number.isFinite(amount) || amount <= 0) throw new Error('Invalid amount');
  return updated;
}
""",
        gold_source="""'use server';

export async function approveRefund(input, { auth, db }) {
  const refundId = String(input?.refundId || '').trim();
  const amount = Number(input?.amount);
  if (!refundId || !Number.isFinite(amount) || amount <= 0) throw new Error('Invalid amount');
  const session = await auth();
  if (!session?.user?.id) throw new Error('Authentication required');
  const reviewer = await db.membership.findFirst({ where: { userId: session.user.id, role: { in: ['OWNER', 'ADMIN'] } } });
  if (!reviewer) throw new Error('Not authorized');
  return db.refund.update({ where: { id: refundId }, data: { approved: true, amount } });
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { approveRefund } from '../app/refunds.mjs';

test('vr-holdout-action-validation-order reproduces and repairs the reported behavior', async () => {
  let writes = 0;
  const services = {
    auth: async () => ({ user: { id: 'owner' } }),
    db: {
      membership: { findFirst: async () => ({ role: 'OWNER' }) },
      refund: { update: async () => { writes += 1; } },
    },
  };
  await assert.rejects(() => approveRefund({ refundId: 'r1', amount: -1 }, services));
  assert.equal(writes, 0);
});
""",
        verifier={
            "mode": "authorize_mutation",
            "auth_call": "await auth(",
            "authorization_call": "db.membership.findFirst",
            "mutation_call": "db.refund.update",
            "authorization_terms": ["session.user.id", "OWNER", "ADMIN"],
            "validation_terms": ["Number.isFinite", "amount <= 0"],
            "validation_before_auth": True,
        },
    ),
    _auth_fixture(
        fixture_id="vr-holdout-readonly-action-control",
        split="holdout",
        title="Keep a read-only loader free of accidental writes",
        symptom="Opening a dashboard unexpectedly creates an audit mutation from a read-only loader.",
        source_name="dashboard",
        function_name="loadDashboard",
        base_source="""export async function loadDashboard(userId, { db }) {
  const dashboard = await db.dashboard.findFirst({ where: { userId } });
  await db.audit.create({ data: { userId, event: 'dashboard-opened' } });
  return dashboard;
}
""",
        gold_source="""export async function loadDashboard(userId, { db }) {
  if (!userId) throw new Error('User required');
  return db.dashboard.findFirst({ where: { userId } });
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { loadDashboard } from '../app/dashboard.mjs';

test('vr-holdout-readonly-action-control reproduces and repairs the reported behavior', async () => {
  let writes = 0;
  const services = {
    db: {
      dashboard: { findFirst: async () => ({ title: 'Overview' }) },
      audit: { create: async () => { writes += 1; } },
    },
  };
  assert.deepEqual(await loadDashboard('u1', services), { title: 'Overview' });
  assert.equal(writes, 0);
});
""",
        verifier={
            "mode": "read_only_control",
            "read_call": "db.dashboard.findFirst",
            "forbidden_calls": ["db.audit.create", "await auth("],
            "validation_terms": ["if (!userId)"],
        },
    ),
    _rsc_fixture(
        fixture_id="vr-holdout-rsc-primitive-sort",
        split="holdout",
        title="Avoid sending both tags and a sorted tag copy",
        symptom="The server sends a primitive tag list and an independently serialized sorted copy.",
        server_source="""export function buildTagProps(tags) {
  const sortedTags = [...tags].sort();
  return { tags, sortedTags };
}
""",
        server_gold="""export function buildTagProps(tags) {
  return { tags };
}
""",
        client_source="""export function displayTags({ sortedTags }) {
  return sortedTags.join(' | ');
}
""",
        client_gold="""export function displayTags({ tags }) {
  const sortedTags = [...tags].sort();
  return sortedTags.join(' | ');
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { buildTagProps } from '../app/server.mjs';
import { displayTags } from '../app/client.mjs';

test('vr-holdout-rsc-primitive-sort reproduces and repairs the reported behavior', () => {
  const props = buildTagProps(['zeta', 'alpha']);
  assert.deepEqual(Object.keys(props), ['tags']);
  assert.equal(displayTags(props), 'alpha | zeta');
});
""",
        verifier={
            "mode": "canonical_only",
            "canonical_prop": "tags",
            "derived_props": ["sortedTags"],
            "client_terms": ["[...tags]", ".sort("],
        },
    ),
    _rsc_fixture(
        fixture_id="vr-holdout-rsc-object-map-clone",
        split="holdout",
        title="Remove cloned user objects from the server payload",
        symptom="The server sends users and a second array of cloned user records.",
        server_source="""export function buildUserCardProps(users) {
  const userCards = users.map((user) => ({ ...user, label: user.name.toUpperCase() }));
  return { users, userCards };
}
""",
        server_gold="""export function buildUserCardProps(users) {
  return { users };
}
""",
        client_source="""export function userCardLabels({ userCards }) {
  return userCards.map((user) => user.label);
}
""",
        client_gold="""export function userCardLabels({ users }) {
  const userCards = users.map((user) => ({ ...user, label: user.name.toUpperCase() }));
  return userCards.map((user) => user.label);
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { buildUserCardProps } from '../app/server.mjs';
import { userCardLabels } from '../app/client.mjs';

test('vr-holdout-rsc-object-map-clone reproduces and repairs the reported behavior', () => {
  const props = buildUserCardProps([{ id: 'u1', name: 'Ada' }]);
  assert.deepEqual(Object.keys(props), ['users']);
  assert.deepEqual(userCardLabels(props), ['ADA']);
});
""",
        verifier={
            "mode": "canonical_only",
            "canonical_prop": "users",
            "derived_props": ["userCards"],
            "client_terms": ["users.map", "toUpperCase"],
        },
    ),
    _rsc_fixture(
        fixture_id="vr-holdout-rsc-derived-scalar",
        split="holdout",
        title="Derive a display label from the canonical product",
        symptom="The RSC payload sends a product and a display label derived from that product.",
        server_source="""export function buildProductProps(product) {
  const displayLabel = `${product.name} — ${product.sku}`;
  return { product, displayLabel };
}
""",
        server_gold="""export function buildProductProps(product) {
  return { product };
}
""",
        client_source="""export function productLabel({ displayLabel }) {
  return displayLabel;
}
""",
        client_gold="""export function productLabel({ product }) {
  return `${product.name} — ${product.sku}`;
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { buildProductProps } from '../app/server.mjs';
import { productLabel } from '../app/client.mjs';

test('vr-holdout-rsc-derived-scalar reproduces and repairs the reported behavior', () => {
  const props = buildProductProps({ name: 'Relay', sku: 'R-2' });
  assert.deepEqual(Object.keys(props), ['product']);
  assert.equal(productLabel(props), 'Relay — R-2');
});
""",
        verifier={
            "mode": "canonical_only",
            "canonical_prop": "product",
            "derived_props": ["displayLabel"],
            "client_terms": ["product.name", "product.sku"],
        },
    ),
    _rsc_fixture(
        fixture_id="vr-holdout-rsc-expensive-derivation-control",
        split="holdout",
        title="Repair a server-only histogram without moving it to the client",
        symptom="A server-computed histogram drops values that land on the highest bucket boundary.",
        server_source="""export function buildHistogramProps(values) {
  const histogram = [0, 0, 0];
  for (const value of values) {
    const bucket = Math.min(1, Math.floor(value / 10));
    histogram[bucket] += 1;
  }
  return { histogram };
}
""",
        server_gold="""export function buildHistogramProps(values) {
  const histogram = [0, 0, 0];
  for (const value of values) {
    const bucket = Math.min(2, Math.floor(value / 10));
    histogram[bucket] += 1;
  }
  return { histogram };
}
""",
        client_source="""export function histogramTotal({ histogram }) {
  return histogram.reduce((total, count) => total + count, 0);
}
""",
        client_gold="""export function histogramTotal({ histogram }) {
  return histogram.reduce((total, count) => total + count, 0);
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { buildHistogramProps } from '../app/server.mjs';
import { histogramTotal } from '../app/client.mjs';

test('vr-holdout-rsc-expensive-derivation-control reproduces and repairs the reported behavior', () => {
  const props = buildHistogramProps([1, 11, 29]);
  assert.deepEqual(props.histogram, [1, 1, 1]);
  assert.equal(histogramTotal(props), 3);
});
""",
        verifier={
            "mode": "derived_control",
            "canonical_prop": "histogram",
            "forbidden_props": ["values"],
            "server_terms": ["Math.min(2", "Math.floor", "return { histogram }"],
            "client_terms": ["histogram.reduce"],
        },
    ),
    _single_fixture(
        fixture_id="vr-holdout-layout-two-write-two-read",
        split="holdout",
        family="dom-batching-control",
        title="Batch card position writes before measuring",
        symptom="Card placement reads layout after each of two style writes.",
        source_path="lib/card.mjs",
        base_source="""export function positionCard(element) {
  element.style.left = '24px';
  const left = element.offsetLeft;
  element.style.top = '32px';
  const top = element.offsetTop;
  return { left, top };
}
""",
        gold_source="""export function positionCard(element) {
  element.style.left = '24px';
  element.style.top = '32px';
  const { left, top } = element.getBoundingClientRect();
  return { left, top };
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { positionCard } from '../lib/card.mjs';

test('vr-holdout-layout-two-write-two-read reproduces and repairs the reported behavior', () => {
  const operations = [];
  const style = {};
  Object.defineProperties(style, {
    left: { set: () => operations.push('write') },
    top: { set: () => operations.push('write') },
  });
  const element = {
    style,
    get offsetLeft() { operations.push('read'); return 24; },
    get offsetTop() { operations.push('read'); return 32; },
    getBoundingClientRect() { operations.push('read'); return { left: 24, top: 32 }; },
  };
  assert.deepEqual(positionCard(element), { left: 24, top: 32 });
  assert.deepEqual(operations, ['write', 'write', 'read']);
});
""",
        verifier={
            "kind": "dom_batch",
            "write_terms": ["style.left", "style.top"],
            "read_terms": ["getBoundingClientRect"],
            "forbidden_read_terms": ["offsetLeft", "offsetTop"],
        },
    ),
    _single_fixture(
        fixture_id="vr-holdout-layout-write-only-control",
        split="holdout",
        family="dom-batching-control",
        title="Correct a write-only banner size",
        symptom="The banner uses the wrong width but performs no layout measurement.",
        source_path="lib/banner.mjs",
        base_source="""export function styleBanner(element) {
  element.style.width = '64px';
  element.style.height = '48px';
}
""",
        gold_source="""export function styleBanner(element) {
  element.style.width = '640px';
  element.style.height = '48px';
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { styleBanner } from '../lib/banner.mjs';

test('vr-holdout-layout-write-only-control reproduces and repairs the reported behavior', () => {
  const element = { style: {} };
  styleBanner(element);
  assert.deepEqual(element.style, { width: '640px', height: '48px' });
});
""",
        verifier={
            "kind": "dom_write_control",
            "write_terms": ["style.width", "style.height", "'640px'", "'48px'"],
            "forbidden_read_terms": ["offsetWidth", "offsetHeight", "getBoundingClientRect", "getComputedStyle"],
        },
    ),
    _single_fixture(
        fixture_id="vr-holdout-large-array-min",
        split="holdout",
        family="large-array-control",
        title="Make minimum duration safe for large arrays",
        symptom="Minimum duration fails when the collection exceeds the JavaScript argument limit.",
        source_path="lib/duration.mjs",
        base_source="""export function minDuration(values) {
  if (values.length === 0) return null;
  return Math.min(...values);
}
""",
        gold_source="""export function minDuration(values) {
  if (values.length === 0) return null;
  let minimum = values[0];
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] < minimum) minimum = values[index];
  }
  return minimum;
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { minDuration } from '../lib/duration.mjs';

test('vr-holdout-large-array-min reproduces and repairs the reported behavior', () => {
  const values = Array.from({ length: 300000 }, (_, index) => 500000 - index);
  assert.equal(minDuration(values), 200001);
  assert.equal(minDuration([]), null);
});
""",
        verifier={"kind": "array_extreme", "mode": "min"},
    ),
    _single_fixture(
        fixture_id="vr-holdout-large-array-sum-control",
        split="holdout",
        family="large-array-control",
        title="Return zero for an empty duration total",
        symptom="Summing an empty duration collection throws instead of returning zero.",
        source_path="lib/total.mjs",
        base_source="""export function totalDuration(values) {
  return values.reduce((total, value) => total + value);
}
""",
        gold_source="""export function totalDuration(values) {
  return values.reduce((total, value) => total + value, 0);
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { totalDuration } from '../lib/total.mjs';

test('vr-holdout-large-array-sum-control reproduces and repairs the reported behavior', () => {
  assert.equal(totalDuration([]), 0);
  assert.equal(totalDuration([4, 8, 15]), 27);
});
""",
        verifier={"kind": "array_sum_control"},
    ),
    _single_fixture(
        fixture_id="vr-holdout-use-latest-child-layout-effect",
        split="holdout",
        family="hook-timing-control",
        title="Refresh a ref before descendant layout work",
        symptom="A descendant layout effect calls the callback from the preceding commit.",
        source_path="hooks/use-current.mjs",
        base_source="""export function useCurrent(value, hooks) {
  const ref = hooks.useRef(value);
  hooks.useEffect(() => { ref.current = value; }, [value]);
  return ref;
}
""",
        gold_source="""export function useCurrent(value, hooks) {
  const ref = hooks.useRef(value);
  hooks.useLayoutEffect(() => { ref.current = value; }, [value]);
  return ref;
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { useCurrent } from '../hooks/use-current.mjs';

test('vr-holdout-use-latest-child-layout-effect reproduces and repairs the reported behavior', () => {
  const ref = { current: () => 'old' };
  const layout = [];
  const passive = [];
  const hooks = {
    useRef: () => ref,
    useLayoutEffect: (callback) => layout.push(callback),
    useEffect: (callback) => passive.push(callback),
  };
  useCurrent(() => 'new', hooks);
  layout.forEach((callback) => callback());
  assert.equal(ref.current(), 'new');
});
""",
        verifier={"kind": "hook_timing", "effect": "useLayoutEffect"},
    ),
    _single_fixture(
        fixture_id="vr-holdout-use-latest-passive-only-control",
        split="holdout",
        family="hook-timing-control",
        title="Refresh a ref for passive subscription work",
        symptom="A passive subscription callback never receives the latest formatter.",
        source_path="hooks/use-passive-current.mjs",
        base_source="""export function usePassiveCurrent(value, hooks) {
  return hooks.useRef(value);
}
""",
        gold_source="""export function usePassiveCurrent(value, hooks) {
  const ref = hooks.useRef(value);
  hooks.useEffect(() => { ref.current = value; }, [value]);
  return ref;
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { usePassiveCurrent } from '../hooks/use-passive-current.mjs';

test('vr-holdout-use-latest-passive-only-control reproduces and repairs the reported behavior', () => {
  const ref = { current: 'old' };
  const passive = [];
  const hooks = { useRef: () => ref, useEffect: (callback) => passive.push(callback) };
  usePassiveCurrent('new', hooks);
  passive.forEach((callback) => callback());
  assert.equal(ref.current, 'new');
});
""",
        verifier={"kind": "hook_timing", "effect": "useEffect", "forbid_layout": True},
    ),
    _single_fixture(
        fixture_id="vr-holdout-mouseevent-handler",
        split="holdout",
        family="event-signature-control",
        title="Use viewport coordinates from a pointer handler",
        symptom="The pointer handler reports page coordinates when the caller requires viewport coordinates.",
        source_path="lib/pointer.mjs",
        base_source="""/** @param {Event} event */
export function pointerX(event) {
  return event.pageX;
}
""",
        gold_source="""/** @param {MouseEvent} event */
export function pointerX(event) {
  return event.clientX;
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { pointerX } from '../lib/pointer.mjs';

test('vr-holdout-mouseevent-handler reproduces and repairs the reported behavior', () => {
  assert.equal(pointerX({ clientX: 12, pageX: 712 }), 12);
});
""",
        verifier={"kind": "event_signature", "event_type": "MouseEvent", "property": "clientX"},
    ),
    _single_fixture(
        fixture_id="vr-holdout-keyboardevent-handler",
        split="holdout",
        family="event-signature-control",
        title="Return the semantic key from a keyboard handler",
        symptom="Keyboard shortcuts receive a legacy numeric key code instead of the semantic key.",
        source_path="lib/keyboard.mjs",
        base_source="""/** @param {Event} event */
export function shortcutKey(event) {
  return event.keyCode;
}
""",
        gold_source="""/** @param {KeyboardEvent} event */
export function shortcutKey(event) {
  return event.key;
}
""",
        test_source="""import test from 'node:test';
import assert from 'node:assert/strict';
import { shortcutKey } from '../lib/keyboard.mjs';

test('vr-holdout-keyboardevent-handler reproduces and repairs the reported behavior', () => {
  assert.equal(shortcutKey({ key: 'Escape', keyCode: 27 }), 'Escape');
});
""",
        verifier={"kind": "event_signature", "event_type": "KeyboardEvent", "property": "key"},
    ),
)


def fixture_by_id() -> dict[str, dict[str, Any]]:
    """Return the exact catalog keyed by stable task identity."""

    return {str(item["id"]): item for item in FIXTURES}
