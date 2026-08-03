import assert from 'node:assert/strict';
import test from 'node:test';

import { renameTeam } from '../app/actions.mjs';
import { resetFixture, updates } from '../lib/deps.mjs';

test('rejects a signed-in non-member without mutating state', async () => {
  resetFixture({ session: { user: { id: 'user-2' } } });

  await assert.rejects(
    renameTeam({ teamId: 'team-1', name: 'Renamed team' }),
    /authori[sz]ed|membership/i,
  );
  assert.deepEqual(updates(), []);
});

test('permits an owner and updates exactly the authorized team', async () => {
  resetFixture({
    session: { user: { id: 'user-1' } },
    memberships: [{ userId: 'user-1', teamId: 'team-1', role: 'OWNER' }],
  });

  const result = await renameTeam({ teamId: 'team-1', name: 'Renamed team' });
  assert.deepEqual(result, { id: 'team-1', name: 'Renamed team' });
  assert.deepEqual(updates(), [{ id: 'team-1', name: 'Renamed team' }]);
});

test('keeps authentication and authorization inside the exported action', async () => {
  resetFixture();
  await assert.rejects(
    renameTeam({ teamId: 'team-1', name: 'Renamed team' }),
    /authentication/i,
  );
  assert.deepEqual(updates(), []);
});
