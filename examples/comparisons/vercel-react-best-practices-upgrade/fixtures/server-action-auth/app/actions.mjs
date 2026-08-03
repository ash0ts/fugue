'use server';

import { auth, db } from '../lib/deps.mjs';

export async function renameTeam(input) {
  const session = await auth();
  if (!session?.user?.id) {
    throw new Error('Authentication required');
  }

  const teamId = String(input?.teamId || '').trim();
  const name = String(input?.name || '').trim();
  if (!teamId || name.length < 2 || name.length > 80) {
    throw new Error('Invalid team update');
  }

  return db.team.update({
    where: { id: teamId },
    data: { name },
  });
}
