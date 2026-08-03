const state = {
  session: null,
  memberships: [],
  updates: [],
};

export function resetFixture({ session = null, memberships = [] } = {}) {
  state.session = session;
  state.memberships = memberships;
  state.updates = [];
}

export function updates() {
  return [...state.updates];
}

export async function auth() {
  return state.session;
}

export const db = {
  membership: {
    async findFirst({ where }) {
      return (
        state.memberships.find(
          (membership) =>
            membership.teamId === where.teamId &&
            membership.userId === where.userId &&
            where.role.in.includes(membership.role),
        ) || null
      );
    },
  },
  team: {
    async update({ where, data }) {
      const value = { id: where.id, name: data.name };
      state.updates.push(value);
      return value;
    },
  },
};
