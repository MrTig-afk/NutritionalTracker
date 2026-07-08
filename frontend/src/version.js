// Version of THIS build's changelog state. The update modal shows only
// changelog.v2.json entries with v > CHANGELOG_VERSION — i.e. exactly what the
// pending update adds relative to the build the user is currently running.
// RELEASE RULE: every user-visible release bumps this by 1 and prepends
// entries with the new number to public/changelog.v2.json, in the same commit.
export const CHANGELOG_VERSION = 11;
