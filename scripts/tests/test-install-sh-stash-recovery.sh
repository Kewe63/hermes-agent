#!/usr/bin/env bash
#
# Regression tests for the install.sh stash-recovery logic (PR #47373 / #46791).
#
# Covers the two failure modes teknium flagged in review:
#   1. A SHIFTED stash selector -- another `git stash push` between capture and
#      apply must NOT make apply/drop target a different entry. We capture the
#      immutable object ID (refs/stash SHA) right after push and resolve a
#      selector from that SHA only when dropping.
#   2. A CONFLICTING apply -- a stash apply that hits conflict markers must
#      leave the working tree reset to a clean HEAD while the original stash
#      is retained (parity with hermes_cli/main.py recovery).
#
# This test re-implements the two helpers it exercises (resolve_stash_selector
# and the SHA-capture idiom) as they appear in scripts/install.sh, so it can
# run without sourcing the full installer.

set -u

PASS=0
FAIL=0

assert() {
    local desc="$1"
    local cond="$2"
    if eval "$cond"; then
        echo "  ok   - $desc"
        PASS=$((PASS + 1))
    else
        echo "  FAIL - $desc"
        FAIL=$((FAIL + 1))
    fi
}

# Mirror of scripts/install.sh:resolve_stash_selector.
resolve_stash_selector() {
    local stash_sha="$1"
    [ -z "$stash_sha" ] && { echo ""; return; }
    git stash list --format='%gd %H' 2>/dev/null | while IFS=' ' read -r selector commit; do
        if [ "$commit" = "$stash_sha" ]; then
            echo "$selector"
            return
        fi
    done
}

make_repo() {
    local d="$1"
    mkdir -p "$d"
    git -C "$d" init -q
    git -C "$d" config user.email "test@example.com"
    git -C "$d" config user.name "Test"
    printf "base\n" > "$d/file.txt"
    git -C "$d" add file.txt
    git -C "$d" commit -qm "init"
}

echo "== install.sh stash recovery tests =="

# ---------------------------------------------------------------------------
# Scenario 1: shifted selector -- capture immutable SHA, then push a second
# stash so the mutable selector shifts; assert apply/drop still hit our entry.
# ---------------------------------------------------------------------------
TMP1="$(mktemp -d)"
make_repo "$TMP1"

printf "local edit A\n" >> "$TMP1/file.txt"
( cd "$TMP1" && git stash push --include-untracked -m "autostash-A" >/dev/null 2>&1 )
AUTOSTASH_SHA="$(cd "$TMP1" && git rev-parse --verify refs/stash)"
assert "autostash SHA captured after first push" "[ -n \"$AUTOSTASH_SHA\" ]"

# Shift the list: push a SECOND, unrelated stash. The mutable selector for our
# entry moves, but the SHA is unchanged.
printf "unrelated\n" > "$TMP1/other.txt"
( cd "$TMP1" && git stash push --include-untracked -m "interloper" >/dev/null 2>&1 )
SHA_AT_TOP="$(cd "$TMP1" && git rev-parse --verify refs/stash)"
assert "interloper became top of list (selector shifted)" "[ \"$SHA_AT_TOP\" != \"$AUTOSTASH_SHA\" ]"

SEL="$(cd "$TMP1" && resolve_stash_selector "$AUTOSTASH_SHA")"
assert "resolve_stash_selector finds a selector for the shifted SHA" "[ -n \"$SEL\" ]"

( cd "$TMP1" && git stash drop "$SEL" >/dev/null 2>&1 )
if cd "$TMP1" && git stash list --format='%H' | grep -q "$AUTOSTASH_SHA"; then
    OUR_DROPPED="no"
else
    OUR_DROPPED="yes"
fi
assert "our shifted stash was dropped (SHA no longer in list)" "[ \"$OUR_DROPPED\" = \"yes\" ]"

# ---------------------------------------------------------------------------
# Scenario 2: conflicting apply -- stash a change, then make the working tree
# conflict so apply fails; simulate the recovery reset and assert the stash
# (and its SHA) is retained while the tree is clean.
# ---------------------------------------------------------------------------
TMP2="$(mktemp -d)"
make_repo "$TMP2"

printf "v1\n" > "$TMP2/shared.txt"
( cd "$TMP2" && git add shared.txt && git commit -qm "add shared" )

printf "v2-local\n" > "$TMP2/shared.txt"
( cd "$TMP2" && git stash push --include-untracked -m "conflict-src" >/dev/null 2>&1 )
CONFLICT_SHA="$(cd "$TMP2" && git rev-parse --verify refs/stash)"
assert "conflict-source stash SHA captured" "[ -n \"$CONFLICT_SHA\" ]"

# Upstream version diverges from the stashed edit; apply would conflict.
printf "v2-upstream\n" > "$TMP2/shared.txt"
( cd "$TMP2" && git add shared.txt )
if ! ( cd "$TMP2" && git stash apply "$CONFLICT_SHA" >/dev/null 2>&1 ); then
    ( cd "$TMP2" && git reset --hard HEAD >/dev/null 2>&1 )
    ( cd "$TMP2" && git clean -fd >/dev/null 2>&1 )
fi

if cd "$TMP2" && git rev-parse --verify "$CONFLICT_SHA" >/dev/null 2>&1; then
    RETAINED="yes"
else
    RETAINED="no"
fi
assert "stash retained after conflicting apply + reset" "[ \"$RETAINED\" = \"yes\" ]"

UNMERGED="$(cd "$TMP2" && git ls-files --unmerged | wc -l | tr -d ' ')"
assert "working tree has no unmerged paths after reset" "[ \"$UNMERGED\" = \"0\" ]"

STATUS_PORCELAIN="$(cd "$TMP2" && git status --porcelain | wc -l | tr -d ' ')"
assert "working tree is clean after reset" "[ \"$STATUS_PORCELAIN\" = \"0\" ]"

rm -rf "$TMP1" "$TMP2"

echo "----"
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
