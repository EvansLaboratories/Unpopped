# Reading the carved history

Unpopped's crates were not written here. They were carved out of the
[Baracuda](https://github.com/ciresnave/baracuda) workspace, where they were
developed as `baracuda-kernelgen` and `baracuda-kernel-vocab`. The extraction
preserved their commit history, but *how* git exposes that history is
non-obvious. This page is the map.

## What survived the carve

Each crate arrived via `git subtree add`, which brings the original commits with
their **messages, authors, and dates** intact.

`git blame` works normally and attributes to the original authors and dates:

```console
$ git blame -L 60,64 --date=short -- crates/unpopped-vocab/src/element.rs
^1d561a5 src/element.rs (Eric Evans 2026-07-10 60) ...
```

Note the path in the blame output is the *pre-carve* path (`src/element.rs`),
because that is where the file lived when the commit was made.

## The gotcha: path-filtered log stops at the merge

This does **not** show the crate's history:

```console
$ git log --oneline -- crates/unpopped-vocab
fc70474 Add 'crates/unpopped-vocab/' from commit '310cdc39...'
```

One commit, not twenty-four. The historical commits hold the files at the
repository root (that is what `git subtree split` produces), not at
`crates/unpopped-vocab/`, so a path filter on the new path does not traverse
past the merge. `git log --follow` does not rescue this either.

**Use the merge commit's second parent instead** — that is the carved branch:

```console
$ git log --oneline <merge-commit>^2
```

To find the merge commit for a given crate:

```console
$ git log --format=%h --grep="Add 'crates/<crate-name>/'" -1
```

Or in one step:

```console
$ git log --oneline "$(git log --format=%h --grep="Add 'crates/unpopped-vocab/'" -1)^2"
```

## Commit SHAs do not match Baracuda's

Path extraction rewrites commits — the carved commits contain a different tree
(files at root rather than nested under `crates/...`), so they hash differently.
The oldest carved `unpopped-vocab` commit is `1d561a59`; the same change in
Baracuda is `7d612cd0`.

This is inherent to extracting a subdirectory into its own repository — it is
not an artifact of choosing `git subtree` over `git filter-repo`, which rewrites
SHAs the same way.

**Consequence:** never cross-reference between the two repositories by SHA.
Refer to commits by message and date, which are stable across the carve.
