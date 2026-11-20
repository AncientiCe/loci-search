# Releasing Loci Search

## First release

The planned first tag is `v0.1.0`. Wait for the release preparation to land on `main` and for the **CI gate** on that commit to pass. Run **Release** from `main` with version `0.1.0` and mode **check**. This mode can run while the repository is private. It tests the release binaries and crate package without creating a tag, release, or crate. Changes pushed to the release tools also trigger this check automatically.

At public launch, upload [the 1280 × 640 social preview](images/loci-search.png) in GitHub repository settings and enable private vulnerability reporting. Verify that the branch ruleset still requires a pull request and the **CI gate** status check and blocks force pushes and deletion on `main`. Verify that `v*` tags are protected against updates and deletion and that immutable releases are enabled.

After CI passes on the exact release commit on `main`, create and push the annotated version tag.

```sh
git tag -a v0.1.0 -m "Loci Search 0.1.0"
git push origin refs/tags/v0.1.0
```

The tag push runs **Release** automatically. It checks that the tag is annotated, matches the tagged `Cargo.toml`, and points at a commit on `main`. It tests and packages Windows and Linux x86-64 archives with SHA-256 checksums and publishes the GitHub release. The release notes are the tag's `## [X.Y.Z]` section of `CHANGELOG.md`, copied verbatim. If the automatic run fails after the tag exists, fix the cause on `main` and run **Release** from `main` in mode **draft** to rebuild the release as a draft for review.

Crates.io requires a first manual upload. From a clean checkout of the approved tag, verify the package with `cargo +1.95.0 publish --dry-run --locked`. Use a short-lived crates.io token for the first `cargo +1.95.0 publish --locked`, then revoke it. The crate name is assigned by crates.io on successful first publication. [Trusted Publishing announcement](https://blog.rust-lang.org/2025/07/11/crates-io-development-update-2025-07/#trusted-publishing)

## Subsequent releases

1. Bump the Cargo version and update `Cargo.lock`. Move the relevant notes from **Unreleased** to a new `## [X.Y.Z]` section at the top of the version history.
2. Merge after CI passes and push the annotated `vX.Y.Z` tag. The release publishes with that version's changelog section.
3. Run **Release** again from `main` with the same version and mode **publish-crate** to upload the crate.

Configure the `crates-io` GitHub environment to allow deployments only from `main`, with a required reviewer where the repository plan supports it. On the crate's settings page, configure a Trusted Publisher for owner `AncientiCe`, repository `loci-search`, workflow `release.yml`, and environment `crates-io`. The workflow obtains a short-lived token only after verifying the source, testing the crate, auditing dependencies, and confirming that the matching GitHub release is published. Enable Trusted Publishing-only mode on crates.io after the first successful trusted upload.
