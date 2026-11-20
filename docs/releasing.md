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

After the GitHub release is published, the same run uploads the crate to crates.io from the `crates-io` environment. The first upload needs a token because a Trusted Publisher can only be added to an existing crate. Before pushing the first tag, create a crates.io API token limited to `loci-search` with the `publish-new` and `publish-update` scopes and a short expiry, and save it as the `CARGO_REGISTRY_TOKEN` secret of the `crates-io` environment. After the first upload, delete the secret and revoke the token.

## Subsequent releases

1. Bump the Cargo version and update `Cargo.lock`. Move the relevant notes from **Unreleased** to a new `## [X.Y.Z]` section at the top of the version history.
2. Wait for CI to pass on `main`, then push the annotated `vX.Y.Z` tag. The run publishes the GitHub release with that version's changelog section and then uploads the crate.
3. If the crate upload fails after the GitHub release is out, run **Release** from `main` with the same version and mode **publish-crate**.

The `crates-io` GitHub environment allows deployments only from `main` and `v*` tags. Add a required reviewer where the repository plan supports it. On the crate's settings page, configure a Trusted Publisher for owner `AncientiCe`, repository `loci-search`, workflow `release.yml`, and environment `crates-io`. The workflow obtains a short-lived token only after verifying the source, testing the crate, auditing dependencies, and confirming that the matching GitHub release is published. Enable Trusted Publishing-only mode on crates.io after the first successful trusted upload.
