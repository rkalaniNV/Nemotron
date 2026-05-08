# Nemotron Customizer Deploy Assets

This folder contains deployment support for Nemotron Customizer workflows.

For this repo, "Nemotron Customizer" means the generic step system under
`src/nemotron/steps/`. The assets here do not imply that unrelated recipes,
cookbooks, docs, or examples elsewhere in the repository are airgap-ready.

Available deploy flows:

- `airgap/` - build an offline bundle and Docker image for selected
  `src/nemotron/steps/` workflow targets.
