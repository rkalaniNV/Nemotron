# Run:AI Reference

Use this sample when the platform team exposes Kubernetes-backed GPU jobs through Run:AI.

Required fields:
- `container_image`
- `cluster`
- `project`

Optional fields:
- `node_pool`

Notes:
- include PVC mount definitions if data is not baked into the image

Common failures:
- wrong project name
- PVC not mounted at the expected path
- node pool incompatible with requested GPU count
