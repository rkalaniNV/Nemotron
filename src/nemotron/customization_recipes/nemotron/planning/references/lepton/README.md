# Lepton Reference

Use this sample when jobs run through DGX Cloud via Lepton and you already know the target node group and mounted storage path.

Required fields:
- `container_image` or `container`
- `node_group`

Optional fields:
- `resource_shape`

Notes:
- include at least one mount for model or data access

Common failures:
- wrong `node_group`
- missing mount for data
- container image tag mismatch
