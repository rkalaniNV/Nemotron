# Slurm Reference

Use this sample when jobs are launched through a Slurm login node and the cluster provides a shared filesystem.

Required fields:
- `container_image`
- `host`
- `user`
- `account`
- `partition`
- `remote_job_dir`

Common failures:
- wrong `partition`
- remote job directory not writable
- container image unavailable on the cluster
