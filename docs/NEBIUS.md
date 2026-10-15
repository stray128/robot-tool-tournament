# Running on Nebius

How this project uses Nebius AI Cloud and Token Factory, and what we observed while setting it up.

## Services used

| Service | Use |
|---|---|
| Token Factory | Nemotron 3 Super proposes finger designs (forced tool call). Nemotron 3.5 Lightning powers the conversational agent. OpenAI-compatible API at `https://api.tokenfactory.nebius.com/v1/`. |
| Serverless AI Jobs | Every scored simulation batch runs as a job from one pinned container image. |
| Object Storage | Job outputs (results, manifests, replays) are written to a bucket mounted at `/out`. |
| Container Registry | Hosts the simulation image. |
| MysteryBox | Holds the Token Factory key; jobs read it with `--env-secret`. |

## One-time setup (CLI)

```bash
curl -sSL https://artifacts.nebius.cloud/cli/install.sh | bash
nebius profile create <name> --endpoint api.nebius.cloud --federation-endpoint auth.nebius.com
nebius config set parent-id <project-id>

# results bucket and S3 keys for a service account (use a named AWS profile!)
nebius storage bucket create --parent-id <project-id> --name <bucket>
nebius iam v2 access-key create --parent-id <project-id> --account-service-account-id <sa-id>

# container registry and job secret
nebius registry create --parent-id <project-id> --name rtt
nebius mysterybox secret create --parent-id <project-id> --name tf-api-key \
  --secret-version-payload '[{"key":"NEBIUS_API_KEY","string_value":"..."}]'
```

## Smoke job

```bash
nebius ai job create --name rtt-smoke \
  --image nvidia/cuda:13.1.1-base-ubuntu24.04 \
  --container-command bash --args "-c 'nvidia-smi; mkdir -p /out/smoke && echo ok > /out/smoke/hello.txt'" \
  --env-secret NEBIUS_API_KEY=tf-api-key \
  --platform gpu-h100-sxm --preset 1gpu-16vcpu-200gb \
  --timeout 1h --restart-policy never \
  --volume <bucket-id>:/out:rw --subnet-id <subnet-id>
```

Measured on 2026-10-15 (eu-north1, H100 80GB, driver 580.173.02): about 2 min provisioning, 45 s starting, 15 s image pull, then the workload. The secret arrived in the environment, and the bucket write was visible from the laptop immediately.

## Observations (feeds the hackathon feedback section)

- **Zero to first job:** the CLI install and browser login took about 2 minutes, with no manual token copying. `nebius profile create` was the smoothest step.
- **Capacity is invisible at submit time:**
  - A job on `gpu-l40s-a` in eu-north1 sat in `PROVISIONING` for more than 10 minutes.
  - The logs only said "Waiting before retrying resource provisioning", and the instance showed `FAILED`/`STOPPED`.
  - `nebius capacity resource-advice list` showed L40S on-demand availability at 0 while H100 was HIGH.
  - Surfacing that advice in `ai job create`, or failing fast, would have saved the wait.
- **`--volume` error message:** passing the bucket *name* returned `unsupported volume source type "rtt"`. The bucket *ID* works. The error could say "use a bucket ID or s3:// URI".
- **`--dry-run` is excellent.** It validated the full job spec without creating anything.
- **Cancel is slow:** `nebius ai job cancel` blocked for more than 2 minutes on a job that never started.
- **Quotas surprise:** the default eu-north1 quota for `compute.instance.non-gpu.vcpu` is 0, so CPU-only jobs there are impossible without a request. uk-south2 has 200.
- **Budgets are console-only.** There is no CLI command to create a budget or alert.
- **Deprecation noise:** `nebius iam project list` warns that it will be removed on 2026-12-16 in favour of `iam v2 project`.
- **Token Factory:**
  - The model catalogue lists only `tools`/`reasoning` as features for the NVIDIA models.
  - In practice, every NVIDIA model accepted forced `tool_choice`, strict `response_format` json_schema and `chat_template_kwargs`.
  - Strict json_schema was **not** enforced for Nemotron 3.5 Lightning: it returned free text first. See [PROBE_LLM.md](PROBE_LLM.md).

## Simulation image and the G1 gate on Serverless Jobs (2026-10-15)

### Image

`docker/Dockerfile.sim` builds one image for both executors (CPU MuJoCo, MuJoCo Warp on the GPU):

- Base `nvidia/cuda:13.1.1-base-ubuntu24.04`. Warp links NVRTC statically and uses the host driver, so the CUDA toolkit libraries of the `runtime` variant are not needed.
- Ubuntu's Python 3.12.3 (the same interpreter as the development venv), `uv sync --frozen` without dev extras, project installed editable at `/app`, entrypoint `rtt`.
- Headless rendering: `MUJOCO_GL=egl` plus `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`, so the job runtime injects the NVIDIA EGL library. Mesa EGL (llvmpipe) and OSMesa are installed as the CPU fallback, so the same image renders on a laptop without a GPU.
- `/app/BUILD_INFO.json` carries the git SHA, a dirty flag and the `uv.lock` sha256; every job manifest copies it. `RTT_GIT_SHA` (`<sha>` or `<sha>-dirty`) is set for the run manifests of `rtt.store`. The build args come after the system and dependency layers, so a new commit rebuilds and re-pushes only the code layers.
- Size: 2.06 GB unpacked, 717 MB compressed in Container Registry.

```bash
jobs/nebius/build_image.sh rtt-sim:dev    # docker build with the git SHA baked in
docker run --rm rtt-sim:dev m1 --no-files # CPU smoke test: G1 passes in about 4 s
jobs/nebius/push_image.sh rtt-sim:dev     # push; the digest reference goes to ~/.config/rtt/image.env
```

`push_image.sh` uses `nebius registry docker-credential` as a Docker credential helper inside a throwaway `DOCKER_CONFIG`, so `~/.docker/config.json` is never edited and no token is written to disk. The first push took 140 s. A code-only re-push took 10 s, because only the top layers change.

### Running jobs

`python -m rtt.cloud` (or `jobs/nebius/m1.sh` / `jobs/nebius/smoke.sh`) does the following:

1. Reads the account settings from `~/.config/rtt/nebius.env` and the image reference from `~/.config/rtt/image.env`. No IDs are stored in the repo.
2. Checks `capacity resource-advice` and picks the first platform with capacity, in the order `gpu-h100-sxm`, `gpu-h200-sxm`, `gpu-l40s-a`.
3. Submits the image **by digest**. The Token Factory key arrives via `--env-secret`, the bucket is mounted by ID at `/out` and the restart policy is `never`.
4. Polls the job, retrying failed polls. A job that has not started within 15 minutes (still `PROVISIONING`, `STARTING` or `IMAGE_PULLING`) is cancelled and the next platform is tried. If the laptop stops watching a live job, the spend so far is still written to the ledger.
5. Appends wall time, run time and estimated dollars to `runs/cloud/spend.jsonl`, enforcing a spend cap.
6. Fetches `runs/<run_id>/` from the bucket and verifies the manifest, report and files.

```bash
python -m rtt.cloud capacity                # which fallback platforms have capacity now
jobs/nebius/m1.sh --dry-run                 # server-side validation; creates nothing
jobs/nebius/m1.sh                           # run, wait, record spend, fetch, verify
python -m rtt.cloud spend                   # the spend ledger
```

Inside the job, `python -m rtt.cloud.entry m1 --out /out/runs/<run_id>` runs the gate. It writes one replay and one MP4 per case, plus `report.json` and `manifest.json`. The manifest records the image name and digest (never the registry path, which contains the registry ID), build info, platform and preset, input hashes, timings, the GPU and driver, the OpenGL renderer and Warp's devices. It records environment variable names only, never their values.

### Measured: G1 gate on `gpu-h100-sxm` / `1gpu-16vcpu-200gb`, eu-north1

Job `rtt-m1-20261015-102052` ran image digest `sha256:d3bec3f74838…`. It **COMPLETED** and the fetched run verified: 7/7 cases as expected, 21 replay and MP4 files, plus the report and the manifest.

| Phase | Job 1 (`…101446`) | Job 2 (`…102052`) |
|---|---|---|
| `PROVISIONING` | about 67 s | about 54 s |
| `STARTING` | about 47 s | about 43 s |
| `IMAGE_PULLING` (717 MB) | about 22 s | about 22 s |
| Created → container start | 127 s | 119 s |
| Container run | 8 s | 7.8 s |
| Gate inside the container (physics, 7 MP4 renders, publish) | 5.6 s | 4.9 s + 0.9 s publish |
| Physics throughput (CPU MuJoCo, one process) | 41.5k steps/s | 41.8k steps/s |
| Estimated cost, wall-based (upper bound) | $0.169 | $0.159 |
| Estimated cost, run-only (lower bound) | $0.010 | $0.010 |

The state timings come from 10 s polling. Total estimated spend for these two jobs is **$0.33** (wall-based). With the earlier `nvidia-smi` smoke job (112 s wall, about $0.14), the account total is about **$0.47**. The cancelled L40S smoke job never started.

What the container saw:
- NVIDIA H100 80GB HBM3, driver 580.173.02 (CUDA 13.1). This meets Warp 1.18's R580+ requirement.
- Warp 1.18.0 found `cuda:0` (sm_90).
- The EGL renderer was `NVIDIA H100 80GB HBM3/PCIe/SSE2`, so the graphics capability was honoured.
- 16 vCPUs.
- `NEBIUS_API_KEY` was present (from MysteryBox).
- No variable carries the job's own ID, so the laptop stores it in `runs/<run_id>/job.json` next to the run.

**Prices** (`rtt/cloud/cost.py`): on-demand GPU-hour prices from https://nebius.com/prices, read 2026-10-15.

| Platform | Price per GPU-hour | Tag |
|---|---|---|
| H100 | $4.50 | [sourced] |
| H200 | $5.40 | [sourced] |
| L40S (Intel) | from $1.55 | [assumed]: the "from" price is used as-is |
| Preemptible H100/H200 | from $0.79 | Spot price, not used for estimates |

The jobs documentation says a job is "billed only while the job is running". The wall-based figure (creation to finish) is therefore an upper bound [assumed], and the spend cap is enforced on it.

### Bucket-mount caveats (found on the first live job)

The `/out` bucket mount does not behave like a local disk:

- **No seeking during writes.** The MP4 muxer's trailer write failed with `Invalid argument`, so no MP4 reached the bucket. The Python call did not raise.
- **Write-then-rename is unsafe.** It left a 0-byte `manifest.json` and `report.json` next to their full `.tmp` sources.

`rtt.cloud.entry` therefore writes every file to a local stage directory first. It then copies each file to `/out` front to back, checks the sizes, and copies `manifest.json` last. Job 2 used this path, and all 22 files plus the manifest arrived intact.

Job 1's run directory (`runs/m1-20261015-101446/`) still holds those broken files. It is superseded by job 2.

## Integrated runs: `rtt job` (2026-10-15, after calibration)

`rtt job {smoke,m1,search}` wraps `python -m rtt.cloud run`. It checks capacity, submits the
pinned image, waits, records the spend, then fetches and verifies the run. `rtt job search`
runs `rtt search` inside the container: `python -m rtt.cloud.entry search --arm ... --runs
... --budget ...`. The search's own outputs go to `search/` and the job manifest goes to
the top level.

```bash
jobs/nebius/build_image.sh && jobs/nebius/push_image.sh    # or: make image push
rtt job m1 --dry-run                                        # server-side validation only
rtt job m1
rtt job search --arm random --runs 1 --budget 40 --seeds 101
rtt job search --arm nemotron --runs 1 --budget 40 --seeds 101 --spend-cap 0.25
rtt export-web runs/cloud/<run_id>/search                   # Nebius provenance on /compute
```

All three jobs used image `rtt-sim@sha256:a446ee7de0ec…`, built from the integration
working tree (`git_dirty: true`; rebuild after the commit). Each ran on `gpu-h100-sxm` /
`1gpu-16vcpu-200gb` in eu-north1, under the calibrated task (`823e58a28a93`) and training
suite (`2dc46d1f79d8`).

| Job | Before the container started | Container run | Work | Estimated cost (wall-based) |
|---|---|---|---|---|
| `rtt-m1-20261015-133917` | 114 s | 8 s | G1 7/7 as expected; 20,495 steps, identical to the laptop | $0.153 |
| `rtt-search-20261015-134158` (random) | 124 s | 22 s | 40 designs, 1,920 episodes in 18.4 s (104 episodes/s on 16 vCPUs); best 94 % | $0.182 |
| `rtt-search-20261015-134451` (Nemotron) | 117 s | 142 s | 40 designs, 1,920 episodes in 18.4 s; 16 Token Factory calls from inside the job | $0.323 compute + $0.058 tokens |

- **Spend.** The ledger total after these jobs is $0.99, a wall-based upper bound for every
  job on the account so far. That is under the $3 development cap.
- **Token Factory from a job.** The key arrived through `--env-secret`, and all 16 calls
  succeeded: 123,007 prompt and 23,912 completion tokens (11,331 of them reasoning), with a
  maximum latency of 10.5 s. The response cache and spend ledger came back with the run
  (`search/llm_cache`, `search/llm_ledger.jsonl`), so `rtt replay-llm` can re-run the arm
  for free.
- **CPU throughput.** Physics on the job's 16 vCPUs ran about 3.3x faster than on the
  shared laptop (104 against 31 episodes/s), so one full arm takes about 3 minutes of CPU
  time per 10 runs. The GPU idles throughout; see the CPU-quota note above.
