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

> **Superseded defaults.** The sections from here down to "uk-south2 CPU jobs" describe
> the first day's setup, when every job ran on the eu-north1 H100. Since then the default
> target for scored jobs is `uks2-cpu` (uk-south2, `cpu-d3`, CPU MuJoCo), with the H100 as
> fallback only; see [uk-south2 CPU jobs](#uk-south2-cpu-jobs-the-default-target-for-scored-runs-2026-10-15).
> Scored images are built from a commit (`make image`, i.e. `build_image.sh --ref HEAD`),
> never from the working tree.

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

## uk-south2 CPU jobs: the default target for scored runs (2026-10-15)

CPU MuJoCo is the executor of record (Addendum B), and eu-north1 has no non-GPU vCPU
quota. Scored jobs therefore run on CPU-only VMs in uk-south2, which has a 200 vCPU quota.
The H100 in eu-north1 is now the fallback only.

### Targets

`rtt.cloud.nebius_jobs` names two region profiles:

| Target | Region | Platforms, in order | Role |
|---|---|---|---|
| `uks2-cpu` | uk-south2 | `cpu-d3` (AMD EPYC Genoa), preset `16vcpu-64gb` | Default. `--vcpus 2…256` picks another preset with 4 GiB per vCPU. |
| `eun1-gpu` | eu-north1 | `gpu-h100-sxm`, `gpu-h200-sxm`, `gpu-l40s-a` | Fallback. |

uk-south2 offers one CPU platform, `cpu-d3`; there is no `cpu-e2` there. The only GPU
platform in the region is `gpu-rtx6000-a`.

`python -m rtt.cloud run` (and `rtt job`) submits to `uks2-cpu`. The job moves to
`eun1-gpu` in three cases:

- uk-south2 is not configured.
- The free vCPU quota does not cover the preset.
- The job ends before its container starts, for example a `StartFailed` image pull.

`--no-fallback` disables the move. `--target eun1-gpu`, or `--platform gpu-h100-sxm`, asks
for the GPU directly. `job.json` records `target`, `region`, `fallback_from`, `price` and
every attempt, including failed ones. The site's cost therefore covers the whole job.

Capacity advice has no rows for CPU platforms. For a CPU target, the capacity check reads
the `compute.instance.non-gpu.vcpu` quota (`quotas quota-allowance get-by-name`) instead.

```bash
python -m rtt.cloud capacity                          # both targets: CPU quota, GPU advice, prices
rtt job m1                                            # uks2-cpu, falling back to eun1-gpu
python -m rtt.cloud run search --arm random --runs 1 --budget 40 --seeds 101
python -m rtt.cloud run search --arm random --runs 10 --budget 40 --vcpus 32
python -m rtt.cloud run m1 --target eun1-gpu          # the H100 on purpose
```

### Configuration

Each uk-south2 resource is set by the same variable as its eu-north1 counterpart, with an
`_UKSOUTH2` suffix. These live in `~/.config/rtt/nebius.env` and `image.env`, never in the
repository.

| Variable | uk-south2 resource |
|---|---|
| `NEBIUS_PROJECT_ID_UKSOUTH2` | the project |
| `NEBIUS_SUBNET_ID_UKSOUTH2` | the project's default subnet |
| `NEBIUS_BUCKET_NAME_UKSOUTH2`, `NEBIUS_BUCKET_ID_UKSOUTH2` | bucket `rtt-runs-uks2-<random suffix>` |
| `NEBIUS_S3_PROFILE_UKSOUTH2`, `NEBIUS_S3_ENDPOINT_UKSOUTH2` | AWS profile `nebius-uks2`, `https://storage.uk-south2.nebius.cloud` (the bucket's `status.domain_name`) |
| `NEBIUS_REGISTRY_ID_UKSOUTH2`, `RTT_IMAGE_UKSOUTH2` | registry `rtt-uks2` and the digest reference pushed there |
| `NEBIUS_SECRET_NAME_UKSOUTH2` | the eu-north1 `tf-api-key` secret, by ID |

`NebiusConfig.for_region("uk-south2")` builds the region's settings. It never inherits the
eu-north1 project, subnet, bucket, S3 profile, registry or image. A missing value is
reported under its suffixed name and never silently replaced by the eu-north1 one.
`check_same_image` refuses a run when the two registries hold different digests. The CPU
job and its H100 fallback always run the same build.

`jobs/nebius/build_image.sh --ref HEAD` builds from a `git archive` of one commit, so the
image never picks up half-written files from the working tree. `jobs/nebius/push_image.sh`
pushes to every configured registry. The digest is the same in both regions.

| Image | Value |
|---|---|
| Image | `rtt-sim@sha256:07d0bea9b8fa…` |
| Built from | `b184c81c5b93`, `git_dirty: false` |
| Push to eu-north1 | 60 s (5 s on a re-push) |
| First push to the new uk-south2 registry | 94 s |

The S3 key for the existing service account `rtt-automation` was created with the
uk-south2 project as its parent. The account is in the tenant's `editors` group, so the
bucket needs no extra grant. The key went straight into the AWS profile.

Gate and random-search jobs no longer get the Token Factory key. Only the probe and the
`nemotron`/`zeroshot` search arms get it. See `needs_token_factory`.

### Prices

CPU-only VMs are billed per vCPU-hour plus per GiB-hour. The rates are from
https://docs.nebius.com/compute/resources/pricing (from 2026-10-01, read 2026-10-15). The
marketing page's "AMD EPYC Genoa from $0.13" equals the 4 vCPU / 16 GiB preset at these
rates.

| Platform / preset | USD per hour | Tag |
|---|---|---|
| `cpu-d3` vCPU | $0.015 per vCPU-hour | [sourced] |
| `cpu-d3` RAM | $0.0045 per GiB-hour | [sourced] |
| `cpu-d3` `16vcpu-64gb` | **$0.528** | [sourced] |
| `cpu-d3` `32vcpu-128gb` | $1.056 | [sourced] |
| `cpu-e2` (Intel Ice Lake, eu-north1 only) | $0.012 per vCPU-hour + $0.0045 per GiB-hour | [sourced] |
| `gpu-h100-sxm` `1gpu-16vcpu-200gb` | $4.50 | [sourced] |
| Enhanced-throughput object storage | $0.11 per GiB-month | [sourced]; a run is about 1 MB |

The wall-based estimate is still the upper bound that the spend cap enforces [assumed].

The cap also holds when several `rtt job` processes share the ledger, for example the
random and Nemotron arms launched side by side. Before submitting, each job reserves its
worst case in `runs/cloud/spend.jsonl.inflight.json` under a file lock. While it runs,
it raises that reservation to its live spend. A new job is refused when recorded spend
plus every reservation plus its own worst case would pass the cap. A running job is
cancelled once its spend reaches what the cap leaves after everything else. Reservations
of processes that have died are dropped. `python -m rtt.cloud spend` lists the jobs in
flight.

### Measured: uk-south2 `cpu-d3` against eu-north1 H100

Both columns are jobs from this session's image or the last H100 image, with the same task
(`823e58a28a93`) and the same training suite. States were polled every 10 s.

| | m1, H100 (`…133917`) | m1, `cpu-d3` (`…171556`) | random search, H100 (`…134158`) | random search, `cpu-d3` (`…172651`) |
|---|---|---|---|---|
| `PROVISIONING` | 54–67 s (earlier H100 jobs) | about 55 s | | about 55 s |
| `STARTING` | 43–47 s (earlier H100 jobs) | about 43 s | | about 43 s |
| `IMAGE_PULLING` (717 MB, same-region registry) | about 22 s | about 21 s | | about 22 s |
| Created → container start | 114 s | 112 s | 124 s | 111 s |
| Container run | 8 s | 22 s | 22 s | 21 s |
| Work inside the container | gate 4.6 s | gate 20.7 s | 40 designs, 1,920 episodes, evaluator 18.4 s | 40 designs, 1,920 episodes, evaluator 18.7 s |
| Physics | 20,495 steps, 31.0k steps/s | 20,495 steps, 30.1k steps/s | 104 episodes/s | 103 episodes/s |
| Headless GL renderer | NVIDIA EGL | Mesa llvmpipe (software) | | |
| Wall | 122 s | 134 s | 146 s | 132 s |
| **Estimated cost, wall-based** | **$0.153** | **$0.020** | **$0.182** | **$0.019** |
| Estimated cost, run-only | $0.010 | $0.003 | $0.027 | $0.003 |

- **Same results, 8–10x cheaper.** The CPU random search reproduced the H100 run byte for
  byte, on a different CPU vendor: `episodes.parquet`, `designs.parquet`, `designs.jsonl`
  and `history.jsonl` have identical sha256 hashes. The frozen pick (d06, 93.75 %)
  differs only in its `frozen_at` timestamp. Gate G1 took the same 20,495 steps on both.
- **Physics speed is unchanged.** 16 Genoa vCPUs run the search at the speed of the H100
  host's 16 vCPUs (18.7 s against 18.4 s).
- **`--vcpus 32` should halve the evaluator time [assumed, not measured].** It costs the
  same per CPU-second, plus the provisioning overhead.
- **Rendering is the one slowdown.** Without a GPU, the 7 MP4s render on llvmpipe. The
  gate takes 20.7 s instead of 4.6 s, which is still under 1 cent. `--no-video` removes
  the cost when no video is needed.
- **Provisioning dominates both platforms** (about 110 s before the container starts). One
  job carrying many runs amortises it better than many small jobs.
- **Probe job `…173010`**, `cpu-d3`, 116 s wall, $0.017. The secret, referenced by ID from
  the eu-north1 project, arrived in the uk-south2 job (`token_factory_key_present: true`).
  The Nemotron arm can therefore run in uk-south2 without a copy of the key.
- **Spend:**
  - The failed first m1 attempt (`…170620`, cross-region pull) cost about $0.026.
  - All four uk-south2 jobs together cost about **$0.08**, wall-based.
  - The ledger total is $1.07, under the $3 development cap.

### Observations (feedback)

- **A new bucket can be unusable by default.**
  - In uk-south2 the tenant has no `storage.bucket.size.standard` quota, only
    `enhanced-throughput`.
  - `storage bucket create` succeeded with the default STANDARD class. The first
    `PutObject` then failed with `QuotaLimitExceeded … used 0 of 0`.
  - Fix: `bucket update --default-storage-class enhanced_throughput --force-storage-class`.
  - Bucket create could pick a class the region has quota for, or warn.
- **A job cannot pull from a registry in another project of the same tenant.**
  - A uk-south2 job ran `IMAGE_PULLING` for about 60 s against the eu-north1 registry. It
    then ended in `ERROR`/`StartFailed` with "Could not pull the image… registry
    credentials allow access".
  - `--dry-run` had passed.
  - Fix: a second registry in uk-south2 with the same digest pushed to it.
  - A dry run that checks pull access, or tenant-wide registry read for jobs, would have
    saved one failed job.
- **Secrets are per project, but a cross-project ID works.**
  - `--env-secret NAME=tf-api-key` fails in the uk-south2 dry run with `SecretNotFound`.
  - The eu-north1 secret's **ID** passes the dry run and is delivered at run time. This is
    useful, but undocumented.
- **Capacity advice is GPU-only.** CPU capacity has to be inferred from the vCPU quota. A
  CPU row in `capacity resource-advice` would make one capacity check work for both.
- **CPU prices are split across two pages.** nebius.com/prices shows only "from $0.13".
  The per-vCPU and per-GiB rates needed to price a preset are on the docs pricing page.
- **No `NEBIUS_JOB_ID` inside the container** (unchanged). The laptop still stores the ID
  next to each run.
