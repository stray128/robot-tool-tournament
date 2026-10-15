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
