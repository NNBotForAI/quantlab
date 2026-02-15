# QuantLab Runner Skill for OpenClaw

This skill enables OpenClaw to run QuantLab pipeline commands safely.

## Installation

1. Copy this skill to your OpenClaw workspace:
   ```bash
   mkdir -p ~/.openclaw/workspace/skills/quantlab_runner
   cp -r * ~/.openclaw/workspace/skills/quantlab_runner/
   ```

2. Ensure the command allowlist is in place:
   ```bash
   cp policies/command_allowlist.json ~/.openclaw/openclaw/policies/
   ```

## Usage

OpenClaw can run QuantLab commands using the following patterns:

### Update Market Data
```
python -m quantlab.pipeline.run data:update --config configs/examples/us_momentum.json
```

### Run Backtest
```
python -m quantlab.pipeline.run backtest:run --config configs/examples/us_momentum.json
```

### Run Optimization
```
python -m quantlab.pipeline.run optimize:run --config configs/examples/us_momentum.json --trials 100
```

### Run Robustness Analysis
```
python -m quantlab.pipeline.run robustness:run --config configs/examples/us_momentum.json --walk-forward --bootstrap
```

### Build Report
```
python -m quantlab.pipeline.run report:build --run-id <run_id>
```

## Security

The command allowlist restricts execution to only approved QuantLab commands:
- ✅ Allowed: data:update, backtest:run, optimize:run, robustness:run, report:build
- ❌ Blocked: rm, curl, wget, ssh, scp, printenv, cat ~/, cat /etc/

## Example OpenClaw Integration

```python
# In an OpenClaw session, you can ask:
"Run backtest for the US momentum strategy"

# OpenClaw will execute:
python -m quantlab.pipeline.run backtest:run --config configs/examples/us_momentum.json --output results

# Then summarize the results
"Show me the backtest results"
```

## Configuration

Update the skill configuration as needed:

```yaml
# ~/.openclaw/workspace/skills/quantlab_runner/config.yaml
default_config: configs/examples/us_momentum.json
results_dir: results
parallel_backend: joblib
max_workers: 4
```
