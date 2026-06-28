# Deployment Notes

This project is a research-oriented synthetic defect generation and segmentation validation framework. It is not packaged as a production service, but it now includes a lightweight local inference interface.

## What Is Ready

| Capability | Status |
| --- | --- |
| Final metric collection | Ready |
| Reproduction health check | Ready |
| Diagnostic evidence summaries | Ready |
| Local inference script | Ready |
| Dry-run inference check | Ready |

## What Is Local Only

Model checkpoints are intentionally not committed. They are ignored by `.gitignore`:

```text
*.pt
*.pth
*.ckpt
*.safetensors
```

To run real inference, train a final model locally and pass its checkpoint:

```powershell
python scripts/18_inference_demo.py `
  --category leather `
  --image path/to/inspection_image.png `
  --checkpoint path/to/best_model.pt `
  --output-dir outputs/demo_inference
```

## Public Dry-Run

For a no-checkpoint public sanity check:

```powershell
python scripts/18_inference_demo.py --category leather --dry-run
```

This writes:

```text
outputs/demo_inference/demo_requirements.md
outputs/final_report/deployment_readiness.md
```

## Output Contract

The inference script writes:

| Output | Purpose |
| --- | --- |
| `probability_mask.png` | Pixel probability map |
| `binary_mask.png` | Thresholded segmentation mask |
| `overlay.png` | Visual overlay for inspection |
| `metadata.json` | Category, threshold, image score, area ratio, and output paths |

## Production Gaps

A production deployment would still need:

```text
1. Model registry and checkpoint versioning.
2. Calibration on the target factory camera/domain.
3. Runtime latency benchmarking.
4. Monitoring for false positives on normal images.
5. Human review workflow for uncertain predictions.
```

The current repository is best viewed as a reproducible research and interview-ready project, with a clear path from synthetic data experiments to local inference.
