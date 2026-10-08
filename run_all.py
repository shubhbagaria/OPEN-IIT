"""One-command reproduction of every deliverable.

    python run_all.py            # tabular pipeline + streaming demo + figures (CPU, ~4 min)
    python run_all.py --slm      # + SLM SFT data, LoRA fine-tune, eval, CUDA latency benchmark (GPU)
"""
import sys

from ptp_copilot import demo_streaming, figures, pipeline

if __name__ == "__main__":
    pipeline.main()
    demo_streaming.main()
    figures.main()
    if "--slm" in sys.argv:
        from ptp_copilot import slm
        slm.build_sft()
        slm.train()
        slm.evaluate()
        slm.benchmark()
