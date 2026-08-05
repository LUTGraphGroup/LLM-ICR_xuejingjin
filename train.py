"""Compatibility entry point.

The supplied paper states that the LLM backbone was not fine-tuned. In this reference
package, "training" means validation-set tuning of CAIR weights and a decision threshold
from cached candidate outputs. See `python train.py --help`.
"""
from llm_icr.cli.tune_cair import main

if __name__ == '__main__':
    main()
