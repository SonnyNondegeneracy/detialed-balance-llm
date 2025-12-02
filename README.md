# Detailed Balance in Large Language Model-Driven Agents

Supplemental code for the paper "Detailed balance in large language model-driven agents".

## Overview

This repository contains the implementation and experimental code for validating detailed balance conditions in LLM-driven agents. We demonstrate that LLM agents exhibit physical principles analogous to statistical mechanics through two distinct agent frameworks.

## Agent Frameworks

### 1. Conditioned Word Generation Agent

An agent that generates words where the sum of letter indices equals 100 (e.g., ATTITUDE, EXCELLENT).

- **Models tested**: GPT5-Nano, Claude-4, Gemini-2.5-flash
- **Purpose**: Examine generative dynamics and validate detailed balance through potential functions
- **Implementation**: `run_agent/word_count.py`

### 2. IdeaSearchFitter Agent

An agent based on symbolic regression tasks that searches for mathematical expressions.

- **Task**: Fitting the `nikuradse_2` dataset from PMLB
- **State space**: Expression trees represented as numexpr strings
- **Dataset**: 50,228 state transitions, 21,697 unique transitions, 7,484 unique states
- **Implementation**: `run_agent/ideasearch_demo_mse.py`

## Repository Structure

```
.
├── main.py                      # Main execution script for running experiments
├── V_hand.py                    # Hand-crafted potential function implementation
├── ideasearch_potential.py      # Potential function discovered by IdeaSearch
├── build_potential.py           # Utilities for constructing potential functions
├── random_optimize_manual.py    # Gradient descent optimization for potential parameters
├── plot.py                      # Visualization and plotting utilities
├── api_keys.json               # API keys configuration for LLM models
│
├── run_agent/                   # Agent implementations
│   ├── word_count.py           # Conditioned Word Generation Agent (sum of letter indices = 100)
│   └── ideasearch_demo_mse.py  # IdeaSearchFitter Agent for symbolic regression
│
├── samll case/                  # Small-scale experiments and model-specific analysis
│   ├── transition_claude.py    # Transition kernel analysis for Claude models
│   ├── transition_gemini.py    # Transition kernel analysis for Gemini models
│   └── calculate_potential.py  # Potential function calculation for small cases
│
├── search/                      # IdeaSearch configuration and execution
│   └── [search scripts]        # Scripts for automated potential function discovery
│
├── figures/                     # Generated visualizations and plots
│   └── [output plots]          # Action minimization, transition analysis, etc.
│
└── data/                        # Experimental data (available on Hugging Face)
    └── [transition data]       # State transition records and statistics
```

## Data

The experimental data is available on Hugging Face:
[Dataset link to be added]

## Requirements

```bash
pip install numpy scipy matplotlib ideasearch-fit # Core dependencies
pip install numexpr  # For expression evaluation
# Add your LLM API keys to api_keys.json
```

## Usage

### Running the Conditioned Word Generation Agent

```bash
python run_agent/word_count.py
```

### Running the IdeaSearchFitter Agent

```bash
python run_agent/ideasearch_demo_mse.py
```

### Potential Function Discovery

```bash
# Run IdeaSearch to discover potential functions
python seach/run.py

# Optimize discovered potential function
python random_optimize_manual.py
```

### Visualization

```bash
python main.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
[Citation to be added upon publication]
```

## License

MIT License. See LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
