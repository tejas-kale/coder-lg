# CodeR Implementation

This is an implementation of the CodeR multi-agent system for resolving GitHub issues, based on the paper ["CodeR: Issue Resolving with Multi-Agent and Task Graphs"](https://arxiv.org/html/2406.01304v2).

## Overview

CodeR is a multi-agent framework that uses pre-defined task graphs to automatically repair bugs and add new features to code repositories. The system achieves state-of-the-art performance on the SWE-bench lite benchmark, resolving 29% of issues with a single submission attempt.

### Key Features

- Multi-agent architecture with 5 specialized agents:
  - Manager: Oversees the process and selects plans
  - Reproducer: Generates test cases to reproduce issues
  - Fault Localizer: Identifies problematic code regions
  - Editor: Performs code modifications
  - Verifier: Validates changes through testing

- Task graph-based planning system
- Integration with software engineering tools for fault localization
- BM25-based code search capabilities

## Implementation Details

This implementation uses:
- LangGraph for the multi-agent orchestration
- Large Language Models for code understanding and generation
- Python for the core system

### Development Tools

This project was developed using:
- [BetterDictation](https://betterdictation.com) - An offline speech-to-text tool that uses OpenAI's Whisper model for fast and accurate transcription
- [Cursor](https://www.cursor.com) - An AI-powered code editor that enhances productivity through intelligent code completion and editing features

## Project Status

🚧 **Under Development** 🚧

This is an active implementation project aiming to reproduce and potentially improve upon the results from the original paper.

## Getting Started

Instructions for setup and usage coming soon.

## Evaluation

The system will be evaluated on:
- SWE-bench lite
- Other relevant software engineering benchmarks

## Citation

```bibtex
@article{chen2024coder,
  title={CodeR: Issue Resolving with Multi-Agent and Task Graphs},
  author={Chen, Dong and Lin, Shaoxin and Zeng, Muhan and Zan, Daoguang and Wang, Jian-Gang and Cheshkov, Anton and Sun, Jun and Yu, Hao and Dong, Guoliang and Aliev, Artem and others},
  journal={arXiv preprint arXiv:2406.01304},
  year={2024}
}
```

## License

Coming soon