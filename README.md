# textalignsynth

## Hearing Your Way Through Music Recordings: A Text Alignment and Synthesis Approach

This is a repository accompanying the following paper:

```bibtex
@inproceedings{StrahlOBM25_TextAlignSynth_SMC,
  author    = {Sebastian Strahl and Yigitcan {\"O}zer and Hans-Ulrich Berendes and Meinard M{\"u}ller},
  title     = {Hearing Your Way Through Music Recordings: A Text Alignment and Synthesis Approach},
  booktitle = {Proceedings of the Sound and Music Computing Conference ({SMC})},
  address   = {Graz, Austria},
  year      = {2025}
}
```

This repository contains an implementation of parts of the processing pipeline described in above paper. The implementation comprises text comment generation for the case studies described in the paper, text-to-speech synthesis using the [TTS python package](https://github.com/coqui-ai/TTS), post-processing of the synthesized speech signals, and superposition with the original recording.

For details and references, please see the paper.

## Installation

### 1. Set up Python environment
We recommend setting up a Python environment including Pytorch before installing `textalignsynth`. You may use the [example environment](environment.yaml) provided as part of this package:

```bash
git clone https://github.com/groupmm/textalignsynth.git
cd textalignsynth
conda env create -f environment.yaml
conda activate textalignsynth
```

### 2. Install `textalignsynth`

#### Option 1: Installation without cloning this repository:

```bash
pip install "git+https://github.com/groupmm/textalignsynth.git#egg=textalignsynth"
```

#### Option 2: Installation by cloning this repository:

```bash
git clone https://github.com/groupmm/textalignsynth.git
cd textalignsynth
pip install -e .
```

#### Warnings:
- :warning: Does not work on Windows machines! Workaround: Use Windows Subsystem for Linux (WSL).
- :warning: German TTS model requires `espeak-ng` or `espeak` to be intalled on the machine!


## Contribution
Automated code style checks via [pre-commit](https://pre-commit.com/):

```bash
pip install pre-commit
pre-commit install
```

## License
The code for this toolbox is published under an [MIT license](LICENSE).
This does not apply to the data files:
- Schubert songs are taken from the [Schubert Winterreise Dataset](https://zenodo.org/records/10839767).
- Beethoven pieces are taken from the [Beethoven Piano Sonatas Dataset](https://zenodo.org/records/12783403).
- Wagner operas are taken from the [Wagner Ring Dataset](https://zenodo.org/records/7672157).

## Acknowledgements

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) Grant No. 500643750 (MU 2686/15-1). The authors are with the [International Audio Laboratories Erlangen](https://audiolabs-erlangen.de/), a joint institution of the [Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)](https://www.fau.eu/) and [Fraunhofer Institute for
Integrated Circuits IIS](https://www.iis.fraunhofer.de/en.html).
