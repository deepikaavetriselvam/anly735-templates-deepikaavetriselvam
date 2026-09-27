# Predictive AI Replication Laboratory

The weekly Replication Laboratory develops your ability to critically evaluate Predictive AI research through computational investigation.

The goal is **not simply to make code run**.

The goal is to determine whether a published result can be reproduced, understand why results may differ, and identify what the replication teaches us about the original research.

## Replication Types

### Direct Replication
Use the authors' original data and/or code to reproduce a reported result.

### Partial Replication
Reproduce part of the original analysis when only some research materials are available.

### Proxy Replication
Use a comparable public dataset and similar analytical approach when the original data or code are unavailable.

Proxy replication does not imply that numerical results should match the original study.

## Core Workflow

**Claim → Replicate → Compare → Diagnose → Extend**

## What Matters

A strong replication laboratory demonstrates:

1. a clearly identified research claim;
2. an appropriate replication strategy;
3. transparent data and methods;
4. reproducible analysis;
5. meaningful comparison with the original study;
6. critical interpretation of differences; and
7. a defensible research extension.

## Repository Expectations

Your repository should contain the materials necessary to understand and reproduce your analysis.

A typical repository may include:

```text
README.md
replication-lab.qmd
references.bib
data/
code/
figures/
```
Do not upload restricted, confidential, licensed, or personally identifiable data.

When data cannot be redistributed, document how an authorized researcher can obtain them.

## Submission Workflow

1. Accept the weekly GitHub Classroom assignment.
2. Clone the repository.
3. Read the assigned anchor paper.
4. Identify the target research claim.
5. Complete the replication analysis.
6. Complete replication-lab.qmd.
7. Render the report to Word.
8. Verify that your analysis is reproducible.
9. Commit and push your source files.
10. Submit the .docx report to Canvas.

## Guiding Principle

Do not chase identical numbers. Investigate reproducibility.

## Replication Laboratory #2: Plasticity-Loss Proxy Experiment

This repository contains a proxy replication examining whether prior training changes a neural network's ability to learn after the target relationship changes. A continued learner is first trained on Task A and subsequently adapted to Task B. Its post-change learning behavior is compared with a freshly initialized network trained on Task B.

### Computational Environment

- Python 3.13.15
- PyTorch 2.14.0+cpu
- NumPy 2.5.3
- Pandas 3.0.6
- Matplotlib 3.11.2
- Windows, CPU execution

The exact Python dependencies used for the experiment are recorded in `requirements.txt`.

### Reproducing the Experiment

From the `replication-lab` directory, create and activate a Python virtual environment and install the recorded dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
