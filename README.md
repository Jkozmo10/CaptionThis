![CaptionThisBanner](https://github.com/Jkozmo10/CaptionThis/assets/87344382/12ff49b1-a849-4725-bd6d-dad13dbb2477)
# CaptionThis 📷 🔤

**CaptionThis** is a Python command-line deep learning model that generates captions describing images provided as inputs.

## Table of Contents

- [Overview](#overview)
  - [Summary](#summary)
  - [Team](#team)
- [Getting Started](#getting-started)
  - [Setup](#setup)
  - [Project Structure](#project-structure)
- [Contributing](#contributing)
  - [Making Changes](#making-changes)
  - [Committing Changes](#commiting-changes)
  - [Making Pull Requests](#making-pull-requests)
- [Documents and Artifacts](#documents-and-artifacts)
- [References](#references)

## Overview

### Summary

CaptionThis is a deep learning project aimed at generating descriptive captions for images using Python. The system is accessible through a command-line interface and leverages a large training dataset for improving caption quality.

### Team

The CaptionThis team consists of 6 Cal Poly students. The team members are 
listed below:

- [Kelly Becker](https://github.com/kbecke05)
- [Anirudh Divecha](https://github.com/anirudhdivecha)
- [Luis D. Garcia](https://github.com/luisdavidgarcia)
- [Jeremy Kozlowski](https://github.com/Jkozmo10)
- [Keila Mohan](https://github.com/keilamohan)
- [Nikhil Nagarajan]()

## Getting Started

Here is all you need to know to setup this repo on your local machine to start 
developing!

### Setup

1. Clone this repository `git clone https://github.com/Jkozmo10/CaptionThis.git`

### Project Structure

- [TrainingSets](./TrainingSets/) Contains the scripts to scrape images from 
    [Google's Conceputal Captions Data Sets](https://ai.google.com/research/ConceptualCaptions/download)

## Contributing
Here are all of the steps you should follow whenever contributing to this repo!

### Making Changes

1. Before you start making changes, always make sure you're on the main branch, 
then `git pull` to make sure your code is up to date
2. Create a branch with the name relating to the change you will make 
`git checkout -b <name-of-branch>`
3. Make changes to the code

### Commiting Changes

When interacting with Git/GitHub, feel free to use the command line, 
VSCode extension, or Github desktop. These steps assume you have already made 
a branch using `git checkout -b <branch-name>` and you have made all neccessary 
code changes for the provided task.

1. View diffs of each file you changed using the [VSCode Github extension](https://code.visualstudio.com/docs/sourcecontrol/github) 
 or [GitHub Desktop](https://desktop.github.com/)
2. `git add .` (to stage all files) or `git add <file-name>` (to stage specific file)
3. `git commit -m " <description>"` or
   `git commit` to get a message prompt
4. `git push -u origin <name-of-branch>`

### Making Pull Requests

1. Go to the Pull Requests tab on [this repo](https://github.com/Jkozmo10/CaptionThis/pulls)
2. Find your PR, and provide a description of your change, steps to test it, and any other notes
3. Link your PR to the corresponding **Issue**
4. Request a reviewer to check your code
5. Once approved, your code is ready to be merged in 🎉

## Documents and Artifacts
1. [Project Proposal](https://docs.google.com/document/d/1zY6C1oZQD-xH8PxsW7HxgOw04BbP47yM9s87nwDO5K8/edit?usp=drive_link)
2. [Features, Requirements, and Evaluation Criteria](https://docs.google.com/document/d/1ofBOCf_vS02fTwZD2EXzXBWywHsfNvTdzwL9iSkE48A/edit?usp=drive_link)
3. [System Design and Architecture](https://docs.google.com/document/d/1rq2T96CJmd9xNLXJ0FJWdUbd96fcu4hN5mUcfXDsmYs/edit?usp=drive_link)
4. [Implementation and Prototypes](https://docs.google.com/document/d/1zRcj3RjaOIZ7m6m30ssQxjJk8Yn4bs4dmvsVqAUMf3A/edit?usp=drive_link)

## References

1. [Pre-Trained Model](https://huggingface.co/docs/transformers/training#optimizer-and-learning-rate-scheduler)
2. [Helpful Training Google Colab Notebooks](https://huggingface.co/docs/transformers/notebooks)
3. [Example Google Colab Image Captioning](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb#scrollTo=lTI8wKxgql9i)
4. [Generating Datasets](https://huggingface.co/docs/datasets/image_dataset#generate-the-dataset)
