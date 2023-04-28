# webui

<img width="600" alt="webui_slide" src="https://user-images.githubusercontent.com/1429346/235155916-757835cf-735b-4cf6-a30b-b20f3271281e.png">

Project homepage: [https://uimodeling.github.io/](https://uimodeling.github.io/)

This repository contains the code and download scripts for the following papers:

* [WebUI: A Dataset for Enhancing Visual UI Understanding with Web Semantics](https://dl.acm.org/doi/abs/10.1145/3544548.3581158) (CHI 2023 - Bst Paper Honorable Mention)
* [WebUI: A Dataset of Web UIs and Associated Metadata to Support Computational UI Modeling](https://drive.google.com/file/d/1f_EeNMXH2TA3o0LixUcbmfgN1PyiGVQ2/view) (CHI 2023 Computational UI Workshop)

## Description
The WebUI dataset contains 400K web UIs captured over a period of 3 months and cost about $500 to crawl. We grouped web pages together by their domain name, then generated training (70%), validation (10%), and testing (20%) splits. This ensured that similar pages from the same website must appear in the same split. We created four versions of the training dataset. Three of these splits were generated by randomly sampling a subset of the training split: Web-7k, Web-70k, Web-350k. We chose 70k as a baseline size, since it is approximately the size of existing UI datasets. We also generated an additional split (Web-7k-Resampled) to provide a small, higher quality split for experimentation. Web-7k-Resampled was generated using a class-balancing sampling technique, and we removed screens with possible visual defects (e.g., very small, occluded, or invisible elements). The validation and test split was always kept the same.

## Repository Structure
Information about each directory:
* `crawler/`  - contains code for the crawler used to collect the WebUI dataset
* `downloads/` - contains scripts to download datasets and pre-trained models
* `models/` - contains scripts for training and reproducing the experiments in the paper
* `notebooks/` - contains example notebooks for running the models
* `sample/` - a sample data point from the WebUI dataset
* `scripts/` - data processing, dataset generation, and model export scripts

## Getting Started

First, install dependencies with `pipenv install` and activate a virtual environment with `pipenv shell`
Example inference code is found in the `notebooks/` directory. To train, check out the `models/` directory. Alternatively, check out [web demos](https://huggingface.co/spaces?sort=modified&search=biglab%2Fwebui) of the models (no installation required).


## Important Notes

* Please see the COPYRIGHT.txt file for information about the data contained within this repository.
* Not all data samples have the same number of files (e.g., same number of device screenshots) due to the fact that the crawler used a timeout during collection
* The [dataset released on HuggingFace](https://huggingface.co/datasets?search=biglab/webui) was filtered using a [list of explicit words](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) and therefore contains fewer samples than the experiments originally used in the paper.
