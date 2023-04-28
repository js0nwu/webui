# webui

Project homepage: [https://uimodeling.github.io/](https://uimodeling.github.io/)

This repository contains the code and download scripts for the following papers:

* [WebUI: A Dataset for Enhancing Visual UI Understanding with Web Semantics](https://dl.acm.org/doi/abs/10.1145/3544548.3581158) (CHI 2023 - Bst Paper Honorable Mention)
* [WebUI: A Dataset of Web UIs and Associated Metadata to Support Computational UI Modeling](https://drive.google.com/file/d/1f_EeNMXH2TA3o0LixUcbmfgN1PyiGVQ2/view) (CHI 23 Computational UI Workshop)


Please see the COPYRIGHT.txt file for information about the data contained within this repository.

Information about each directory:
* crawler/  - contains code for the crawler used to collect the WebUI dataset
* downloads/ - contains scripts to download datasets and pre-trained models
* models/ - contains scripts for training and reproducing the experiments in the paper
* notebooks/ - contains example notebooks for running the models
* sample/ - a sample data point from the WebUI dataset
* scripts/ - data processing, dataset generation, and model export scripts


Important notes -
* Not all data samples have the same number of files (e.g., same number of device screenshots) due to the fact that the crawler used a timeout during collection
* The [dataset released on HuggingFace](https://huggingface.co/datasets?search=biglab/webui) was filtered using a [list of explicit words](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words)
