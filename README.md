# MLOps Project 2025, group 83

Participating students:
Martin Moos Hansen (s203822), Leire Hernandez Martinez (s243266), Mikkel Koefoed Lindtner (s205421), Lucas Pedersen (s203768) and Rita Saraiva (s205717).

## Table of Contents
1. [Project Description](#project-description)
    1. [Overall goal of the project](#overall-goal-of-the-project)
    2. [What framework are you going to use, and do you intend to include the framework into your project?](#what-framework-are-you-going-to-use-and-do-you-intend-to-include-the-framework-into-your-project)
    3. [What data are you going to run on?](#what-data-are-you-going-to-run-on)
    4. [What models do you expect to use?](#what-models-do-you-expect-to-use)
    5. [Closing remarks?](#closing-remarks)
2. [Project Structure](#project-structure)

## Project description

### Overall goal of the project
The goal of this project, is to train Generative Adversarial Networks ([GAN](https://dl.acm.org/doi/abs/10.1145/3422622)), on the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Then, use an open-source framework, such as [CNNDetection](https://github.com/PeterWang512/CNNDetection) to validate whether we can create a GAN capable of generating images, which a GAN Detector cannot detect as GAN generated (or at the very least, perform similar to other GANs). The value proposition, is creating a Neural Network, with the ability of generating images, which hopefully look somewhat realistic.

### What framework are you going to use, and do you intend to include the framework into your project?
There are several open-source frameworks which we have considered for this task. So far, we have identified the following projects as being of interest:

* [CNNDetection](https://github.com/PeterWang512/CNNDetection) (For Automated GAN Detection)
* [Albumentations](https://albumentations.ai/) (For doing data augmentations)
* [TorchGAN](https://github.com/torchgan/torchgan) (Library which is specialised in making GANs in PyTorch)

Though, we will need to get somewhat deeper in the project development process to know which of the frameworks provide value to the project, and we might discover further libraries throughout.

Likewise, we intend to do most things using the PyTorch framework, though maybe also PyTorch Lightning, if certain models can be implemented more succinctly in this library. Furthermore, we also intend to include the open-source frameworks in the project.

### What data are you going to run on?
We are going to train the models on the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), however, we may augment the data, or add to it, if we find this to be of relevance for the project.

The dataset is ~160 MB of data. At an initial stage of tests, this should be adequate for testing that our model does indeed work. However, in order to meet all the requirements of the course as well as make the project more challenging for the group, we shall attempt to artificially enlarge the dataset, via the [Albumentations](https://albumentations.ai/) package, specifically designed to do data augmentation.

### What models do you expect to use?
We expect to use a multitude of different models. Which models we use, depends on what seems to work well in practice. So far, we have identified the following models:

* Simple Custom FFNN Network (For a very barebones MVP).
* A GAN similar to the one presented in the Deep Learning Course: [02456 Deep Learning](https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch/blob/master/7_Unsupervised/7.3-generative-adversarial-networks.ipynb) (This one is based on MNIST, and hence will need to be augmented to work for CIFAR-100).
* GANs using the [TorchGAN](https://github.com/torchgan/torchgan) library.
* Anything else that may pop up during the project development process.

### Closing remarks?
As an ambitious and final benchmark of our project, we aim conduct testing with a human comparison. In an ideal case the images generated would be close to indistinguishable of real images in terms of quality and realism, including that of the training data set. Therefore, a comparison test would comprise of a random shuffling of real and generated images and asking the test subject to guess on which ones where artificially created. A sign of excellence, would be achieving a correct guess rate of 50% when half of the images are real, and the other half are fake. However, the group, despite the ambition, realizes the difficulty of achieving this level of robustness on a GAN.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
