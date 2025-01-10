# MLOps Project 2025, group 83

Participating students:
Martin Moos Hansen (s203822), Leire Hernandez Martinez (s243266), Mikkel Koefoed Lindtner (s205421), Lucas Pedersen and Rita Saraiva.


## Project description
This project aims to to train a Generative Adversarial Network ([GAN](https://dl.acm.org/doi/abs/10.1145/3422622)), on the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The goal is to generate images close to those in the dataset.The implementation will be done with [PyTorch](https://pytorch.org/). More over, we shall use [Albumentations](https://albumentations.ai/) for data augmentation as our third-party package.

Given the fact that none of group member have worked with GANs before, we shall attempt to implement the models ourselves. As such, we will be exploiting the PyTorch framework to build this model, and thus not start with a pre-trained model. Our plan is to use the implementation of a GAN, presented in the course [02456](https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch/blob/master/7_Unsupervised/7.3-generative-adversarial-networks.ipynb) Deep Learning, as a baseline model. In this manner, we can more easily verify that our data preprocessing has been implemented correctly. Note that this model was originally implemented for the MNIST dataset, so it is reasonable to claim that a more complicated model will be needed. Thus in subsequent iterations, we will be adapting it to fit the new dataset. 

The dataset is ~160 MB of data. At an initial stage of tests, this should be adequate for testing that our model does indeed work. However, in order to meet all the rquirements of the course as well as make the project more challenging for the group, we shall attempt to artificially enlarge the dataset, via the [Albumentations](https://albumentations.ai/) package, specifically designed to do data augmentation.

As an ambitious and final benchmark of our project, we aim conduct testing with human comparison. In an ideal case the images generated would be close to indistinguishable of real images in terms of quality and realism -including that of the training data set. Therefore, a comparison test would comprise of a random shuffling of real and generated images and asking the test subject to guess on which ones where artificially created. A sign of excellence would be to get that an equal number of real and generated images would be guess to be real. However, the group, despite the ambition, realizes the difficulty of achieving this level of robustness on a GAN.


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
