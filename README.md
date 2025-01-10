# MLOps Project 2025, group 83

Participating students:
Martin Moos Hansen, Leire Hernandez Martinez, Mikkel Koefoed Lindtner, Lucas Pedersen and Rita Saraiva.


## Project description
The brief summary of this project is to train a Generative Adversarial Network ([GAN](https://dl.acm.org/doi/abs/10.1145/3422622)), on the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), using [Albumentations](https://albumentations.ai/) for data augmentation as our third-party package. The goal is to generate images close to those in the dataset.

The implementation will be done in [PyTorch](https://pytorch.org/). For educational purposes we would like to implement the models ourselves, since we have not yet worked with GANs. So we will be exploiting the PyTorch framework to build this model, and not just borrow a pre-trained model.

Our plan so far is to use the implementation of a GAN, which was presented in the Deep Learning course [02456](https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch/blob/master/7_Unsupervised/7.3-generative-adversarial-networks.ipynb), as inpiration for a baseline model. Of course altering it to fit the new dataset. Hopefully this model can serve as inspiration for a very minimal product, which should help us verify that our data preprocessing has been implemented correctly. This model is originally implemented for the MNIST numbers dataset, so we will presumably need to create a more complicated model at some point.

The dataset is ~160 MB of data. In initial tests, this should be fine to thattest our models work. To make the project more interesting, and artificially create a larger dataset, we'll use the [Albumentations](https://albumentations.ai/) to do data augmentation.


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
