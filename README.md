# Conformal Risk Control
This is the official repository of [Conformal Risk Control](http://arxiv.org/abs/2208.02814) by Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster.

<p align="center">
    <a style="text-decoration:none !important;" href="http://arxiv.org/abs/2208.02814" alt="arXiv"> <img src="https://img.shields.io/badge/paper-arXiv-red" /> </a>
    <a style="text-decoration:none !important;" href="https://people.eecs.berkeley.edu/%7Eangelopoulos/" alt="website"> <img src="https://img.shields.io/badge/website-Berkeley-yellow" /> </a>
    <a style="text-decoration:none !important;" href="https://docs.conda.io/en/latest/miniconda.html" alt="package management"> <img src="https://img.shields.io/badge/conda-env-green" /> </a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"> <img src="https://img.shields.io/badge/license-MIT-blue.svg" /> </a>

</p>

## Technical background
In the risk control problem, we are given some loss function $L_i(\lambda) = \ell(X_i,Y_i,\lambda)$.
For example, in multi-label classification, you can think of the loss function as the false negative proportion $L_i(\lambda) = 1 - \frac{|Y_{i} \cap C_{\lambda}(X_{i})|}{|Y_i|}$, where $C_{\lambda}(X_{i})$ is the set-valued output of a machine learning model. 
As $\lambda$ grows, so does the set $C_{\lambda}(X_{i})$, which shrinks the false negative proportion.
We seek to choose $\hat{\lambda}$ based on the first $n$ data points to control the expected value of its loss <i>on a new test point</i> at some user-specified risk level $\alpha$, $$\mathbb{E}\big[L_{n+1}(\hat{\lambda})\big] \leq \alpha.$$

The conformal risk control algorithm is in `core/get_lhat.py`. It is 5 lines long, including the function header.

## Examples
Each of the `{polyps, coco, hierarchical-imagenet, qa}` folders contains a worked example of conformal risk control with a different risk function.
`polyps` does gut polyp segmentation with false negative rate control. `coco` does multi-label classification with false negative rate control. `hierarchical-imagenet` does hierarchical classification and chooses the resolution of its prediction by bounding the graph distance to an ancestor of the true label. Finally, `qa` controls the F1-score in open-world question answering.

### Setup
For the computer vision experiments, run
```
  conda env create -f environment.yml
  conda activate conformal-risk
```
This will install all dependencies for the vision experiments.

For the question-answering task, follow the instructions in `qa/README.md`.

### Reproducing the experiments
After setting up the environment, enter the example folder and run the appropriate `risk_histogram.py` file.
To produce the grids of images in the paper, run the python file containing the word `grid` in each folder.

### Citation 

```
@article{angelopoulos2022conformal,
  title={Conformal Risk Control},
  author={Angelopoulos, Anastasios N and Bates, Stephen and Fisch, Adam and Lei, Lihua and Schuster, Tal},
  journal={arXiv preprint arXiv:2208.02814},
  year={2022}
}
```

