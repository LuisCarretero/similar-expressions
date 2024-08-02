
## Generative model to create algebraic expressions with similar semantics to use with Symbolic Regression



### Specification
- Start univariate
- Loss function: Based on evaluation of the expression uniform distribution of values over some range (will later add an actual dataset and noise)
- Model architecture: Copy the grammar-vae architecture for now.


### TODO
- Create a dataset of expressions (with similar semantics)
- Create max seq length (calc from tree size). Use padding token



### Ideas
- Use Numpy expand and similar to create equivalent expressions
- Use methods from https://arxiv.org/pdf/1912.01412 to create better datasets
