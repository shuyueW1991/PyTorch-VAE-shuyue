# Line-wise understanding of VAE in code and math:  in Socratic way.
With some updates now and then.
The understanding is led by Socratic query & answers.


## Why AE in the first place?
- Autoencoder, by compression, provides a data transformer to higher dimension, where: 
    - many solutions can be achieved other than in low-dimension world. (E.g. classification);
    - computation economy, with `irrelevant` information being neglected.


## AE is just another `main-feature` extractor, right?
Let's take PCA (Principal Component Analysis) and POD (Proper Orthogonal Decomposition) as a comparable target in this discuss, since it is also dimensionality reduction techniques.

Now, PCA is quite different from AE (Autoencoder):
- PCA: Linear transformation. It finds a linear combination of the original features that best captures the variance in the data. 
    - Imagine stretching and rotating a high-dimensional cloud of data points to a lower-dimensional space while preserving as much of the spread as possible.
    - The new features obtained by PCA is usually one dimension less than original data.

POD is also a feature extractor in that:
- POD: Suppose we have a data matrix X (n rows representing data points and m columns representing features).
    - We calculate the covariance matrix C of X, which captures the relationships between features.
        - Calculate the mean vector for each column: μ = (mean(X[:,0]), mean(X[:,1]), ..., mean(X[:,n-1])). 
            - $μ$ is a row vector, obviously.
        - Center the data: $X_{centered} = X - μ$
        - Calculate the covariance matrix: $C = (X_{centered}.T) * X_{centered}$ 
    - We perform eigenvalue decomposition on C: $C = VΛV^T$, where:
        - V is a matrix containing the eigenvectors (modes) as columns.
        - Λ is a diagonal matrix containing the corresponding eigenvalues.
    The columns of V represent the orthogonal basis vectors (modes) of variability.
    The diagonal entries of Λ represent the eigenvalues, which quantify the variance captured by each mode. 
    Ordering the eigenvalues in descending order gives the most significant modes first.

Whilst...
- AE: Non-linear transformation through a neural network. It uses an encoder-decoder structure. The encoder compresses the data into a lower-dimensional latent representation, and the decoder tries to reconstruct the original data from this compressed version. 
    - Think of it like squeezing the data points through a bottleneck and then trying to inflate them back to their original shapes.
    - The extracted features in AE are of dimensions assigned beforehand.


### Why VAE against AE?
This [blog](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) illustrates so good that:
- AutoEncoder lacks `regularisation` in generative process. 
- `regularisation` provides a “gradient” over the information encoded in the latent space, instead of simply overfitting on trained data.
- Need to implement this genre of `regularisation` in loss function.
- The `regularisation` in VAE is to make the input of certain data to be a probability.


## How is that probability getting involved in VAE?
- The input is encoded as distribution over the latent space
- a point from the latent space is sampled from that distribution
- the sampled point is decoded and then can be computed in a “reconstruction term” (`on the final layer`) that tends to make the encoding-decoding scheme as performant as possible, and a “regularisation term” (`on the latent layer`) that tends to regularise the latent space by making the distributions returned by the encoder close to a standard normal distribution. 
- That regularisation term is expressed as the Kulback-Leibler divergence between the returned distribution and a standard Gaussian.
- The error is backpropagated through the network.
- The regularity that is expected from the latent space in order to make generative process possible can be expressed through two main properties: 
    - continuity (two close points in the latent space should not give two completely different contents once decoded); 
    - completeness (for a chosen distribution, a point sampled from the latent space should give “meaningful” content once decoded).
- Some notations on VAE math:
    - $p(z)$: a latent representation, aka prior ;
    - $p(x|z)$: conditional likelihood distribution, probabilistic decoder，aka likelihood;
    - $p(z|x)$: likewise, conditional likelihood distribution, probabilistic encoder, aka posterior.
- Put them together: $$p(z|x) = \frac{p(x|z)p(z)}{p(x)} = \frac{p(x|z)p(z)}{\int{p(x|u)p(u)du}}$$.
- And the task is to calculate the $p(z|x)$.
- If we came to get this, we then have latent vector $z$ that consists of the `regularity` lacking in AE approach. Then, imagine you point your finger stiring in the realm of the distribution of the latent vector $z$, all of its matches through decoder would be anything less than  a `normal` thing. That's why we care about $p(z|x)$.


## Describe the loss in VAE training, like what it is after all.
- Assumption: 
    - $p(z)$ is a standard Gaussian distribution  
    - $p(x|z)$ is a Gaussian distribution:
        - whose mean is defined by a deterministic function f of the variable of z  
            - The function f is assumed to belong to a family of functions denoted F 
        - whose covariance matrix has the form of a positive constant c that multiplies the identity matrix I. 
    - Therefore: 
        $$p(z) = \mathcal{N}(0,I)$$ 
        $$p(x|z) = \mathcal{N}(f(z),cI)$$
- The integral in the dominator of the $p(z|x)$ makes the problem intractable.
- Thus requires the use of approximation techniques such as variational inference (VI), a technique to approximate complex distributions.
    - The idea is to set a parametrised family of distribution (for example the family of Gaussians, whose parameters are the mean and the covariance) and to look for the best approximation of our target distribution among this family. 
    - The best element in the family is one that minimise a given approximation error measurement (most of the time the Kullback-Leibler divergence between approximation and target) and is found by gradient descent over the parameters that describe the family.
    - [This blog](https://towardsdatascience.com/bayesian-inference-problem-mcmc-and-variational-inference-25a8aa9bce29) illustrates pretty well the variational inference.
- In order to calculate $p(z|x)$, we have the $p(z)$ and the $ p(x|z)$ in their assumed forms.
- The rest is, as matter of fact, $p(x|z)$ by itself. For sure, it can also make life easier by assuming it of some sort of form.
- But if so, there would be no necissity to do the Bayes formula calculation.
- Therefore, just go ahead to approximate $p(z|x)$ by a Gaussian distribution $q_x(z)$ whose mean and covariance are defined by two functions, g and h, of the parameter x. These two functions are supposed to belong, respectively, to the families of functions G and H: $$q_x(z) = \mathcal{N}(g(x),h(x))$$. 
- In the spirit of getting a close distribution to that of the real $p(x|z)$, we look for the $g^*$, and $h^*$ s.t. (many thanks to [this colab](https://colab.research.google.com/drive/1_yGmk8ahWhDs23U4mpplBFa-39fsEJoT?usp=sharing#scrollTo=JsNG8hlsonO1)): 
    $$\mathbf{D}_{KL}(q_x(z) | p(z(x))) \\
    = \sum_{z} q_x(z) \log_{p(z|x)}{q_x(z)} \quad \text{Defintion}\\
     = \sum_{z} q_x(z)[ \log{q_x(z)} - (\log{p(x|z)} + \log{p(z)} - \log{p(x)}) ] \quad \text{open the log fraction}\\
     \Rightarrow \quad \\
     \mathbf{D}_{KL}(q_x(z) | p(z(x))) - \sum_{z}{q_x(z)[\log{q_x(z)} - \log{p(x|z)} - \log{p(z)}]} = \sum_{z}{q_x(z)\log{p(x)}} \quad \text{which is a CONSTANT.}
    $$
- Since,$\sum_{z}{q_x(z)[\log{q_x(z)} - \log{p(x|z)} - \log{p(z)}]}$ by definition can be expressed as:
    $$
    \mathbf{E}_{z \sim q_x(z)}{[- \log{p(x|z)} + \log{\frac{q_x(z)}{p(z)}}]}
    $$, 
    we can thus have:
    $$ 
        {KL(q_x(z), p(z|x))} \quad \text{Task of comparing the real and fake latent distribution, which is the LOSS we want to minimize}\\
        = \mathbf{E}_{z \sim q_x(z)}{[- \log{p(x|z)} + \log{q_x(z)}  - \log{p(z)} ]} \quad \text{According to the above deductions.}\\
        = \mathbf{E}_{z \sim q_x(z)}{[ \log{q_x(z)}  - \log{p(z)} ]} - \mathbf{E}_{z \sim q_x(z)}{\log{p(x|z)}}
    $$, where the first expectation can be viewed as `KL divergence loss` while the second expectaion as `reconstruction loss` (if you remember $q_x(z)$ is actually a fake $p(z|x)$, then the expectation is a quasi $p(x)$.
- Due to the magic bonds between all those terms in the Bayesian relation and in encoder-decoder paradigm, there is no more $p(z|x)$ in the `loss` that we will be doing with.







    










### Describe the latent and input/output data/dimension/variables in the inductive bias of AE/VAE architecture.




### Why pytorch.lightning in this repo?
- mps architecture is not supported in vanilla torch.
- fancier way of training/fit implement.




