# Main Problems: Too many Goals for One Paper

There are really 2-3 separable papers / goals here re: the eigenvalues and ABIDE /
ADHD-200 data:

1. eigenvalue-based features (eigenfeatures), which:
   - must necessarily be poor for both prediction and description
   - will be poor features whether extracted from the whole-brain or ROIs
   - will not be salvageable with perturbation-based methods
2. Exploring efficient deep learning architectures for fMRI
3. Solving various serious general issues for both DL and classical ML with tiny, extremely
   **heterogeneous data** (e.g. ABIDE / ADHD-200, other medical or subject-level / panel data)

# Problem / Paper #1: Eigenvalues are Bad / Limited Features

Broadly, the problem here is that eigenvalues have a number of properties that render them
undesirable for prediction. The main problems are their quasi-linearity, non-locality, and
permutation insensitivity at multiple levels.

## Primary Issue: Eigenvalue Extraction is "Quasi-Linear"

Given a correlation matrix $\mathbf{M}$ of size $n \times n$, then since $\mathbf{M}$ is symmetric, it is diagonalizable,
and thus we can eigendecompose $\mathbf{M}$ to

$$ \mathbf{M} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1} $$

where $\mathbf{Q}$ is the eigenvectors of $\mathbf{M}$, and $\mathbf{\Lambda}$ is diagonal with
$n - 1$ or fewer non-zero eigenvalues of $\mathbf{M}$. That is, we can write


$$ \mathbf{M} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1}  $$

or equivalently

$$ \mathbf{\Lambda}  = \mathbf{Q}^{-1} \mathbf{M} \mathbf{Q} $$

which, for illustration, might be better illustrated as

$$\begin{aligned}
eigs(\mathbf{M}) = eigs(\mathbf{M^{\intercal}})  &= \mathbf{\Lambda}^\intercal \\
&= (\mathbf{Q}^{-1} \mathbf{M} \mathbf{Q})^\intercal \\
&=  (\mathbf{Q}^{-1} \mathbf{M} \mathbf{Q})^\intercal \\
&=  \mathbf{Q}^\intercal (\mathbf{Q}^{-1} \mathbf{M})^\intercal & \qquad (2) \\
&=  \texttt{Linear}_{\mathbf{Q}}((\texttt{Linear}_{\mathbf{Q}^{-1}}(\mathbf{M}))^\intercal) \\
&=  (\texttt{Linear}_{\mathbf{Q}} \circ transpose \circ \texttt{Linear}_{\mathbf{Q}^{-1}})(\mathbf{M})
\end{aligned}$$

i.e., the function which implements eigenvalue extraction of $\mathbf{M}$ can be seen as two
matrix multiplications (e.g. linear operations) parameterized by the weights of $\mathbf{Q}$, with an
intermediate transposition. Transposition is linear and so can itself be re-written as a ($n^2 \times n^2$) matrix multiplication of a
[particular kind](https://math.stackexchange.com/a/1143642), and this matrix-representation of the
transpose is the same for any choice of matrix in $\mathbf{\mathbb{R}}^{n \times n}$, so we might
rewrite the above as:

$$\begin{aligned}
eigs(\mathbf{M}) &=  (\texttt{Linear}_{\mathbf{Q}} \circ transpose \circ \texttt{Linear}_{\mathbf{Q}^{-1}})(\mathbf{M}) \\
&= \mathbf{A}_{\mathbf{M}} \mathbf{M}
\end{aligned}$$

For some matrix $\mathbf{A}_{\mathbf{M}} \in \mathbb{R}^{n \times n}$. Alternately, we might note that it is ultimately
arbitrary whether we decide to write linear transformations using left vs. right multiplications, so if we define

$$ f(\mathbf{A}) = \mathbf{Q}^{-1}\mathbf{A}; \quad g(\mathbf{A}) = \mathbf{A}\mathbf{Q}$$

then $f$ and $g$ are still linear functions, and $\mathbf{\Lambda} = (f \circ g)(\mathbf{M})$, and
so this linear transformation $f \circ g$ also has a matrix representation (and it is in fact
$\mathbf{A}_{\mathbf{M}}$).  ***This means that each eigenvalue of $\mathbf{M}$ is ultimately some
<u>linear</u> combination of $n$ the values of $\mathbf{M}$***^[In the more general case, a
similar argument to above can be given by rewriting $\mathbf{M}$ with the singular value
decomposition, albeit with the domain being the complex numbers, and noting that the complex
conjugation operation is also linear.]. That is,

$$ \lambda_{i} = \sum_{i=1}^n a_{ij}m_{ji}$$

Granted, the $eigs$ operator itself is highly non-linear (almost nothing in general can be said
about $eigs(\mathbf{A} + \mathbf{B})$), and so the values $a_{ij}$ of course depend heavily on
$\mathbf{M}$ in a non-linear way, but this is just a sketch of the basic intuition that eigenvalues
are tools that arise from *linear* equations and, and so the *eigenvalues* are in some sense
"fundamentally linear". Most importantly, ***when eigenvalues are used to summarize that system,
they produce a linear summary of that system***.

Another way to illustrate this might be from the basic equation of an eigenvalue/eigenvector pair,
$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$, where $\mathbf{A} \in \mathbb{R}^{n \times n}$. We can
write:

$$\begin{aligned}
\mathbf{A}\mathbf{v} &= \lambda \mathbf{v} \\
\mathbf{v}^{\intercal} \mathbf{A}^{\intercal} = (\mathbf{A}\mathbf{v})^{\intercal} &= (\lambda \mathbf{v})^{\intercal} = \lambda \mathbf{v}^{\intercal} \\
\mathbf{v}^{\intercal} \mathbf{A}^{\intercal} \mathbf{v} &= \lambda \mathbf{v}^{\intercal} \mathbf{v}\\
\mathbf{v}^{\intercal} \mathbf{A}^{\intercal} \mathbf{v} &= \lambda \Vert\mathbf{v}\Vert^2 \\
\mathbf{v}^{\intercal} \mathbf{A}^{\intercal} \mathbf{v} &= \lambda \quad \text{since we usually require } \Vert\mathbf{v}\Vert = 1 \\
\mathbf{v} \mathbf{A} \mathbf{v}^{\intercal} &= \lambda \quad \text{after transposing again} \\
\end{aligned}$$

So $\lambda$ is clearly a big long *linear* combination of $v_i$s and $a_{ij}$s.

### The Quasi-Linear Extraction is *too* Global for MRI/fMRI (Human brain)

For fMRI, the above properties have some unfortunate consequences. In fMRI, we start with a matrix
$\mathbf{M}$ with shape `(N, T)`, where `N` is the number of voxels or ROIs under consideration
(e.g. after brain masking), and `T` is the number of timepoints. If using some ROI reductions, each
ROI time-series is summarized by some kind of pooling / aggregating function (e.g. mean, median,
max). Typically, $\texttt{T} \in [80, 300]$, $\texttt{N} \le 200$ if using anatomical ROIs, and `N`
is quite large, on the order of $64^3$ to $128^3$ or so, if using voxels. We are interested in the
eigenvalues of the correlation (or covariance) matrix of $\mathbf{M}$, which we can represent with
$\mathbf{R}$ (or $\mathbf{C}$)^[The correlation matrix $\mathbf{R}$ is just the covariance matrix
of $\mathbf{\tilde{M}}$, where $\mathbf{\tilde{M}}$ has been feature-wise standardized].

In the full-voxel case, $\texttt{N} \gg \texttt{T}$ and so we must / do use $\mathbf{M}^\intercal$
to compute the eigenvalues in an efficient manner. However, this means we deal with the $\texttt{T}
\times \texttt{T}$ matrix $\mathbf{R} = \mathbf{M}^{\intercal} \mathbf{M}$. Already, we can see that
when `N` is large, most elements of $\mathbf{R}$ will be a linear combination of most elements of
$\mathbf{M}$. That is, $\mathbf{R}^{\intercal}_{ij}$ is a linear combination of *all* voxel values
at time $i$ with *all* voxel values at time $j$.

Per the arguments in the section above, the *eigenvalues* of $\mathbf{R}$ will again each be linear
combinations of elements of $\mathbf{R}$, which are again linear combinations of the elements of the
original fMRI $\mathbf{M}$. (Already one  should be getting a feel for the problem here - we are
summing too many things).

Now, most correlations will not be zero, as that is extremely unlikely simply due to noise /
numerical imprecision alone, and because we are dealing with time series (which almost always show
some non-zero correlation) and because we know in general fMRI tends to have global patterns (e.g. a
global mean signal, frequency components related to noise, breathing, heartrate, etc). So
$\mathbf{R}$ will have $q = \min(\texttt{N}, \texttt{T}) - 1$ nonzero (and effectively distinct)
eigenvalues, i.e. $\mathbf{R}$ has rank $q$, i.e. is "almost" orthogonal. However, if we don't get
this distinctness (basically impossible) it would be because some time series are an exact linear
combination of some other ROIs' timeseries, and we could simply reduce $\mathbf{M}$ to the linearly
independent timeseries, and still be in this same case. Likewise, if we did not subtract the means
and dealt with the [autocorrelation matrix](https://en.wikipedia.org/wiki/Autocorrelation#Matrix),
the rank would just be $q$.

So $\mathbf{R}$ will be full-rank, or nearly-full-rank, and so, since it is also symmetric,
the eigenvectors $\mathbf{Q}$ will be either full-rank and orthogonal such that
$\mathbf{Q}^\intercal = \mathbf{Q}^{-1}$, or will "nearly" have this property (e.g. the size $q - 1$
block matrix / reduction will have this property). Thus intuitively, there is a hard limit on the
amount of "zeros" or "near-zero" values in $\mathbf{Q}$.  In fact, we can prove this more precisely by noting $\mathbf{R}$ is symmetric / Hermitian,
and using the [eigenvector-eigenvalue identity](https://arxiv.org/abs/1908.03795).

The argument I am trying to make here, with a lot of handwaving, is that virtually no elements of
the eigenvectors $\mathbf{Q}$ will have magnitude zero, and so **_each_ eigenvalue of $\mathbf{R}$
is an enormous weighted sum of _every single voxel / ROI of the fMRI matrix_ $\mathbf{M}$**.
I argue this means the vector of eigenvalues $\mathbf{\Lambda}$ is a ***very bad*** feature for
prediction, because $\mathbf{\Lambda}$ contains *only global information* from the image, and
***local information / features cannot be constructed / reverse-engineered from $\mathbf{\Lambda}$***.

The only way to preserve local information is if the brain is partitioned into ROIs, and then the
eigenvalues of the correlation matrices of each of those ROIs are computed, so that `N` above is
the count of ROIs. However, as we see below, this also has problems.

## Second Issue: Eigenvalues Are too Permutation-Insensitive

### Eigenvalue Indices do not Have the Same Meaning Across Matrices

Given two matrices / fMRI scans with eigenvalues $\mathbf{\Lambda}$ and $\mathbf{\Lambda}^{\prime}$,
because the eigenvectors are different in each case, then the distribution of the contributions to
e.g. $\lambda_i$ will in general be quite different from the distribution of the contributions to
$\lambda_i^{\prime}$, so that the **components of $\mathbf{\Lambda}$ do not have a consistent
meaning from subject to subject**^[If this is not clear, consider eigenvalue extraction to be like
PCA which is done per-subject. Of each principal component will have a different meaning in each
subject.]

E.g. we might hope that $\mathbf{\Lambda}$ is like a `T`-dimensional embedding of the fMRI, such
that if $\vert \mathbf{\Lambda}^\prime - \mathbf{\Lambda} \vert$ is small, then $\mathbf{M}$ is in
some sense similar to $\mathbf{M}^{\prime}$. But this is not the case. This is further confounded by
eigenvalue sorting (algorithms return eigenvalues sorted by eigenvalue magnitude, because there is
no "natural" ordering possible here).

At best, we can hope that the largest eigenvalue(s) corresponds to something like a global / mean
signal shared across most signals of an fMRI, that the smallest eigenvalues correspond to noise, and
that intermediate eigenvalues capture some other phenomena. Roughly, we can only sort of *hope* that
the meaning of $\lambda_i$ is similar to the meaning of $\lambda_j^{\prime}$ only in some region
$\delta_{\min} < \vert i - j \vert < \delta_{\max}$.

In fact, I saw this empirically: permutation-invariant networks cannot overfit the voxel
eigenvalues at all, linear models gradually / steadily show decreasing loss approaching zero, and
convolutional models (which have a bias toward detecting local features) most rapidly overfit. There
is also weak evidence from my experiments that a larger kernel (7 vs 3) perform better, suggesting
that the useful features in the eigenvalues are local, but not too local, i.e. $\delta_{\min} > 3$.

However, it is not reasonable to assume that that $\delta_\min$ and $\delta_\max$ are fixed /
independent of the eiegenindex $i$. The largest eigenvalues (small $i$) (corresponding to the
largest principle components) possibly have a similar source within some small $\delta_\min$, whereas for
the smallest eigenvalues (large $i$), which correspond largely to noise components, it is possible
that only the general trend of these eigenvalues has information. E.g. perhaps the smallest 50
eigenvalues can be summarized with a single "noise" value, but perhaps the largest 20 eigenvalues
need more like 5-10 values to be summarized. But then again, perhaps not. This is all hard to justify.


### Eigenvalues Are Fundamentally Permutation-Invariant

Two matrices $A$ and $B$ are similar if there exists an invertible matrix $M$ such that $A = M^{-1}
B M$, and we can write $A \sim B$. Similar matrices have identical eigenvalues, but not identical
eigenvectors. *Permutation matrices* (which permute rows / columns of a matrix) are invertible, and
so do not change the eigenvalues of a matrix.  Thus the `(n_features, n_features)` correlation
matrix $C$ from the timeseries of data with `(n_features, n_timepoints)`, yields identical
eigenvalues to *any* permutation of those $n$ features.

Unfortunately, this means radically-different systems of correlations can produce identical eigenvalues.



#### A Simple Failure Case

A simple, but empirically-unlikely failure case are the two diagonal correlation matrices

$$ C_1 = diag(\lambda_1, \dots, \lambda_n, 0, ..., 0), C_2 = diag(0, ..., 0, \lambda_1, \dots,  \lambda_n) $$

that are completely different but have identical eigenvalues. But this can be generalized to cases that
*a priori*, are very likely theoretically, and less simplistic.

Suppose for example that autism is characterized by some distinct pattern of correlations on some
subset of ROIs, where the number of those ROIs is $n$. We might represent this pattern of
correlations as a submatrix $A$ of size $n \times n$, and say that we generally observe the $n$ ROIs
have a correlation matrix "close" to $A$, whereas controls tend to be "farther" from $A$. Suppose also
that these ROIs are relatively uncorrelated with other ROIs, so that we can represent the full $N
\times N$ correlation matrix $M$ across all ROIs as a block matrix containing $A$ as

$$
M = \begin{pmatrix}
  A & \vert & \mathbf{\Sigma} \\
\hline
  \mathbf{\Sigma}^{\top} & \vert & C
\end{pmatrix}
$$

where $\Sigma$ will be a random matrix with relatively small values (likely with eigenvalues similar
to the GOE), and where $C$ is a pattern of correlations common to controls and autistic individuals
both. For now, let us just pretend $\Sigma = \mathbf{0}$ to show what would happen in this even
simpler case, e.g. pretend we have something like

$$
M = \begin{pmatrix}
  A & \vert & \mathbf{0} \\
\hline
  \mathbf{0}^{\top} & \vert & C
\end{pmatrix}
$$

Then in this case the eigenvalues of the full system $M$ are the eigenvalues of $A$ plus the
eigenvalues of $C$.  When these eigenvalues are computed, they will be sorted ascending
all together, ***and it will now be impossible to know which eigenvalues correspond to $A$ and which
to $C$*** (they will be interleaved in unpredictable ways depending on noise).

Worse, if some collection of ROIs $A$ produces a particular distribution of eigenvalues only in e.g.
autistic subjects, but some *other* collection of ROIs $C$ has a similar distribution of
eigenvalues, then these again get mixed together in a much worse way. The only way that eigenvalues
can clearly identify a difference is if there large-scale, global differences in the correlation
patterns, but as per above, other methods (e.g. deep learning with the fMRI directly) are *much*
more likely to be able to find such patterns reliably.

Essentially, if something like "in subgroup 1, region A is strongly correlated with region B, and
weakly with region C, whereas in subgroup 2, the AB and AC correlation strengths are reversed" is
true, eigenvalues cannot see this.


### Eigenvalues are Especially bad for Heterogeneous Data (e.g. ABIDE, fMRI, MRI generally)

Cross-validated or holdout prediction accuracies using eigenvalue features are usually about 2-3%
better than guessing, maybe 5-7% if you "hypertune" by testing millions of combinations (i.e. just
overfit, perhaps). This has been the case even when I have tried different normalization strategies,
preprocessing strategies, etc., and at least for the ABIDE data, this means you are looking at validation
prediction accuracies of under 60% (compared to dubious literature reported ~71% best).

***ABIDE is heterogenous, and the spectra across sites are clearly, obviously dramatically visually
different, and the clusters are also clearly, obviously different***. This is largely due differences
in the underlying sampling resolutions (spatial and temporal) and sampling durations.

Thus, even if some clustering is
predictively useful for site A, it will be likely suboptimal (or even entirely useless) for site B.


Even if one tries to order the eigenvalues by the eigenvectors, we deal with the same problem. The
only way to order the eigenvectors would be
[lexicographically](https://en.wikipedia.org/w/index.php?title=Ordered_vector_space&oldid=1063366727#Examples_3),
but this is just the same problem, since at any eigenvector index $i$ there is "noise", and so this
ordering is noisy too.


# What we Want to Do

Our goal for a paper is basically:

1. extract various features from fMRI (1D, 2D, 3D, 4D)
   - some are from other papers, some are eigenvalue-based
2. compare predictive models using these features
3. argue that eigenvalue features have some advantages




# Heterogeneity

# Time Complexity of Different Tests

# Biased Feature Selection in Published Papers

- this is a problem because if we extract any novel features or compare to previous
  papers, ours will appear worse without selection, unless we replicate the biased
  feature selection procedure
- a better paper would investigate the consequences of these biased feature-selection
  methods and propose solutions to avoid them



# Summary

This means for fMRI, computing the this means the ordering of voxels is irrelevant, and thus if
there is a difference between individuals that is identifiable only due to a *specific set of
locations*, then this is *probably* not detectable via eigenvalues. So global eigenvalues are
insensitive to spatial differences in the functional connectivity.

Eigenperturbation images inherit all these problems. Namely, the eigenindex dimension is plagued by
the ordering issues discussed above, e.g. eigenimage "channels" mean different things from subject
to subject, which makes learning very difficult.


