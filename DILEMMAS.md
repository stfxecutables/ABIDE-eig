# Main Problems: Too many Goals for One Paper

There are really 2-3 separable papers / goals here re: the eigenvalues and ABIDE /
ADHD-200 data:

1. exploring eigenvalue-based features (eigenfeatures) is doomed to fail
   - for both prediction and description
   - includes whole-brain, ROI-based, and perturbation-based eigenfeatures
2. Exploring efficient deep learning architectures for fMRI
3. Solving various serious general issues for both DL and classical ML with tiny, extremely
   heterogeneous data of ABIDE / ADHD-200

# Problem / Paper #1: Eigenfeatures are Bad / Limited Features

## Primary Issue: Eigenvalue Extraction is Worse than Linear

Given a correlation matrix $\mathbf{M}$ of size $n \times n$, then since $\mathbf{M}$ is symmetric, it is diagonalizable,
and thus we can eigendecompose $\mathbf{M}$ to

$$
\mathbf{M} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1}
$$

where $\mathbf{Q}$ is the eigenvectors of $\mathbf{M}$, and $\mathbf{\Lambda}$ is diagonal with
the diagonal entries being the $n - 1$ non-zero eigenvalues of $\mathbf{M}$. That is, we can write:

$$
\begin{aligned}


\mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1} &= \mathbf{M} \\
\mathbf{Q}^{-1} \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1} &= \mathbf{Q}^{-1} \mathbf{M} \\
\mathbf{\Lambda} \mathbf{Q}^{-1} &= \mathbf{Q}^{-1} \mathbf{M} \\
\mathbf{\Lambda} \mathbf{Q}^{-1} \mathbf{Q} &= \mathbf{Q}^{-1} \mathbf{M} \mathbf{Q} \\
\mathbf{\Lambda}  &= \mathbf{Q}^{-1} \mathbf{M} \mathbf{Q} & \qquad (1)\\
\end{aligned}
$$

which, for illustration, might be better illustrated as

$$
\begin{aligned}
eigs(\mathbf{M})  &= \mathbf{\Lambda}^\intercal  = (\mathbf{Q}^{-1} \mathbf{M} \mathbf{Q})^\intercal \\
&=  (\mathbf{Q}^{-1} \mathbf{M} \mathbf{Q})^\intercal \\
&=  \mathbf{Q}^\intercal (\mathbf{Q}^{-1} \mathbf{M})^\intercal & \qquad (2) \\
&=  \texttt{Linear}_{\mathbf{Q}}((\texttt{Linear}_{\mathbf{Q}^{-1}}(\mathbf{M}))^\intercal) \\
&=  (\texttt{Linear}_{\mathbf{Q}} \circ transpose \circ \texttt{Linear}_{\mathbf{Q}^{-1}})(\mathbf{M})
\end{aligned}
$$

i.e., the function which implements eigenvalue extraction of $\mathbf{M}$ can be implemented as two
matrix multiplications (e.g. linear operations) parameterized by the weights of $\mathbf{Q}$, with an
intermediate transposition. Transposition is linear and so can itself be re-written as a ($n^2 \times n^2$) matrix multiplication of a
particular kind, (e.g. https://math.stackexchange.com/a/1143642), and this matrix-representation of the
transpose is the same for any choice of matrix in $\mathbf{\mathbb{R}}^{n \times n}$, so we might
rewrite the above as:

$$
\begin{aligned}
eigs(\mathbf{M}) &=  (\texttt{Linear}_{\mathbf{Q}} \circ transpose \circ \texttt{Linear}_{\mathbf{Q}^{-1}})(\mathbf{M}) \\
&= \mathbf{A}_{\mathbf{Q}} \mathbf{M}
\end{aligned}
$$

For some matrix $\mathbf{A}_{\mathbf{Q}} \in \mathbb{R}^{n \times n}$.  In the more general case, a
nearly identical argument to above can be given by rewriting $\mathbf{M}$ with the singular value
decomposition:

$$
\mathbf{M}  = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^*.
$$

And noting that the complex conjugation operation is also linear. ***This means that each eigenvalue is ultimately
some <u>linear</u> combination of the values of $\mathbf{M}$***.



## Second Issue: Eigenvalues *a priori* Cannot be (and Empirically are not) Very Predictively-Useful for fMRI

This is something I have now seen multiple times empirically on a wide variety of datasets (most too
small to matter, but still). Cross-validated or holdout prediction accuracies are usually about 2-3%
better than guessing, maybe 5-7% if you "hypertune" by testing millions of combinations (i.e. just
overfit, perhaps). This has been the case even when I have tried different normalization strategies,
preprocessing strategies, etc., and at least for the ABIDE data, this means you are looking at validation
prediction accuracies of under 60% (compared to dubious literature reported ~71% best).

### Why?

Two matrices $A$ and $B$ are similar if there exists an invertible matrix $M$ such that
$A = M^{-1} B M$, and we can write $A \sim B$. Similar matrices have identical eigenvalues,
but not identical eigenvectors.

Permutation matrices (which permute rows / columns of a matrix) are invertible, and so do not change
the eigenvalues of a matrix. Thus the `(n_features, n_features)` correlation matrix $C$ from the
timeseries of data with `(n_features, n_timepoints)`, yields identical eigenvalues to any
permutation of those $n$ features. In addition, we can only deal with sorted eigenvalues (there is
no "natural" ordering of eigenvalues since this would require a natural ordering for eigenvectors,
which are `n_features`-dimensional).

When the features are correlations of fMRI ROIs (a voxel can be considered a degenerate
ROI), this means two things:

1. Eigenvalues at eigenindex $i$ from one subject will not generally correspond to the same
   eigenvector from another subject
2. Eigenvalues will be identical for radically-different systems of correlations
   - e.g. two diagonal correlation matrices $C_1 = diag(1, ..., 1, 0, ..., 0)$,
     $C_2 = diag(0, ..., 0, 1, ..., 1)$ have identical eigenvalues


#### A Simple Failure Case

Point (2) is the most concerning for doing classification from fMRI.

Suppose that autism is characterized by some distinct pattern of correlations on some set of $n$
ROIs. We might represent this pattern of correlations as a submatrix $C$ of size $n \times n$, and
say that we generally observe the $n$ ROIs have a correlation matrix "close" to $C$, whereas
controls tend to be "farther" from C. Suppose also that these ROIs are relatively uncorrelated with
other ROIs, so that we can represent the full $N \times N$ correlation matrix $M$ across all ROIs as a block matrix
containing $C$

$$
M = \begin{pmatrix}
  C & \vert & \mathbf{\Sigma} \\
\hline
  \mathbf{\Sigma}^{\top} & \vert & M^{\prime}
\end{pmatrix}
$$

The matrix $\Sigma$ will be a random matrix with relatively small values, and probably have eigenvalues
similar to the GOE. For now, let us just pretend $\Sigma = \mathbf{0}$ to show what would happen in
this even simpler case, e.g. pretend we have something like

$$
M = \begin{pmatrix}
  C & \vert & \mathbf{0} \\
\hline
  \mathbf{0}^{\top} & \vert & M^{\prime}
\end{pmatrix}
$$

Then in this case the eigenvalues of the full system $M$ are the eigenvalues of $C$ plus the
eigenvalues of $M^{\prime}$, .  When these eigenvalues are computed, they will be sorted ascending
all together, ***and it will now be impossible to know which eigenvalues correspond to $C$ and which
to $M^{\prime}$***. This means that when fed into an algorithm as a list of features, **the meaning
of feature (eigenvalue) $i$ can and often *will* be different for every subject**.

In reality, also not every subject would have the same $C$, but more like some correlation matrix
$C + \Sigma$, where $\Sigma$ is specific to that subject. The eigenvalues of this perturbed matrix can
also unfortunately be shuffled about (https://mathoverflow.net/a/4255) so that *even if we somehow
already knew how to just identify the features that contribute to $C$, we still couldn't be sure the
first eigenvalue of $C + \Sigma$ returned by the algorithm corresponds to the first eigenvalue of
$C$*, and we still have the problem of features having different interpretations from subject to
subject.

This means eigenvalues are useful as features only insofar as

1. the eigenvalues clustered around eigenindex $i$, that is, the eigenvalues in eigenindices $[i -
   \delta, i + \delta]$ correspond roughly to "similar" eigenvalues across subjects, and
2. the algorithm is flexible enough to learn how to combine / summarize / find such clusters

However, it is not reasonable to assume that all clusters are the same size, e.g. that $\delta$ is
independent of $i$. The largest eigenvalues (small $i$) (corresponding to the largest principle
components) possibly have a similar source within some small $\delta$, whereas for the smallest
eigenvalues (large $i$), which correspond largely to noise components, it is possible that only the
general trend of these eigenvalues has information. E.g. perhaps the smallest 50 eigenvalues can be
summarized with a single "noise" value, but perhaps the largest 20 eigenvalues are more like 5-10
clusters. But then again, perhaps not. This is all hard to justify.

Thus using fixed windows on the eigenvalues will be suboptimal (e.g. Conv1D). Methods that use the
entire spectrum unprocessed (e.g. MLP, RandomForest, LSTM), *might* be able to learn / find combinations
that work well for prediction, except...

### Eigenvalues are Especially bad for Heterogeneous Data (e.g. ABIDE, fMRI, MRI generally)

***ABIDE is heterogenous, and the spectra across sites are clearly, obviously dramatically visually
different, and the clusters are also clearly, obviously different***. Thus, even if some clustering is
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


