**Main Problem**: There are really 2-3 papers here:

1. Exploring efficient Deep Learning architectures for fMRI
2. fMRI feature extraction (selection?) and validation
   - this would also include justifying eigenvalue-based features
3. ABIDE / ADHD-200 and generally small-data heterogeneity issues in DL

# What we Want to Do

Our goal for a paper is basically:

1. extract various features from fMRI (1D, 2D, 3D, 4D)
   - some are from other papers, some are eigenvalue-based
2. compare predictive models using these features
3. argue that eigenvalue features have some advantages




# Heterogeneity

# Complexity of Different Tests

# Biased Feature Selection in Published Papers

- this is a problem because if we extract any novel features or compare to previous
  papers, ours will appear worse without selection, unless we replicate the biased
  feature selection procedure
- a better paper would investigate the consequences of these biased feature-selection
  methods and propose solutions to avoid them

# Eigenvalues Are Bad Features

Two matrices $A$ and $B$ are similar if there exists an invertible matrix $M$ such that
$A = M^{-1} B M$, and we can write $A \sim B$. Similar matrices have identical eigenvalues,
but not identical eigenvectors.

Permutation matrices (which permute rows / columns of a matrix) are invertible, and so do not change
the eigenvalues of a matrix. Thus the `(n_features, n_features)` correlation matrix $C$ from the
timeseries of data with `(n_features, n_timepoints)`, yields identical eigenvalues to any
permutation of those $n$ features. In addition, we can only deal with sorted eigenvalues (there is
no "natural" ordering of eigenvalues since this would require a natural ordering for eigenvectors,
which are `n_features`-dimensional).

In particular, when the features are ROIs (a voxel can be considered a degenerate
ROI), this means two things:

1. Eigenvalues at eigenindex $i$ from one subject will not generally correspond to the same
   eigenvector from another subject
2. Eigenvalues will be identical for radically-different systems of correlations
   - e.g. two diagonal correlation matrices $C_1 = diag(1, ..., 1, 0, ..., 0)$,
     $C_2 = diag(0, ..., 0, 1, ..., 1)$ have identical eigenvalues

## A Worked Example

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

Then in this case the eigenvalues of the full system $M$ are the eigenvalues of $C$, $\lambda_1, ..., \lambda_n$, plus the eigenvalues
of $M^{\prime}$, $\xi_{n+1}, ..., \xi_{N}$.

When these eigenvalues are computed, ***they will be sorted ascending, and it will now be impossible to know
which eigenvalues correspond to $C$ and which to $M^{\prime}$***.

In addition, in actual fact not every subject would have the same $C$, but more like $C + \Sigma$. The eigenvalues
of this can also unfortunately be shuffled about (https://mathoverflow.net/a/4255) so that we can't be sure the
first eigenvalue returned by the algorithm corresponds to the "same" eigenvector.

Even if one tries to order the eigenvalues by the eigenvectors, we deal with the same problem. The
only way to order the eigenvectors would be
[lexicographically](https://en.wikipedia.org/w/index.php?title=Ordered_vector_space&oldid=1063366727#Examples_3),
but this is just the same problem, since at any eigenvector index $i$ there is "noise", and so this
ordering is noisy too.


Re point (1) however, it is probably reasonable to assume that the eigenvalues "around" eigenindex
$i$, that is, the eigenvalues in eigenindices $[i - \delta, i + \delta]$ correspond roughly to
"similar" eigenvalues across subjects. However, it is not reasonable to assume that $\delta$ is
independent of $i$, e.g. the largest eigenvalues (small $i$) (corresponding to the largest principle
components) likely have a similar source within some small $\delta$, whereas for the smallest
eigenvalues (large $i$), which correspond largely to noise componenents, it is possible that only
the general trend of these eigenvalues has information.

This means for fMRI, computing the this means the ordering of voxels is irrelevant, and thus if
there is a difference between individuals that is identifiable only due to a *specific location*,
then this is not detectable via eigenvalues. E.g. if ADHD subjects have increased signal
correlations in voxels 1 through 100 inclusive, but non-ADH have the same increased signal
correlations in voxels 100 through 200 inclusive, these correlation matrices look identical in terms
of eigenvalues. So global eigenvalues are insensitive to spatial differences in the functional
connectivity.

Eigenperturbation images do not have this problem, but instead suffer from a different problem, namely,
that the deletion of a single voxel (i.e. row/column pair of a correlation matrix)


