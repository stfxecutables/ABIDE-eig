# About

This directory is for attempts to replicate (NOT reproduce, just replicate) various ABIDE-related DL
approaches and papers, including:

* Eslami, T., Mirjalili, V., Fong, A., Laird, A. R., & Saeed, F. (2019). ASD-DiagNet: A Hybrid
  Learning Approach for Detection of Autism Spectrum Disorder Using fMRI Data. Frontiers in
  Neuroinformatics, 13, 70. https://doi.org/10.3389/fninf.2019.00070
  [link](https://www.frontiersin.org/articles/10.3389/fninf.2019.00070/full)
* Subah, F. Z., Deb, K., Dhar, P. K., & Koshiba, T. (2021). A Deep Learning Approach to Predict
  Autism Spectrum Disorder Using Multisite Resting-State fMRI. Applied Sciences, 11(8), 3636.
  https://doi.org/10.3390/app11083636
  [link](https://mdpi-res.com/d_attachment/applsci/applsci-11-03636/article_deploy/applsci-11-03636.pdf)
* Heinsfeld, A. S., Franco, A. R., Craddock, R. C., Buchweitz, A., & Meneguzzi, F. (2018).
  Identification of autism spectrum disorder using deep learning and the ABIDE dataset. NeuroImage:
  Clinical, 17, 16–23. https://doi.org/10.1016/j.nicl.2017.08.017
  [link](https://www.sciencedirect.com/science/article/pii/S2213158217302073)

## Motivation

Almost none of the above papers are even close to replicating, in that they report accuracies of
over 70.0%, but careful and thorough replication attempts often struggle to get close to even 60%.
E.g. Heinsfeld et al. report a 70.3% accuracy, but Eslami et. al. report their attempts to replicate
hit only 65%.

Subah et. al. report numerous accuracies *way* above 80%, but the models are ludicrously simple (and
the 80% dropout make no sense), and attempts to get close don't even begin to approach 55% accuracy
(with multiple repeats of the model showing that about half models generated with identical params
actually are worse than guessing).

The feature selection of Eslami et al. (choose the 50% of correlations with largest absolute mean
magnitude across all subjects) is nonsensical (high average values could imply no variance, so
selection on position might as well be random) and the model design also makes no sense. The use
of the autoencoder implies a fundamental misunderstanding of DL / autoencoders, as a learned
autoencoder bottleneck representation is learned only in service of minimizing *decoder error*, which
has no relation to classification error. Besides, any reduction to a layer with the same size as
the bottleneck is an equivalent / identical "learned representation", so the reasoning is spurious.
*If* the model even works (and I see no evidence for this), it is because the weight sharing and
autoencoder just have a regularizing effect overall.