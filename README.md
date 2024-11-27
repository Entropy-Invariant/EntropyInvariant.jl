# EntropyInfo

[![Build Status](https://github.com/felix.servant/EntropyInfo.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/felix.servant/EntropyInfo.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Build Status](https://gitlab.com/felix.servant/EntropyInfo.jl/badges/master/pipeline.svg)](https://gitlab.com/felix.servant/EntropyInfo.jl/pipelines)
[![Coverage](https://gitlab.com/felix.servant/EntropyInfo.jl/badges/master/coverage.svg)](https://gitlab.com/felix.servant/EntropyInfo.jl/commits/master)

This code is an improved nearest neighbor method for estimating differential entropy[^1] for continuous variables, invariant under change of variables, and positive. This approximation claim to solve the limiting density of discrete points formulated by Edwin Thompson Jaynes[^2]. All the details of the estimation can be found on the paper[^4].

The entropy H(X) is defined as:

H(X) = - ∫ₓ p(x) log(p(x) / m(x)) dx

The main novelty is m(x), the invariant measure. We introduce a proposition to describe this measure[^4] with the following properties:

1\) m(X) = r_X > 0
2\) m(aX) = am(X) = ar_X
3\) m(X + b) = m(X) = r_X

We found that the median value of the nearest-neighbor distance of each point is an appropriate measure for these properties.

The nearest-neighbor estimation was initialy adapted by G Varoquaux[^3] in python from this paper[^1].

In this package you can find other information theory quantities such as **joint entropy**, **conditional entropy**, **mutual information**, **conditional mutual information** and **interaction of information**. All based on a generalisation of limiting density of discrete points for more than one variable. This quantities are deduced from the chain rules (joint and relatives entropy decomposition).

Less common quantities can also be computed including **redundancy**, **information quality ratio** or **normalized mutual information**.

Two additional functions are optimised for computing mutual information and conditional mutual information between every pair of data points. This results in a symmetric matrix.

**Example usage**:

```julia
n = 1000
k = 3
p1 = rand(n)

println("Entropy invariant:")
println(entropy(p1, k=k, verbose=true))
println(entropy(1e5*p1.-123.465, k=k))
println(entropy(1e-5*p1.+654.321, k=k))

p2 = rand(n)
p3 = rand(n)           

println("\nJoint entropy invariant:")
println(EntropyInfo.entropy(hcat(p1,p2,p3), verbose=true))
println(entropy(hcat(p1, 1e5*p2.-123.456, 1e-5*p3.+654.123)))

p1 = rand(n)
p2 = 2*p1+rand(n)

println("\nMutual Information invariant: ")
println(mutual_information(p1, p2, k=k))
println(mutual_information(1e5*p1.+123.456, 1e-5*p2.+654.321, k=k))

```

In extreme cases, when the neighbourhood distance is small. The logarithm of the distance is strongly negative. This can lead to negative entropy. We therefore recommend setting the "degenerate" parameter to true. This parameter adds 1 to each distance, so that the logarithm is always positive.

Please inform us if you discover any bugs or errors in the code, or if you believe another quantity should be added.

[^1]: [Estimating mutual information](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138) DOI: 10.1103/PhysRevE.69.066138
[^2]: [Prior Probabilities](https://ieeexplore.ieee.org/document/4082152) DOI: 10.1109/TSSC.1968.300117
[^3]: [Python git](https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429) G Varoquaux
[^4]: An invariant estimation of entropy and mutual information

**AUTHORS:** Félix TRUONG, Alexandre GIULIANI

**References:**