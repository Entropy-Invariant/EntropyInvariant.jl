# EntropyInvariant

This code is an improved nearest neighbor method for estimating differential entropy[^1] for continuous variables, invariant under change of variables, and positive. This approximation claim to solve the limiting density of discrete points formulated by Edwin Thompson Jaynes[^2]. All the details of the estimation can be found on the paper[^4].

$$
\begin{equation*}
\begin{split}
H(X)=-\int_X p(x)\log\left(\frac{p(x)}{m(x)}\right)\mathrm{d}x
\end{split}
\end{equation*}
$$

The main novelty is m(x) the invariante measure.
We introduce a proposition to describe this measure[^4] with the following properties:

$$
\begin{equation*}
\begin{aligned}
1\) & \quad m(X) = r_X > 0, \\
2\) & \quad m(aX) = am(X) = ar_X, \\
3\) & \quad m(X+b) = m(X) = r_X
\end{aligned}
\end{equation*}
$$

We found that the median value of the nearest-neighbor distance of each point is an appropriate measure for these properties.

The nearest-neighbor estimation was initialy adapted by G Varoquaux[^3] in python from this paper[^1].

In this package you can find other information theory quantities such as **joint entropy**, **conditional entropy**, **mutual information**, **conditional mutual information** and **interaction of information**. All based on a generalisation of limiting density of discrete points for more than one variable. This quantities are deduced from the chain rules (joint and relatives entropy decomposition).

Less common quantities can also be computed including **redundancy**, **information quality ratio** or **normalized mutual information**.

The methode for computing the entropy can also be specified. 
- `method::String = "inv"` (optional): The method to use for entropy computation. Options are:
  - `"knn"`: k-Nearest Neighbors (k-NN) based entropy estimation.
  - `"histogram"`: Histogram-based entropy estimation.
  - `"inv"`: Invariant entropy estimation (default).

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

# Using k-NN method
data = rand(1, 100)  # 100 points in 1 dimension
println("Entropy (k-NN): ", entropy(data, method="knn", k=5, verbose=true))

# Using histogram method
data = rand(100)  # 100 points in 1 dimension
println("Entropy (Histogram): ", entropy(data, method="histogram", nbins=10) )

# Using invariant method
data = rand(100)  # 100 points in 1 dimension
println("Entropy (Invariant): ", entropy(data, method="inv", k=3))

```

In extreme cases, when the neighbourhood distance is small. The logarithm of the distance is strongly negative. This can lead to negative entropy. We therefore recommend setting the "degenerate" parameter to true. This parameter adds 1 to each distance, so that the logarithm is always positive.

Please inform us if you discover any bugs or errors in the code, or if you believe another quantity should be added.

[^1]: [Estimating mutual information](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138) DOI: 10.1103/PhysRevE.69.066138
[^2]: [Prior Probabilities](https://ieeexplore.ieee.org/document/4082152) DOI: 10.1109/TSSC.1968.300117
[^3]: [Python git](https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429) G Varoquaux
[^4]: An invariant estimation of entropy and mutual information (not yet plublished)

**AUTHORS:** Félix TRUONG, Alexandre GIULIANI

**References:**
