# Distributions

n, m = 100000, 100

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Arcsine(1,1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Arcsine(1,100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Arcsine(1, 10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Arcsine(1, 2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Arcsine")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Uniform(1,1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Uniform(1,100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Uniform(1, 10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Uniform(1, 2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Uniform")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Semicircle(1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Semicircle(100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Semicircle(10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Semicircle(2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Semicircle")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(SymTriangularDist(1,1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(SymTriangularDist(1,100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(SymTriangularDist(1, 10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(SymTriangularDist(1, 2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("SymTriangularDist")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Normal(1,1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Normal(1,100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Normal(1, 10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Normal(1, 2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Normal")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(NormalCanon(1,1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(NormalCanon(1,100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(NormalCanon(1, 10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(NormalCanon(1, 2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("NormalCanon")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Cosine(1,1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Cosine(1,100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Cosine(1, 10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Cosine(1, 2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Cosine")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Rayleigh(1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Rayleigh(100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Rayleigh(10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Rayleigh(2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Rayleigh")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Chi(1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Chi(100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Chi(10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Chi(2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Chi")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(NoncentralChisq(1,1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(NoncentralChisq(1,100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(NoncentralChisq(1, 10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(NoncentralChisq(1, 2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("NoncentralChisq")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Kolmogorov(), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Kolmogorov(), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Kolmogorov(), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Kolmogorov(), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Kolmogorov")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Gumbel(1,1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Gumbel(1,100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Gumbel(1, 10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Gumbel(1, 2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Gumbel")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Logistic(1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Logistic(100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Logistic(10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Logistic(2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Logistic")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Exponential(1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Exponential(100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Exponential(10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Exponential(2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Exponential")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Laplace(1, 1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Laplace(1, 100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Laplace(1, 10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Laplace(1, 2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Laplace")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Cauchy(1, 1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Cauchy(1, 100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Cauchy(1, 10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Cauchy(1, 2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Cauchy")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))

all_ent = zeros(4, m);
for i in 1:m
    X = rand(Laplace(1, 1000), n)
    all_ent[1,i] = EntropyInvariant.entropy(X)
    
    X = rand(Laplace(1, 100), n)
    all_ent[2,i] = EntropyInvariant.entropy(X)
    
    X = rand(Laplace(1, 10), n)
    all_ent[3,i] = EntropyInvariant.entropy(X)
    
    X = rand(Laplace(1, 2), n)
    all_ent[4,i] = EntropyInvariant.entropy(X)
end

println("Laplace")
println(mean(all_ent, dims=2))
println(std(all_ent, dims=2))
println(mean(all_ent))
println(std(all_ent))
