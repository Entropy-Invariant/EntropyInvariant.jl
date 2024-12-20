using Statistics
using NearestNeighbors
using SpecialFunctions
using Plots
using Plots.PlotMeasures
using EntropyInvariant
using Distributions
using Random
using Serialization

seed = 42
Random.seed!(seed)

# Figures histogrammes

N = 100:500:10100  # size samples
BINS = 10:1:50 # list different bins
NB = 100; # number of simulation

hist_nor = zeros(NB, length(BINS), length(N))
hist_uni = zeros(NB, length(BINS), length(N))
hist_exp = zeros(NB, length(BINS), length(N))

for nb in 1:NB
    println(nb)
    for bins in 1:length(BINS)
        for n in 1:length(N)
            nor_ = rand(Normal(0,1), N[n])
            hist_nor[nb, bins, n] = EntropyInvariant.entropy(nor_, nbins=BINS[bins], method="histogram")
            
            uni_ = rand(Uniform(0,1), N[n])
            hist_uni[nb, bins, n] = EntropyInvariant.entropy(uni_, nbins=BINS[bins], method="histogram")
            
            exp_ = rand(Exponential(1), N[n])
            hist_exp[nb, bins, n] = EntropyInvariant.entropy(exp_, nbins=BINS[bins], method="histogram")
        end
    end
end

hist_nor2 = zeros(NB, length(BINS), length(N))
hist_uni2 = zeros(NB, length(BINS), length(N))
hist_exp2 = zeros(NB, length(BINS), length(N))

for nb in 1:NB
    println(nb)
    for bins in 1:length(BINS)
        for n in 1:length(N)
            nor_ = rand(Normal(0,1/2), N[n])
            hist_nor2[nb, bins, n] = EntropyInvariant.entropy(nor_, nbins=BINS[bins], method="histogram")
            
            uni_ = rand(Uniform(0,1/2), N[n])
            hist_uni2[nb, bins, n] = EntropyInvariant.entropy(uni_, nbins=BINS[bins], method="histogram")
            
            exp_ = rand(Exponential(2), N[n])
            hist_exp2[nb, bins, n] = EntropyInvariant.entropy(exp_, nbins=BINS[bins], method="histogram")
        end
    end
end

hist_nor3 = zeros(NB, length(BINS), length(N))
hist_uni3 = zeros(NB, length(BINS), length(N))
hist_exp3 = zeros(NB, length(BINS), length(N))

for nb in 1:NB
    println(nb)
    for bins in 1:length(BINS)
        for n in 1:length(N)
            nor_ = rand(Normal(0,3), N[n])
            hist_nor3[nb, bins, n] = EntropyInvariant.entropy(nor_, nbins=BINS[bins], method="histogram")
            
            uni_ = rand(Uniform(0,3), N[n])
            hist_uni3[nb, bins, n] = EntropyInvariant.entropy(uni_, nbins=BINS[bins], method="histogram")
            
            exp_ = rand(Exponential(1/3), N[n])
            hist_exp3[nb, bins, n] = EntropyInvariant.entropy(exp_, nbins=BINS[bins], method="histogram")
        end
    end
end

me_hist_nor = mean(hist_nor, dims=1)[1,:,:]
me_hist_uni = mean(hist_uni, dims=1)[1,:,:]
me_hist_exp = mean(hist_exp, dims=1)[1,:,:]

std_hist_nor = std(hist_nor, dims=1)[1,:,:]
std_hist_uni = std(hist_uni, dims=1)[1,:,:]
std_hist_exp = std(hist_exp, dims=1)[1,:,:]



me_hist_nor2 = mean(hist_nor2, dims=1)[1,:,:]
me_hist_uni2 = mean(hist_uni2, dims=1)[1,:,:]
me_hist_exp2 = mean(hist_exp2, dims=1)[1,:,:]

std_hist_nor2 = std(hist_nor2, dims=1)[1,:,:]
std_hist_uni2 = std(hist_uni2, dims=1)[1,:,:]
std_hist_exp2 = std(hist_exp2, dims=1)[1,:,:]



me_hist_nor3 = mean(hist_nor3, dims=1)[1,:,:]
me_hist_uni3 = mean(hist_uni3, dims=1)[1,:,:]
me_hist_exp3 = mean(hist_exp3, dims=1)[1,:,:]

std_hist_nor3 = std(hist_nor3, dims=1)[1,:,:]
std_hist_uni3 = std(hist_uni3, dims=1)[1,:,:]
std_hist_exp3 = std(hist_exp3, dims=1)[1,:,:];

# 1. Normal(0,1) avec N
p1 = plot(N, vec(mean(me_hist_nor, dims=1)), ribbon=vec(std(std_hist_nor, dims=1)), label="Normal(0, 1)", color="blue", fillalpha = 0.15, lw = 1)
plot!(N, vec(mean(me_hist_nor2, dims=1)), ribbon=vec(std(std_hist_nor2, dims=1)), label="Normal(0, 0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_hist_nor3, dims=1)), ribbon=vec(std(std_hist_nor3, dims=1)), label="Normal(0, 3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of points", ylabel="Entropy (nats)")
annotate!(p1, -2500, 4, text("a)", :left, 12))

# 2. Uniform(0,1) avec N
p2 = plot(N, vec(mean(me_hist_uni, dims=1)), ribbon=vec(std(std_hist_uni, dims=1)), label="Uniform(0, 1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_hist_uni2, dims=1)), ribbon=vec(std(std_hist_uni2, dims=1)), label="Uniform(0, 0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_hist_uni3, dims=1)), ribbon=vec(std(std_hist_uni3, dims=1)), label="Uniform(0, 3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of points", ylabel="Entropy (nats)")
annotate!(p2, -2500, 4, text("c)", :left, 12))

# 3. Exponential(0,1) avec N
p3 = plot(N, vec(mean(me_hist_exp, dims=1)), ribbon=vec(std(std_hist_exp, dims=1)), label="Exponential(1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_hist_exp2, dims=1)), ribbon=vec(std(std_hist_exp2, dims=1)), label="Exponential(0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_hist_exp3, dims=1)), ribbon=vec(std(std_hist_exp3, dims=1)), label="Exponential(3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of points", ylabel="Entropy (nats)")
annotate!(p3, -2500, 4, text("e)", :left, 12))

# 4. Normal(0,1) avec Hist
p4 = plot(BINS, vec(mean(me_hist_nor, dims=2)), ribbon=vec(std(std_hist_nor, dims=2)), label="Normal(0, 1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(BINS, vec(mean(me_hist_nor2, dims=2)), ribbon=vec(std(std_hist_nor2, dims=2)), label="Normal(0, 0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(BINS, vec(mean(me_hist_nor3, dims=2)), ribbon=vec(std(std_hist_nor3, dims=2)), label="Normal(0, 3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of bins", ylabel="Entropy (nats)")
annotate!(p4, 1, 4, text("b)", :left, 12))

# 5. Uniform(0,1) avec Hist
p5 = plot(BINS, vec(mean(me_hist_uni, dims=2)), ribbon=vec(std(std_hist_uni, dims=2)), label="Uniform(0, 1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(BINS, vec(mean(me_hist_uni2, dims=2)), ribbon=vec(std(std_hist_uni2, dims=2)), label="Uniform(0, 0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(BINS, vec(mean(me_hist_uni3, dims=2)), ribbon=vec(std(std_hist_uni3, dims=2)), label="Uniform(0, 3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of bins", ylabel="Entropy (nats)")
annotate!(p5, 1, 4, text("d)", :left, 12))

# 6. Exponential(0,1) avec Hist
p6 = plot(BINS, vec(mean(me_hist_exp, dims=2)), ribbon=vec(std(std_hist_exp, dims=2)), label="Exponential(1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(BINS, vec(mean(me_hist_exp2, dims=2)), ribbon=vec(std(std_hist_exp2, dims=2)), label="Exponential(0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(BINS, vec(mean(me_hist_exp3, dims=2)), ribbon=vec(std(std_hist_exp3, dims=2)), label="Exponential(3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of bins", ylabel="Entropy (nats)")
annotate!(p6, 1, 4, text("f)", :left, 12))

# Affichage des six graphiques dans une seule figure
plot(p1, p4, p2, p5, p3, p6, size=(600, 800), layout=(3, 2))
plot!(left_margin=0.2cm, ylim=(1, 4))
#savefig("hist_estimation.png")

# Figures KNN

N = 100:500:10100  # size samples
KNN = 3:3:30 # list different k
NB = 100; # number of simulation

knn_nor = zeros(NB, length(KNN), length(N))
knn_uni = zeros(NB, length(KNN), length(N))
knn_exp = zeros(NB, length(KNN), length(N))

for nb in 1:NB
    for knn in 1:length(KNN)
        for n in 1:length(N)
            nor_ = rand(Normal(0,1), N[n])
            knn_nor[nb, knn, n] = EntropyInvariant.entropy(nor_, k=KNN[knn], method="knn")
            
            uni_ = rand(Uniform(0,1), N[n])
            knn_uni[nb, knn, n] = EntropyInvariant.entropy(uni_, k=KNN[knn], method="knn")
            
            exp_ = rand(Exponential(1), N[n])
            knn_exp[nb, knn, n] = EntropyInvariant.entropy(exp_, k=KNN[knn], method="knn")
        end
    end
end

knn_nor2 = zeros(NB, length(KNN), length(N))
knn_uni2 = zeros(NB, length(KNN), length(N))
knn_exp2 = zeros(NB, length(KNN), length(N))

for nb in 1:NB
    for knn in 1:length(KNN)
        for n in 1:length(N)
            nor_ = rand(Normal(0,0.5), N[n])
            knn_nor2[nb, knn, n] = EntropyInvariant.entropy(nor_, k=KNN[knn], method="knn")
            
            uni_ = rand(Uniform(0,0.5), N[n])
            knn_uni2[nb, knn, n] = EntropyInvariant.entropy(uni_, k=KNN[knn], method="knn")
            
            exp_ = rand(Exponential(2), N[n])
            knn_exp2[nb, knn, n] = EntropyInvariant.entropy(exp_, k=KNN[knn], method="knn")
        end
    end
end

knn_nor3 = zeros(NB, length(KNN), length(N))
knn_uni3 = zeros(NB, length(KNN), length(N))
knn_exp3 = zeros(NB, length(KNN), length(N))

for nb in 1:NB
    for knn in 1:length(KNN)
        for n in 1:length(N)
            nor_ = rand(Normal(0,3), N[n])
            knn_nor3[nb, knn, n] = EntropyInvariant.entropy(nor_, k=KNN[knn], method="knn")
            
            uni_ = rand(Uniform(0,3), N[n])
            knn_uni3[nb, knn, n] = EntropyInvariant.entropy(uni_, k=KNN[knn], method="knn")
            
            exp_ = rand(Exponential(1/3), N[n])
            knn_exp3[nb, knn, n] = EntropyInvariant.entropy(exp_, k=KNN[knn], method="knn")
        end
    end
end

me_knn_nor = mean(knn_nor, dims=1)[1,:,:]
me_knn_uni = mean(knn_uni, dims=1)[1,:,:]
me_knn_exp = mean(knn_exp, dims=1)[1,:,:]

std_knn_nor = std(knn_nor, dims=1)[1,:,:]
std_knn_uni = std(knn_uni, dims=1)[1,:,:]
std_knn_exp = std(knn_exp, dims=1)[1,:,:]



me_knn_nor2 = mean(knn_nor2, dims=1)[1,:,:]
me_knn_uni2 = mean(knn_uni2, dims=1)[1,:,:]
me_knn_exp2 = mean(knn_exp2, dims=1)[1,:,:]

std_knn_nor2 = std(knn_nor2, dims=1)[1,:,:]
std_knn_uni2 = std(knn_uni2, dims=1)[1,:,:]
std_knn_exp2 = std(knn_exp2, dims=1)[1,:,:]



me_knn_nor3 = mean(knn_nor3, dims=1)[1,:,:]
me_knn_uni3 = mean(knn_uni3, dims=1)[1,:,:]
me_knn_exp3 = mean(knn_exp3, dims=1)[1,:,:]

std_knn_nor3 = std(knn_nor3, dims=1)[1,:,:]
std_knn_uni3 = std(knn_uni3, dims=1)[1,:,:]
std_knn_exp3 = std(knn_exp3, dims=1)[1,:,:];

# 1. Normal(0,1) avec N
p1 = plot(N, vec(mean(me_knn_nor, dims=1)), ribbon=vec(std(std_knn_nor, dims=1)), label="Normal(0, 1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_knn_nor2, dims=1)), ribbon=vec(std(std_knn_nor2, dims=1)), label="Normal(0, 0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_knn_nor3, dims=1)), ribbon=vec(std(std_knn_nor3, dims=1)), label="Normal(0, 3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of points", ylabel="Entropy (nats)")
hline!([log(1*sqrt(2*pi*exp(1)))], color="black", label="Theoreticals values", linestyle=:dash, lw=2)
hline!([log(0.5*sqrt(2*pi*exp(1)))], color="black", label="", linestyle=:dash, lw=2)
hline!([log(3*sqrt(2*pi*exp(1)))], color="black", label="", linestyle=:dash, lw=2)
annotate!(p1, -2500, 3, text("a)", :left, 12))

# 2. Uniform(0,1) avec N
p2 = plot(N, vec(mean(me_knn_uni, dims=1)), ribbon=vec(std(std_knn_uni, dims=1)), label="Uniform(0, 1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_knn_uni2, dims=1)), ribbon=vec(std(std_knn_uni2, dims=1)), label="Uniform(0, 0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_knn_uni3, dims=1)), ribbon=vec(std(std_knn_uni3, dims=1)), label="Uniform(0, 3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of points", ylabel="Entropy (nats)")
hline!([log(1-0)], color="black", label="Theoreticals values", linestyle=:dash, lw=2)
hline!([log(0.5-0)], color="black", label="", linestyle=:dash, lw=2)
hline!([log(3-0)], color="black", label="", linestyle=:dash, lw=2)
annotate!(p2, -2500, 3, text("c)", :left, 12))

# 3. Exponential(0,1) avec N
p3 = plot(N, vec(mean(me_knn_exp, dims=1)), ribbon=vec(std(std_knn_exp, dims=1)), label="Exponential(1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_knn_exp2, dims=1)), ribbon=vec(std(std_knn_exp2, dims=1)), label="Exponential(0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_knn_exp3, dims=1)), ribbon=vec(std(std_knn_exp3, dims=1)), label="Exponential(3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of points", ylabel="Entropy (bits)")
hline!([1-log(1)], color="black", label="Theoreticals values", linestyle=:dash, lw=2)
hline!([1-log(0.5)], color="black", label="", linestyle=:dash, lw=2)
hline!([1-log(3)], color="black", label="", linestyle=:dash, lw=2)
annotate!(p3, -2500, 3, text("e)", :left, 12))

# 4. Normal(0,1) avec KNN
p4 = plot(KNN, vec(mean(me_knn_nor, dims=2)), ribbon=vec(std(std_knn_nor, dims=2)), label="Normal(0, 1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_knn_nor2, dims=2)), ribbon=vec(std(std_knn_nor2, dims=2)), label="Normal(0, 0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_knn_nor3, dims=2)), ribbon=vec(std(std_knn_nor3, dims=2)), label="Normal(0, 3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of neighbours (k)", ylabel="Entropy (nats)")
hline!([log(1*sqrt(2*pi*exp(1)))], color="black", label="Theoreticals values", linestyle=:dash, lw=2)
hline!([log(0.5*sqrt(2*pi*exp(1)))], color="black", label="", linestyle=:dash, lw=2)
hline!([log(3*sqrt(2*pi*exp(1)))], color="black", label="", linestyle=:dash, lw=2)
annotate!(p4, -5, 3, text("b)", :left, 12))

# 5. Uniform(0,1) avec KNN
p5 = plot(KNN, vec(mean(me_knn_uni, dims=2)), ribbon=vec(std(std_knn_uni, dims=2)), label="Uniform(0, 1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_knn_uni2, dims=2)), ribbon=vec(std(std_knn_uni2, dims=2)), label="Uniform(0, 0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_knn_uni3, dims=2)), ribbon=vec(std(std_knn_uni3, dims=2)), label="Uniform(0, 3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of neighbours (k)", ylabel="Entropy (nats)")
hline!([log(1-0)], color="black", label="Theoreticals values", linestyle=:dash, lw=2)
hline!([log(0.5-0)], color="black", label="", linestyle=:dash, lw=2)
hline!([log(3-0)], color="black", label="", linestyle=:dash, lw=2)
annotate!(p5, -5, 3, text("d)", :left, 12))

# 6. Exponential(0,1) avec KNN
p6 = plot(KNN, vec(mean(me_knn_exp, dims=2)), ribbon=vec(std(std_knn_exp, dims=2)), label="Exponential(1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_knn_exp2, dims=2)), ribbon=vec(std(std_knn_exp2, dims=2)), label="Exponential(0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_knn_exp3, dims=2)), ribbon=vec(std(std_knn_exp3, dims=2)), label="Exponential(3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of neighbours (k)", ylabel="Entropy (nats)")
hline!([1-log(1)], color="black", label="Theoreticals values", linestyle=:dash, lw=2)
hline!([1-log(0.5)], color="black", label="", linestyle=:dash, lw=2)
hline!([1-log(3)], color="black", label="", linestyle=:dash, lw=2)
annotate!(p6, -5, 3, text("f)", :left, 12))

# Affichage des six graphiques dans une seule figure
plot(p1, p4, p2, p5, p3, p6, size=(600, 800), layout=(3, 2))
plot!(left_margin=0.2cm, ylim=(-1, 3))
#savefig("knn_estimation.png")

# KNN invariant

N = 100:500:10100  # size samples
KNN = 3:3:30 # list different k
NB = 100; # number of simulation

inv_knn_nor = zeros(NB, length(KNN), length(N))
inv_knn_uni = zeros(NB, length(KNN), length(N))
inv_knn_exp = zeros(NB, length(KNN), length(N))

for nb in 1:NB
    for knn in 1:length(KNN)
        for n in 1:length(N)
            nor_ = rand(Normal(0,1), N[n])
            inv_knn_nor[nb, knn, n] = EntropyInvariant.entropy(nor_, k=KNN[knn], method="inv")
            
            uni_ = rand(Uniform(0,1), N[n])
            inv_knn_uni[nb, knn, n] = EntropyInvariant.entropy(uni_, k=KNN[knn], method="inv")
            
            exp_ = rand(Exponential(1), N[n])
            inv_knn_exp[nb, knn, n] = EntropyInvariant.entropy(exp_, k=KNN[knn], method="inv")
        end
    end
end

inv_knn_nor2 = zeros(NB, length(KNN), length(N))
inv_knn_uni2 = zeros(NB, length(KNN), length(N))
inv_knn_exp2 = zeros(NB, length(KNN), length(N))

for nb in 1:NB
    for knn in 1:length(KNN)
        for n in 1:length(N)
            nor_ = rand(Normal(0,0.5), N[n])
            inv_knn_nor2[nb, knn, n] = EntropyInvariant.entropy(nor_, k=KNN[knn], method="inv")
            
            uni_ = rand(Uniform(0,0.5), N[n])
            inv_knn_uni2[nb, knn, n] = EntropyInvariant.entropy(uni_, k=KNN[knn], method="inv")
            
            exp_ = rand(Exponential(2), N[n])
            inv_knn_exp2[nb, knn, n] = EntropyInvariant.entropy(exp_, k=KNN[knn], method="inv")
        end
    end
end

inv_knn_nor3 = zeros(NB, length(KNN), length(N))
inv_knn_uni3 = zeros(NB, length(KNN), length(N))
inv_knn_exp3 = zeros(NB, length(KNN), length(N))

for nb in 1:NB
    for knn in 1:length(KNN)
        for n in 1:length(N)
            nor_ = rand(Normal(0,3), N[n])
            inv_knn_nor3[nb, knn, n] = EntropyInvariant.entropy(nor_, k=KNN[knn], method="inv")
            
            uni_ = rand(Uniform(0,3), N[n])
            inv_knn_uni3[nb, knn, n] = EntropyInvariant.entropy(uni_, k=KNN[knn], method="inv")
            
            exp_ = rand(Exponential(1/3), N[n])
            inv_knn_exp3[nb, knn, n] = EntropyInvariant.entropy(exp_, k=KNN[knn], method="inv")
        end
    end
end

me_inv_knn_nor = mean(inv_knn_nor, dims=1)[1,:,:]
me_inv_knn_uni = mean(inv_knn_uni, dims=1)[1,:,:]
me_inv_knn_exp = mean(inv_knn_exp, dims=1)[1,:,:]

std_inv_knn_nor = std(inv_knn_nor, dims=1)[1,:,:]
std_inv_knn_uni = std(inv_knn_uni, dims=1)[1,:,:]
std_inv_knn_exp = std(inv_knn_exp, dims=1)[1,:,:]


me_inv_knn_nor2 = mean(inv_knn_nor2, dims=1)[1,:,:]
me_inv_knn_uni2 = mean(inv_knn_uni2, dims=1)[1,:,:]
me_inv_knn_exp2 = mean(inv_knn_exp2, dims=1)[1,:,:]

std_inv_knn_nor2 = std(inv_knn_nor2, dims=1)[1,:,:]
std_inv_knn_uni2 = std(inv_knn_uni2, dims=1)[1,:,:]
std_inv_knn_exp2 = std(inv_knn_exp2, dims=1)[1,:,:]


me_inv_knn_nor3 = mean(inv_knn_nor3, dims=1)[1,:,:]
me_inv_knn_uni3 = mean(inv_knn_uni3, dims=1)[1,:,:]
me_inv_knn_exp3 = mean(inv_knn_exp3, dims=1)[1,:,:]

std_inv_knn_nor3 = std(inv_knn_nor3, dims=1)[1,:,:]
std_inv_knn_uni3 = std(inv_knn_uni3, dims=1)[1,:,:]
std_inv_knn_exp3 = std(inv_knn_exp3, dims=1)[1,:,:];

# 1. Normal(0,1) avec N
p1 = plot(N, vec(mean(me_inv_knn_nor, dims=1)), ribbon=vec(std(std_inv_knn_nor, dims=1)), label="Normal(0, 1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_inv_knn_nor2, dims=1)), ribbon=vec(std(std_inv_knn_nor2, dims=1)), label="Normal(0, 0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_inv_knn_nor3, dims=1)), ribbon=vec(std(std_inv_knn_nor3, dims=1)), label="Normal(0, 3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of points", ylabel="Entropy (nats)")
#hline!([log(1*sqrt(2*pi*exp(1)))], color="black", label="Theoretical value")
annotate!(p1, -2850, 1.3, text("a)", :left, 12))

# 2. Uniform(0,1) avec N
p2 = plot(N, vec(mean(me_inv_knn_uni, dims=1)), ribbon=vec(std(std_inv_knn_uni, dims=1)), label="Uniform(0, 1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_inv_knn_uni2, dims=1)), ribbon=vec(std(std_inv_knn_uni2, dims=1)), label="Uniform(0, 0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_inv_knn_uni3, dims=1)), ribbon=vec(std(std_inv_knn_uni3, dims=1)), label="Uniform(0, 3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of points", ylabel="Entropy (nats)")
#hline!([0], color="black", label="Theoretical value")
annotate!(p2, -2850, 1.3, text("c)", :left, 12))

# 3. Exponential(0,1) avec N
p3 = plot(N, vec(mean(me_inv_knn_exp, dims=1)), ribbon=vec(std(std_inv_knn_exp, dims=1)), label="Exponential(1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_inv_knn_exp2, dims=1)), ribbon=vec(std(std_inv_knn_exp2, dims=1)), label="Exponential(0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_inv_knn_exp3, dims=1)), ribbon=vec(std(std_inv_knn_exp3, dims=1)), label="Exponential(3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of points", ylabel="Entropy (nats)")
#hline!([1-log(1)], color="black", label="Theoretical value")
annotate!(p3, -2850, 1.3, text("e)", :left, 12))

# 4. Normal(0,1) avec KNN
p4 = plot(KNN, vec(mean(me_inv_knn_nor, dims=2)), ribbon=vec(std(std_inv_knn_nor, dims=2)), label="Normal(0, 1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_inv_knn_nor2, dims=2)), ribbon=vec(std(std_inv_knn_nor2, dims=2)), label="Normal(0, 0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_inv_knn_nor3, dims=2)), ribbon=vec(std(std_inv_knn_nor3, dims=2)), label="Normal(0, 3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of neighbours (k)", ylabel="Entropy (nats)")
#hline!([log(1*sqrt(2*pi*exp(1)))], color="black", label="Theoretical value")
annotate!(p4, -5, 1.3, text("b)", :left, 12))

# 5. Uniform(0,1) avec KNN
p5 = plot(KNN, vec(mean(me_inv_knn_uni, dims=2)), ribbon=vec(std(std_inv_knn_uni, dims=2)), label="Uniform(0, 1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_inv_knn_uni2, dims=2)), ribbon=vec(std(std_inv_knn_uni2, dims=2)), label="Uniform(0, 0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_inv_knn_uni3, dims=2)), ribbon=vec(std(std_inv_knn_uni3, dims=2)), label="Uniform(0, 3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of neighbours (k)", ylabel="Entropy (nats)")
#hline!([0], color="black", label="Theoretical value")
annotate!(p5, -5, 1.3, text("d)", :left, 12))

# 6. Exponential(0,1) avec KNN
p6 = plot(KNN, vec(mean(me_inv_knn_exp, dims=2)), ribbon=vec(std(std_inv_knn_exp, dims=2)), label="Exponential(1)", color="blue", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_inv_knn_exp2, dims=2)), ribbon=vec(std(std_inv_knn_exp2, dims=2)), label="Exponential(0.5)", color="red", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_inv_knn_exp3, dims=2)), ribbon=vec(std(std_inv_knn_exp3, dims=2)), label="Exponential(3)", color="green", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of neighbours (k)", ylabel="Entropy (nats)")
annotate!(p6, -5, 1.3, text("f)", :left, 12))

# Affichage des six graphiques dans une seule figure
plot(p1, p4, p2, p5, p3, p6, size=(600, 800), layout=(3, 2))
plot!(left_margin=0.2cm, ylim=(1, 1.3))
#savefig("inv_knn_estimation.png")

# Comparaison Mutual Information

mi_hist2_nor1 = zeros(NB, length(BINS), length(N))
mi_hist2_nor2 = zeros(NB, length(BINS), length(N))
mi_hist2_nor3 = zeros(NB, length(BINS), length(N))
for nb in 1:NB
    for bins in 1:length(BINS)
        for n in 1:length(N)
            nor1_ = rand(Normal(0, 0.1), N[n])
            nor2_ = rand(Normal(0, 1), N[n])
            nor3_ = rand(Normal(0, 10), N[n])
            
            mi_hist2_nor1[nb, bins, n] = EntropyInvariant.mutual_information(nor1_, nor2_, nbins=BINS[bins], method="histogram")
            mi_hist2_nor2[nb, bins, n] = EntropyInvariant.mutual_information(nor1_, nor3_, nbins=BINS[bins], method="histogram")
            mi_hist2_nor3[nb, bins, n] = EntropyInvariant.mutual_information(nor2_, nor3_, nbins=BINS[bins], method="histogram")
        end
    end
end

mi_knn2_nor1 = zeros(NB, length(KNN), length(N))
mi_knn2_nor2 = zeros(NB, length(KNN), length(N))
mi_knn2_nor3 = zeros(NB, length(KNN), length(N))
for nb in 1:NB
    for knn in 1:length(KNN)
        for n in 1:length(N)
            nor1_ = rand(Normal(0, 0.1), N[n])
            nor2_ = rand(Normal(0, 1), N[n])
            nor3_ = rand(Normal(0, 10), N[n])
            
            mi_knn2_nor1[nb, knn, n] = EntropyInvariant.mutual_information(nor1_, nor2_, k=KNN[knn], method="knn")
            mi_knn2_nor2[nb, knn, n] = EntropyInvariant.mutual_information(nor1_, nor3_, k=KNN[knn], method="knn")
            mi_knn2_nor3[nb, knn, n] = EntropyInvariant.mutual_information(nor2_, nor3_, k=KNN[knn], method="knn")
        end
    end
end

mi_inv_knn2_nor1 = zeros(NB, length(KNN), length(N))
mi_inv_knn2_nor2 = zeros(NB, length(KNN), length(N))
mi_inv_knn2_nor3 = zeros(NB, length(KNN), length(N))
for nb in 1:NB
    for knn in 1:length(KNN)
        for n in 1:length(N)
            nor1_ = rand(Normal(0, 0.1), N[n])
            nor2_ = rand(Normal(0, 1), N[n])
            nor3_ = rand(Normal(0, 10), N[n])
            
            mi_inv_knn2_nor1[nb, knn, n] = EntropyInvariant.mutual_information(nor1_, nor2_, k=KNN[knn], method="inv")
            mi_inv_knn2_nor2[nb, knn, n] = EntropyInvariant.mutual_information(nor1_, nor3_, k=KNN[knn], method="inv")
            mi_inv_knn2_nor3[nb, knn, n] = EntropyInvariant.mutual_information(nor2_, nor3_, k=KNN[knn], method="inv")
        end
    end
end

me_hist2_mi_nor1 = mean(mi_hist2_nor1, dims=1)[1,:,:]
me_hist2_mi_nor2 = mean(mi_hist2_nor2, dims=1)[1,:,:]
me_hist2_mi_nor3 = mean(mi_hist2_nor3, dims=1)[1,:,:]

me_knn2_mi_nor1 = mean(mi_knn2_nor1, dims=1)[1,:,:]
me_knn2_mi_nor2 = mean(mi_knn2_nor2, dims=1)[1,:,:]
me_knn2_mi_nor3 = mean(mi_knn2_nor3, dims=1)[1,:,:]

me_inv_knn2_mi_nor1 = mean(mi_inv_knn2_nor1, dims=1)[1,:,:]
me_inv_knn2_mi_nor2 = mean(mi_inv_knn2_nor2, dims=1)[1,:,:]
me_inv_knn2_mi_nor3 = mean(mi_inv_knn2_nor3, dims=1)[1,:,:];

p1 = plot(N, vec(mean(me_hist2_mi_nor1, dims=1)), ribbon=vec(std(me_hist2_mi_nor1, dims=1)), label="I(X;Y)", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_hist2_mi_nor2, dims=1)), ribbon=vec(std(me_hist2_mi_nor2, dims=1)), label="I(X;Z)", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_hist2_mi_nor3, dims=1)), ribbon=vec(std(me_hist2_mi_nor3, dims=1)), label="I(Y;Z)", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of points", ylabel="Entropy (nats)")
hline!([0], color="black", label="Theoretical value", linestyle=:dash, lw=2)
plot!(legend = :bottomleft)
annotate!(p1, -3350, 0.4, text("a)", :left, 12))

p2 = plot(N, vec(mean(me_knn2_mi_nor1, dims=1)), ribbon=vec(std(me_knn2_mi_nor1, dims=1)), label="I(X;Y)", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_knn2_mi_nor2, dims=1)), ribbon=vec(std(me_knn2_mi_nor2, dims=1)), label="I(X;Z)", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_knn2_mi_nor3, dims=1)), ribbon=vec(std(me_knn2_mi_nor3, dims=1)), label="I(Y;Z)", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of points", ylabel="Entropy (nats)")
hline!([0], color="black", label="Theoretical value", linestyle=:dash, lw=2)
plot!(legend = :topleft)
annotate!(p2, -3350, 0.4, text("c)", :left, 12))

p3 = plot(N, vec(mean(me_inv_knn2_mi_nor1, dims=1)), ribbon=vec(std(me_inv_knn2_mi_nor1, dims=1)), label="I(X;Y)", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_inv_knn2_mi_nor2, dims=1)), ribbon=vec(std(me_inv_knn2_mi_nor2, dims=1)), label="I(X;Z)", fillalpha = 0.15, lw = 4)
plot!(N, vec(mean(me_inv_knn2_mi_nor3, dims=1)), ribbon=vec(std(me_inv_knn2_mi_nor3, dims=1)), label="I(Y;Z)", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of points", ylabel="Entropy (nats)")
hline!([0], color="black", label="Theoretical value", linestyle=:dash, lw=2)
plot!(legend = :topleft)
annotate!(p3, -3350, 0.4, text("e)", :left, 12))

p4 = plot(BINS, vec(mean(me_hist2_mi_nor1, dims=2)), ribbon=vec(std(me_hist2_mi_nor1, dims=2)), label="I(X;Y)", fillalpha = 0.15, lw = 4)
plot!(BINS, vec(mean(me_hist2_mi_nor2, dims=2)), ribbon=vec(std(me_hist2_mi_nor2, dims=2)), label="I(X;Z)", fillalpha = 0.15, lw = 4)
plot!(BINS, vec(mean(me_hist2_mi_nor3, dims=2)), ribbon=vec(std(me_hist2_mi_nor3, dims=2)), label="I(Y;Z)", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of bins", ylabel="Entropy (nats)")
hline!([0], color="black", label="Theoretical value", linestyle=:dash, lw=2)
plot!(legend = :bottomleft)
annotate!(p4, -4, 0.4, text("b)", :left, 12))

p5 = plot(KNN, vec(mean(me_knn2_mi_nor1, dims=2)), ribbon=vec(std(me_knn2_mi_nor1, dims=2)), label="I(X;Y)", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_knn2_mi_nor2, dims=2)), ribbon=vec(std(me_knn2_mi_nor2, dims=2)), label="I(X;Z)", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_knn2_mi_nor3, dims=2)), ribbon=vec(std(me_knn2_mi_nor3, dims=2)), label="I(Y;Z)", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of neighbours (k)", ylabel="Entropy (nats)")
hline!([0], color="black", label="Theoretical value", linestyle=:dash, lw=2)
plot!(legend = :topleft)
annotate!(p5, -6, 0.4, text("d)", :left, 12))

p6 = plot(KNN, vec(mean(me_inv_knn2_mi_nor1, dims=2)), ribbon=vec(std(me_inv_knn2_mi_nor1, dims=2)), label="I(X;Y)", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_inv_knn2_mi_nor2, dims=2)), ribbon=vec(std(me_inv_knn2_mi_nor2, dims=2)), label="I(X;Z)", fillalpha = 0.15, lw = 4)
plot!(KNN, vec(mean(me_inv_knn2_mi_nor3, dims=2)), ribbon=vec(std(me_inv_knn2_mi_nor3, dims=2)), label="I(Y;Z)", fillalpha = 0.15, lw = 4)
plot!(xlabel="Number of neighbours (k)", ylabel="Entropy (nats)")
hline!([0], color="black", label="Theoretical value", linestyle=:dash, lw=2)
plot!(legend = :topleft)
annotate!(p6, -6, 0.4, text("f)", :left, 12))

plot(p1, p4, p2, p5, p3, p6, size=(600, 800), layout=(3, 2))
plot!(left_margin=0.2cm, ylim=(-0.4, 0.4))
#savefig("mi_comparaison.png")

# Mutual information with noise

seed = 42
Random.seed!(seed)

x = pi*(rand(1000).-0.5)
x2 = pi*(rand(1000).-0.5)

y1 = cos.(x).+rand(1000)/100
y2 = y1.+rand(1000)/10
y3 = y1.+rand(1000)/1
y4 = y1.+rand(1000)*10

x = 100*x


ent1_inv = EntropyInvariant.mutual_information(x, y1, method="inv")
ent1_knn = EntropyInvariant.mutual_information(x, y1, method="knn")

cor1 = Statistics.cor(x, y1)
cov1 = Statistics.cov(x, y1)

println(ent1_inv)
println(ent1_knn)
println(cor1)
println(cov1)
println()

ent2_inv = EntropyInvariant.mutual_information(x, y2, method="inv")
ent2_knn = EntropyInvariant.mutual_information(x, y2, method="knn")

cor2 = Statistics.cor(x, y2)
cov2 = Statistics.cov(x, y2)

println(ent2_inv)
println(ent2_knn)
println(cor2)
println(cov2)
println()

ent3_inv = EntropyInvariant.mutual_information(x, y3, method="inv")
ent3_knn = EntropyInvariant.mutual_information(x, y3, method="knn")

cor3 = Statistics.cor(x, y3)
cov3 = Statistics.cov(x, y3)

println(ent3_inv)
println(ent3_knn)
println(cor3)
println(cov3)
println()


ent4_inv = EntropyInvariant.mutual_information(x, y4, method="inv")
ent4_knn = EntropyInvariant.mutual_information(x, y4, method="knn")

cor4 = Statistics.cor(x, y4)
cov4 = Statistics.cov(x, y4)

println(ent4_inv)
println(ent4_knn)
println(cor4)
println(cov4)
println()

p1 = scatter(x, y1, label="", color="black")
plot!(title="Cor=$(round(cor1, digits=2)); MI(kNN)=$(round(ent1_knn, digits=2)); MI(Inv)=$(round(ent1_inv, digits=2))")
annotate!(p1, -190, 1.25, text("a)", :left, 12))
plot!(xlabel="x", ylabel="y")

p2 = scatter(x, y2, label="", color="black")
plot!(title="Cor=$(round(cor2, digits=2)); MI(kNN)=$(round(ent2_knn, digits=2)); MI(Inv)=$(round(ent2_inv, digits=2))")
annotate!(p2, -190, 1.25, text("b)", :left, 12))
plot!(xlabel="x", ylabel="y")

p3 = scatter(x, y3, label="", color="black")
plot!(title="Cor=$(round(cor3, digits=2)); MI(kNN)=$(round(ent3_knn, digits=2)); MI(Inv)=$(round(ent3_inv, digits=2))")
annotate!(p3, -188, 2.45, text("c)", :left, 12))
plot!(xlabel="x", ylabel="y")

p4 = scatter(x, y4, label="", color="black")
plot!(title="Cor=$(round(cor4, digits=2)); MI(kNN)=$(round(ent4_knn, digits=2)); MI(Inv)=$(round(ent4_inv, digits=2))")
annotate!(p4, -188, 13, text("d)", :left, 12))
plot!(xlabel="x", ylabel="y")

plot(p1,p2,p3,p4, size=(500,800), layout=(4,1), left_margin=0.3cm)
#savefig("MI_noise.png")
