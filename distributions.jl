using LinearAlgebra
using Plots



# ----- Y_i variables -----

# create Y_i vector
function create_y(
        n::Int64,
        d::Int64)
    y = Matrix{Float64}[]
    max = 0;
    for i in 1:n
        A = rand(d,d);
        push!(y,A'*A);
        if eigmax(y[i]) > max
            max = eigmax(y[i]);
        end
    end
    for i in 1:n
        y[i] = y[i]/max;
    end
    return y;
end

# create Y_i vector
function create_diagonal_y(
        n::Int64,
        d::Int64)
    y = Matrix{Float64}[]
    max = 0;
    for i in 1:n
        A = zeros(d,d);
        for j in 1:d
            A[j,j] = rand();
        end
        push!(y,A);
        if eigmax(y[i]) > max
            max = eigmax(y[i]);
        end
    end
    for i in 1:n
        y[i] = y[i]/max;
    end
    return y;
end



# ----- Probability distributions ------

# define mu
function lambda_uniform_mu(n,k)
    lambdas = rand(n);
    tot = 0;
    mu = zeros(2^n);
    for s in 1:2^n
        if count_ones(s) == k
            mu[s] = 1;
            for i in 1:n
                if (s÷(2^(i-1)))%2==1
                    mu[s] *= 1/lambdas[i]^2;
                end
            end
            tot += mu[s];
        end
    end
    # normalize
    for s in 1:2^n
        if count_ones(s) == k
            mu[s] = mu[s]/tot;
        end
    end
    return mu;
end

function get_marginals_mu(mu)
    mu_1 = zeros(n);
    for j in 1:n
        for i in 1:2^n
           if (count_ones(i)==k) && ((i÷(2^(j-1)))%2==1)
                mu_1[j] += mu[i]/k;
            end
        end
    end
    return mu_1;    
end

function get_mu_k_1(mu)
    mu_k_1 = zeros(2^n);
    for j in 1:n
        for i in 1:2^n
           if (count_ones(i)==k) && ((i÷(2^(j-1)))%2==1)
                mu_k_1[i-2^(j-1)] += mu[i]/k;
            end
        end
    end
    return mu_k_1;
end

function one_ising_model(
        n::Int64,
        k::Int64)
    if n != 2*k
        println("Check n and k.");
    end
    n_2 = n÷2;

    # parameters u and h
    u = zeros(n_2);
    h = rand(n_2)/n;
    tot = 0;
    for i in 1:n_2
        u[i] = 1/rand()^5-1;
        tot += u[i]^2;        
    end
    # normalize u
    for i in 1:n_2
        u[i] = u[i]/sqrt(tot)/10;  
    end
    

    mu_not_homog = zeros(2^n_2);
    tot = 0;
    for s in 1:2^n_2
        u_scalar_x = 0;
        h_scalar_x = 0;
        for i in 1:n_2
            if s÷(2^(i-1))%2==1
                u_scalar_x += u[i];
                h_scalar_x += h[i];
            end
        end
        mu_not_homog[s] = exp(u_scalar_x^2/2+h_scalar_x);
        tot += mu_not_homog[s]; 
    end

    mu = zeros(2^n);
    for s in 1:2^n
        if (count_ones(s) == n_2) && ((s÷(2^n_2))+(s%2^n_2) == 2^n_2-1) && s%(2^n_2) != 0
            mu[s] = mu_not_homog[s%(2^n_2)]/tot;
        end
    end
    mu[2^n-2^n_2] = mu_not_homog[2^n_2]/tot;
    return mu
end;

function two_spin_system(
        n::Int64,
        k::Int64)
    if n != 2*k
        println("Check n and k.");
    end
    # antiferromagentic 2-spin system
    beta = 0;
    gamma = 1.1;
    lambda = 0.8;

    n_2 = n÷2;

    # star graph 
    # the center is node 0

    mu_not_homog = zeros(2^n_2);
    tot = 0;
    for s in 1:2^n_2
        mu_not_homog[s] = 0;
        if s%2 == 0
            m_0 = 0;
            for i in 2:n_2
                if s÷(2^(i-1))%2==0
                    m_0 += 1;
                end
            end
            mu_not_homog[s] = gamma^m_0*lambda^(n_2-m_0-1);
        else
            m_1 = 0;
            for i in 2:n_2
                if s÷(2^(i-1))%2==1
                    m_1 += 1;
                end
            end
            mu_not_homog[s] = beta^m_1*lambda^(m_1+1);
        end
        tot += mu_not_homog[s];
    end
    mu = zeros(2^n);
    for s in 1:2^n
        if (count_ones(s) == n_2) && ((s÷(2^n_2))+(s%2^n_2) == 2^n_2-1) && s%(2^n_2) != 0
            mu[s] = mu_not_homog[s%(2^n_2)]/tot;
        end
    end
    mu[2^n-2^n_2] = mu_not_homog[2^n_2]/tot;
    
    return mu;
end;

function sparse_distribution(
        n::Int64,
        k::Int64)
    if n%k != 0
        println("check n and k.");
    end
    mu = zeros(2^n);
    for i in 1:n÷k
        s = 0;
        for j in 0:k-1
            s += 2^(k*(i-1)+j);
        end
        mu[s] = 1/(n÷k);
    end
    return mu;
end;

function k_dpp_mu(
        n::Int64,
        k::Int64)
    A = rand(n,n);
    matrix =  A'*A;

    mu = zeros(2^n);
    tot = 0;
    for s in 1:2^n
        if count_ones(s) == k
            index = [];
            for i in 1:n
                if s÷(2^(i-1))%2==1
                    push!(index,i);
                end
            end
            mu[s] = det(matrix[index,index]);
            tot += mu[s];
        end
    end

    for s in 1: 2^n
        mu[s] = mu[s]/tot;
    end
    return mu;
end;

function power_k_dpp_mu(
        n::Int64,
        k::Int64,
        p)
    A = rand(n,n);
    matrix =  A'*A;

    mu = zeros(2^n);
    tot = 0;
    for s in 1:2^n
        if count_ones(s) == k
            index = [];
            for i in 1:n
                if s÷(2^(i-1))%2==1
                    push!(index,i);
                end
            end
            mu[s] = det(matrix[index,index])^p;
            tot += mu[s];
        end
    end

    for s in 1: 2^n
        mu[s] = mu[s]/tot;
    end
    return mu;
end;

function dpp_mu(
        n::Int64,
        k::Int64)
    if n != 2*k
        println("Check n and k!");
    end
    n_2 = k;
    A = rand(n_2,n_2);
    matrix =  A'*A;

    mu_not_homog = zeros(2^n_2);
    tot = 0;
    for s in 1:2^n_2
        index = [];
        for i in 1:n_2
            if s÷(2^(i-1))%2==1
                push!(index,i);
            end
        end
        mu_not_homog[s] = det(matrix[index,index]);
        tot += mu_not_homog[s];
    end
    
    mu = zeros(2^n);
    for s in 1:2^n
        if (count_ones(s) == n_2) && ((s÷(2^n_2))+(s%2^n_2) == 2^n_2-1) && s%(2^n_2) != 0
            mu[s] = mu_not_homog[s%(2^n_2)]/tot;
        end
    end
    mu[2^n-2^n_2] = mu_not_homog[2^n_2]/tot;
    
    return mu;
end;

function bad_mu(n::Int64,k::Int64)
    mu = zeros(2^n);
    tot = 0;
    for s in 1:2^n
        if count_ones(s) == k
            mu[s] = 1/rand()^5-1;
            tot += mu[s];
        end
    end
    for s in 1:2^n
        mu[s] = mu[s]/tot;
    end
    return mu;
end;


# -- Fano matroid --

fano_forbidden_sets = [
    (1,2,3),
    (1,4,5),
    (1,6,7),
    (2,4,6),
    (2,5,7),
    (3,4,7),
    (3,5,6)
];
fano_forbidden_outcomes = zeros(7);
for i in 1:7
    fano_forbidden_outcomes[i] = 2^(fano_forbidden_sets[i][1]-1)+2^(fano_forbidden_sets[i][2]-1)+2^(fano_forbidden_sets[i][3]-1);
end


function fano_matroid(n,k)
    if (n!=7) || (k!=3)
        println("Check n and k");
    end
    mu = zeros(2^n);
    for s in 1:2^n
        if count_ones(s) == k && !(s in fano_forbidden_outcomes)
            mu[s] = 1/(28)
        end
    end
    return mu;
end;


