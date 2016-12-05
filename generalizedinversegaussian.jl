doc"""
    GeneralizedInverseGaussian(a,b,p)

The *generalized inverse Gaussian distribution* with parameters a > 0, b > 0, p real,
and modified Bessel function of the second kind K_p, has probability density function

$f(x; a, b, p) = \frac{(a/b)^{p/2}}{2K_p(\sqrt{ab})}x^{(p-1)}
e^{-(ax+b/x)/2}, \quad x > 0$

```julia
GeneralizedInverseGaussian(a, b, p)    # Generalized Inverse Gaussian distribution with parameters parameters a > 0, b > 0 and p real

params(d)           # Get the parameters, i.e. (a, b, p)
```

External links

* [Generalized Inverse Gaussian distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution)

"""
immutable GeneralizedInverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    p::T

    function GeneralizedInverseGaussian(a::T, b::T, p::T)
        @check_args(GeneralizedInverseGaussian, a > zero(a) && b > zero(b))
        new(a, b, p)
    end
end

GeneralizedInverseGaussian{T<:Real}(a::T, b::T, p::T) = GeneralizedInverseGaussian{T}(a, b, p)
GeneralizedInverseGaussian(a::Real, b::Real, p::Real) = GeneralizedInverseGaussian(promote(a, b, p)...)
GeneralizedInverseGaussian(a::Integer, b::Integer, p::Integer) = GeneralizedInverseGaussian(Float64(a), Float64(b), Float64(p))

@distr_support GeneralizedInverseGaussian 0.0 Inf

#### Conversions

function convert{T <: Real, S <: Real}(::Type{GeneralizedInverseGaussian{T}}, a::S, b::S, p::S)
    GeneralizedInverseGaussian(T(a), T(b), T(p))
end
function convert{T <: Real, S <: Real}(::Type{GeneralizedInverseGaussian{T}}, d::GeneralizedInverseGaussian{S})
    GeneralizedInverseGaussian(T(d.a), T(d.b), T(d.p))
end

#### Parameters

params(d::GeneralizedInverseGaussian) = (d.a, d.b, d.p)
@inline partype{T<:Real}(d::InverseGaussian{T}) = T


#### Statistics

mean(d::GeneralizedInverseGaussian) = (sqrt(d.b) * besselk((d.p + 1), sqrt(d.a * d.b))) / (sqrt(d.a) * besselk(d.p, sqrt(d.a * d.b)))

var(d::GeneralizedInverseGaussian) = (b / a) * ((besselk(d.p + 2, sqrt(d.a * d.b)) / (besselk(d.p, sqrt(d.a * d.b)))) -
 (besselk(d.p + 1, sqrt(d.a * d.b)) / (besselk(d.p, sqrt(d.a * d.b))))^2)

mode(d::GeneralizedInverseGaussian) = ((d.p - 1) + sqrt((d.p - 1)^2 + d.a * d.b)) / d.a


#### Evaluation

function pdf{T<:Real}(d::GeneralizedInverseGaussian{T}, x::Real)
    if x > 0
        a, b, p = params(d)
        return (((a / b)^(p / 2)) / (2 * besselk(p, sqrt(a * b)))) * (x^(p - 1)) * exp(- (a * x + b / x) / 2)
    else
        return zero(T)
    end
end

function logpdf{T<:Real}(d::GeneralizedInverseGaussian{T}, x::Real)
    if x > 0
        a, b, p = params(d)
        return log(sqrt(a) / sqrt(b)) + Calculus.derivative(log(besselk(p, sqrt(a * b))) # How to get the derivative? Maybe log(pdf(x))
    else
        return -T(Inf)
    end
end

function cdf{T<:Real}(d::GeneralizedInverseGaussian{T}, x::Real)
  if x > 0
      a, b, p = params(d)
        return # (5) in Lemonte & Cordeiro 2011 Statistics & Probability Letters 81:506–517
    else
        return zero(T)
    end
end


@quantile_newton GeneralizedInverseGaussian

#### Sampling

# rand method from:
# Hörmann, W. & J. Leydold. (2014). Generating generalized inverse Gaussian random variates.
# J. Stat. Comput. 24: 547–557. doi:10.1007/s11222-013-9387-3

function rand(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    # Algorithm 1
    # Algorithm 2
    # Algorithm 3
end
