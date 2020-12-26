using MultiFloats

import StructArrays
import Base: getproperty

# minimal effort to get MultiFloat struct of arrays to work.
# todo: complex.
Base.getproperty(a::MultiFloat, i::Int) = a._limbs[i]
StructArrays.staticschema(::Type{MultiFloat{T,M}}) where {T,M} = NTuple{M,T}
StructArrays.createinstance(::Type{MultiFloat{T,M}}, args::Vararg{T,M}) where {T,M} = MultiFloat{T,M}(values(args))