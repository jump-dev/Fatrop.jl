# Copyright (c) 2026: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module Fatrop

import Libdl
import MathOptInterface as MOI
import blasfeo_jll

# TODO use `fatrop_jll` once it is available in Yggdrasil
const libfatrop = get(ENV, "JULIA_FATROP_LIBRARY_PATH", "libfatrop")

function __init__()
    # `libfatrop` depends on `libblasfeo` which it finds only if it is
    # already loaded.
    Libdl.dlopen(blasfeo_jll.blasfeo)
    return
end

include("C_wrapper.jl")
include("structure.jl")
include("MOI_wrapper.jl")

end # module Fatrop
