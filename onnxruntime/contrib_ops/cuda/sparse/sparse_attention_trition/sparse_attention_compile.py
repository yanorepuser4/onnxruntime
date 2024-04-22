# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Use triton AoT compiler to convert sparse_attention_triton.py to C source files including cubin and dispatcher.
# Example to use this script (Tested with triton 2.3.0 in Ubuntu 20.04):
#    python sparse_attention_compile.py | sh
#
# Note that fbsa_*.h, fbsa_*.cc and sparse_attention_api.cc under this directory is modified from the generated files.

import math
from itertools import product

def generate_triton_compile_shell_script(dtype = "fp16"):
    assert(dtype in ["fp16", "bf16"])
    print("export TRITON_ROOT=$(pip show triton | grep Location | cut -d' ' -f2)")
    print("export ARCH=\"$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1)\"")
    print("export SM=$(echo $ARCH | sed -e 's/\\.//g')")
    out_dir = f"trition_cubin_{dtype}"
    print(f"rm -rf {out_dir}")
    print(f"mkdir -p {out_dir}")

    block_n_values = [64]
    block_d_values = [64]
    num_block_d_values = [2]
    even_m_values = [True, False]
    even_n_values = [True, False]

    for block_n, block_d, num_blocks_d, even_m, even_n in product(
        block_n_values, block_d_values, num_block_d_values, even_m_values, even_n_values
    ):
        block_m_values = [16, block_n] if block_n != 16 else [block_n]
        for block_m in block_m_values:
            scalar_params = "i32,i32,i32,fp32,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32,i32"
            sig = f"*{dtype}:16,*{dtype}:16,*{dtype}:16,*{dtype}:16,*i32:16,*i32:16,{scalar_params},{block_m},{int(even_m)},{block_n},{int(even_n)},{block_d},{num_blocks_d}"
            prefix = "python ${TRITON_ROOT}/triton/tools/compile.py sparse_attention_triton.py"
            filename = "sparse_attention_kernel_" + f"{dtype}_m{block_m}_{int(even_m)}_n{block_n}_{int(even_n)}_d{block_d}_{num_blocks_d}" + "_sm${SM}"
            name="fbsa_sm${SM}_" + f"{dtype}"
            num_warps = max(1, 2 ** int(math.log2(min(block_m, block_n, block_d) / 16)))
            num_stages = 2
            print(f"{prefix} -n block_sparse_attention_kernel -o {out_dir}/{filename} --out-name {name} -w {num_warps} -ns {num_stages} -s \"{sig}\" -g \"(total_seq_len - past_seq_len + {block_m} - 1) / {block_m}, batch_size * num_heads, 1\"")

    print(f"cd {out_dir}")
    print("python ${TRITON_ROOT}/triton/tools/link.py " + f"sparse_attention_kernel_*.h -o sparse_attention_api_{dtype}" + "_sm${SM}")

    # The following postprocessing are optional since we will edit those files manually.
    # print("for file in *.c; do sed -i 's/int32_t past_seq_len/int32_t past_seq_len, int32_t batch_size/g'  \"$file\"; done")
    # print(f"sed -i 's/ past_seq_len)/ past_seq_len, batch_size)/g'  \"sparse_attention_api_{dtype}" + "_sm${SM}.c\"")
    # print("for file in *.h; do sed -i 's/int32_t past_seq_len/int32_t past_seq_len, int32_t batch_size/g'  \"$file\"; done")
    # print("for file in *.c; do sed -i '/gX \\* gY \\* gZ/d'  \"$file\"; done")

    # Remove signature hash in code.
    print("for file in *.h; do sed -i 's/_0d1d2d3d4d5d678910d11d12d13d14d15d16d17d18d19d20d21d22d2324//g'  \"$file\"; done")
    print("for file in *.c; do sed -i 's/_0d1d2d3d4d5d678910d11d12d13d14d15d16d17d18d19d20d21d22d2324//g'  \"$file\"; done")

    # Keep the signature hash in kernel name.
    print("for file in *.c; do sed -i 's/block_sparse_attention_kernel/block_sparse_attention_kernel_0d1d2d3d4d5d678910d11d12d13d14d15d16d17d18d19d20d21d22d2324/g'  \"$file\"; done")

    # Remove signature hash from filename since we use same signature for all kernels except constants.
    # and we have constants in filename so that we can distinguish them.
    print("for file in *.h; do mv -- \"$file\" \"$(echo $file | cut -f 1 -d '.').h\"; done")
    print("for file in *.c; do mv -- \"$file\" \"$(echo $file | cut -f 1 -d '.').c\"; done")

    # rename *.c to *.cc
    print("for file in *.c; do mv -- \"$file\" \"${file%.c}.cc\"; done")

if __name__ == "__main__":
    generate_triton_compile_shell_script("fp16")
