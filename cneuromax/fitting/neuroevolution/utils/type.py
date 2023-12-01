"""Typing for various Neuroevolution fitting variables."""

from nptyping import Float32, NDArray, Object, Shape, UInt32

agents_batch_type = NDArray[
    Shape["LEN AGENTS BATCH, NUM POPS, AGENT SIZE"],
    Object,
]
agents_type = NDArray[Shape["POP SIZE, NUM POPS, AGENT SIZE"], Object]
exchange_and_mutate_info_batch_type = NDArray[
    Shape[
        "LEN AGENTS BATCH, NUM POPS, "
        "[mpi buffer size, pair position, sending, seeds]"
    ],
    UInt32,
]
exchange_and_mutate_info_type = NDArray[
    Shape[
        "POP SIZE, NUM POPS, [mpi buffer size, pair position, sending, seeds]"
    ],
    UInt32,
]
fitnesses_and_num_env_steps_batch_type = NDArray[
    Shape["LEN AGENTS BATCH, NUM POPS, [fitness, num env steps]"],
    Float32,
]
generation_results_batch_type = NDArray[
    Shape["LEN AGENTS BATCH, NUM POPS, [fitness, seeds, num env steps]"],
    Float32,
]
generation_results_type = NDArray[
    Shape["POP SIZE, NUM POPS, [fitness, seeds, num env steps]"],
    Float32,
]
