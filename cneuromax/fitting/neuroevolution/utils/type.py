"""Typing for various Neuroevolution fitting variables."""

from nptyping import Float32, NDArray, Shape, UInt32

exchange_and_mutate_info_batch_type = NDArray[
    Shape[
        "LEN_AGENTS_BATCH, NUM_POPS, "
        "[mpi_buffer_size, agent_pair_position, sending, seeds]"
    ],
    UInt32,
]
exchange_and_mutate_info_type = NDArray[
    Shape[
        "POP_SIZE, NUM_POPS, "
        "[mpi_buffer_size, agent_pair_position, sending, seeds]"
    ],
    UInt32,
]
fitnesses_and_num_env_steps_batch_type = NDArray[
    Shape["LEN_AGENTS_BATCH, NUM_POPS, [fitness, num_env_steps]"],
    Float32,
]
generation_results_batch_type = NDArray[
    Shape[
        "LEN_AGENTS_BATCH, NUM_POPS, "
        "[fitness, num_env_steps, serialized_agent_size]"
    ],
    Float32,
]
generation_results_type = NDArray[
    Shape[
        "POP_SIZE, NUM_POPS, [fitness, num_env_steps, serialized_agent_size]"
    ],
    Float32,
]
seeds_type = NDArray[Shape["POP_SIZE, NUM_POPS"], UInt32]
