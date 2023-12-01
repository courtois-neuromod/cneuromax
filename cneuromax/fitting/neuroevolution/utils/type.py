"""Typing for various Neuroevolution fitting variables."""

from nptyping import Float32, NDArray, Object, Shape, UInt32

agents_batch_type = NDArray[
    Shape["NUM_POPS, LEN_AGENTS_BATCH"],
    Object,
]
agents_type = NDArray[Shape["NUM_POPS, POP SIZE"], Object]
exchange_and_mutate_info_batch_type = NDArray[
    Shape[
        "NUM_POPS, LEN_AGENTS_BATCH, "
        "[mpi_buffer_size, process_pair_position, sending, seeds]"
    ],
    UInt32,
]
exchange_and_mutate_info_type = NDArray[
    Shape[
        "NUM_POPS, POP SIZE, "
        "[mpi_buffer_size, process_pair_position, sending, seeds]"
    ],
    UInt32,
]
fitnesses_and_num_env_steps_batch_type = NDArray[
    Shape["NUM_POPS, LEN_AGENTS_BATCH, [fitness, num_env_steps]"],
    Float32,
]
generation_results_batch_type = NDArray[
    Shape[
        "NUM_POPS, LEN_AGENTS_BATCH, "
        "[fitness, num_env_steps, serialized_agent_size]"
    ],
    Float32,
]
generation_results_type = NDArray[
    Shape[
        "NUM_POPS, POP SIZE, [fitness, num_env_steps, serialized_agent_size]"
    ],
    Float32,
]
