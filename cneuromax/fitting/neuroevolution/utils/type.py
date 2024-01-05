"""Typing for various Neuroevolution fitting variables."""
from nptyping import Float32, NDArray, Shape, UInt32

Exchange_and_mutate_info_batch_type = NDArray[
    Shape[
        "Len_agents_batch, Num_pops, "
        "[mpi_buffer_size, agent_pair_position, sending, seeds]"
    ],
    UInt32,
]
Exchange_and_mutate_info_type = NDArray[
    Shape[
        "Pop_size, Num_pops, "
        "[mpi_buffer_size, agent_pair_position, sending, seeds]"
    ],
    UInt32,
]
Fitnesses_and_num_env_steps_batch_type = NDArray[
    Shape["Len_agents_batch, Num_pops, [fitness, num_env_steps]"],
    Float32,
]
Generation_results_batch_type = NDArray[
    Shape[
        "'Len_agents_batch', 'Num_pops', "
        "[fitness, num_env_steps, serialized_agent_size]'"
    ],
    Float32,
]
Generation_results_type = NDArray[
    Shape[
        "Pop_size, Num_pops, [fitness, num_env_steps, serialized_agent_size]"
    ],
    Float32,
]
Seeds_type = NDArray[Shape["Pop_size, Num_pops"], UInt32]
