"""Typing for various Neuroevolution fitting variables."""

# mypy: ignore-errors
import numpy as np
from nptyping import Shape

Exchange_and_mutate_info_batch_type = np.ndarray[
    Shape[
        "Len_agents_batch, Num_pops, "
        "[mpi_buffer_size, agent_pair_position, sending, seeds]"
    ],
    np.dtype[np.uint32],
]
Exchange_and_mutate_info_type = np.ndarray[
    Shape[
        "Pop_size, Num_pops, "
        "[mpi_buffer_size, agent_pair_position, sending, seeds]"
    ],
    np.dtype[np.uint32],
]
Fitnesses_and_num_env_steps_batch_type = np.ndarray[
    Shape["Len_agents_batch, Num_pops, [fitness, num_env_steps]"],
    np.dtype[np.float32],
]
Generation_results_batch_type = np.ndarray[
    Shape[
        "Len_agents_batch, Num_pops, "
        "[fitness, num_env_steps, serialized_agent_size]'"
    ],
    np.dtype[np.float32],
]
Generation_results_type = np.ndarray[
    Shape[
        "Pop_size, Num_pops, [fitness, num_env_steps, serialized_agent_size]"
    ],
    np.dtype[np.float32],
]
Seeds_type = np.ndarray[Shape["Pop_size, Num_pops"], np.dtype[np.uint32]]
Seeds_batch_type = np.ndarray[
    Shape["Len_agents_batch, Num_pops"],
    np.dtype[np.uint32],
]
