import random
from typing import Collection, TypeVar

T = TypeVar("T")


def sample_up_to(population: Collection[T], k: int) -> T:
    if not isinstance(population, list):
        population = list(population)
    if len(population) > k:
        population = random.sample(population, k)
    return population
