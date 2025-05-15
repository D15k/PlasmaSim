from dataclasses import dataclass, asdict

@dataclass
class Params:
    alpha: float
    beta: int
    name: str = "default"

class MySim:
    def __init__(self, params: Params):
        # Copy dataclass fields into the instance
        for key, value in asdict(params).items():
            setattr(self, key, value)

# Example usage
params = Params(alpha=1.23, beta=7)
sim = MySim(params)

print(sim.alpha)  # 1.23
print(sim.beta)   # 7
print(sim.name)   # default
