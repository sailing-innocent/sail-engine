import pytest 
import torch 
from module.model.heuristic.aco.deep_aco_tsp import ACO 

# see git@github.com:henry-yeh/DeepACO.git

@pytest.mark.current 
def test_deep_aco():
    _input = torch.rand(size=(5,2))
    distances = torch.norm(_input[:,None] - _input, dim=2, p=2)
    print(distances[0][1])
    print(distances[1][0])
    distances[torch.arange(len(distances)), torch.arange(len(distances))] = 1e10
    aco = ACO(distances)
    aco.sparsify(k_sparse=3)
    print(aco.run(20))
    assert True 