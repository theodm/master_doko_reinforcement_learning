import pytest
import torch
from main import ImperfectInformationNetwork, ImperfectInformationNetworkConfig  # Passe den Modulpfad an

@pytest.fixture
def config():
    return ImperfectInformationNetworkConfig(
        device="cpu", 
        n_embd=64,
        n_head=8,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        n_layer=2
    )

@pytest.fixture
def model(config):
    return ImperfectInformationNetwork(config)

def test_forward(model):
    batch_size = 512  
    input_tensor = torch.randint(0, 1, (batch_size, 311), dtype=torch.int64) 

    output = model.forward(input_tensor)

    print(output)

    assert output.shape == (batch_size, 33), f"Erwartete Shape (batch, 33), aber erhalten: {output.shape}"

    assert not torch.isnan(output).any(), "Output enthält NaN-Werte!"
