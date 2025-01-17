from group83_mlops.model import Generator, Discriminator
import torch
import pytest

def test_gen_latent_space_type():
    """Test that the latent space type is checked for the generator"""
    gen_model = Generator(latent_space_size=1000)
    with pytest.raises(AttributeError):
        gen_model = Generator(latent_space_size='1000')

def test_gen_latent_space_size():
    """Test that the integer bounds are respected for the generator latent space"""
    gen_model = Generator(latent_space_size=1)
    with pytest.raises(ValueError):
        gen_model = Generator(latent_space_size=0)
        gen_model = Generator(latent_space_size=-1)

def test_gen_output_space_size():
    """Check that the output of the generator is between -1 and 1, and throw an error, if no input is sufficiently negative or positive"""
    torch.manual_seed(42)
    output = []
    runs = 5
    gen_model = Generator(latent_space_size=100)
    for i in range(runs):
        dummy_input = torch.randn(1, 100)
        output.extend(gen_model(dummy_input).detach().cpu().numpy().tolist()[0])
    output.sort()

    assert output[0] < -0.5 and output[0] >= -1.0
    assert output[len(output) - 1] > 0.5 and output[len(output) - 1] <= 1.0

def test_dis_output_space_size():
    """Check that the output of the discriminator is between 0 and 1, and throw an error, if no input is sufficiently low or high"""
    torch.manual_seed(42)
    output = []
    runs = 1000
    dis_model = Discriminator()
    for i in range(runs):
        dummy_input = torch.randn(1, 3, 32, 32)
        output.extend(dis_model(dummy_input).detach().cpu().numpy().tolist()[0])
    output.sort()

    assert output[0] < 0.33 and output[0] >= 0
    assert output[len(output) - 1] > 0.67 and output[len(output) - 1] <= 1