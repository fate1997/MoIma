from moima.model import ModelFactory


def test_model_factory():
    """Test ModelFactory."""
    print(ModelFactory.avail)
    assert ModelFactory.avail == ['chemical_vae', 'vade', 'mlp']
    model = ModelFactory.create(name='mlp', 
                                input_dim=10, 
                                hidden_dim=10,
                                n_layers=3,
                                dropout=0.1,
                                activation='relu', 
                                output_dim=10)
    assert model.__class__.__name__ == 'MLP'
    
    model = ModelFactory.create(name='vade', 
                                vocab_size=35, 
                                enc_hidden_dim=292, 
                                latent_dim=292, 
                                emb_dim=128, 
                                dec_hidden_dim=501, 
                                n_clusters=10, 
                                dropout=0.1)
    assert model.__class__.__name__ == 'VaDE'
    
    model = ModelFactory.create(name='chemical_vae',
                                vocab_size=35,
                                enc_hidden_dim=292,
                                latent_dim=292,
                                emb_dim=128,
                                dec_hidden_dim=501,
                                dropout=0.1)
    assert model.__class__.__name__ == 'ChemicalVAE'
    