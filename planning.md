2023-11-23
- [ ] Maybe loss should be calculated in the `forward` function of the model (some loss have parameters like beta)
- [ ] model's `forward` function should input the `data` object, and output a `dict` object, which contains the `loss` and `output` of the model
- [ ] Multiple loss should be supported
- [ ] SMILESFeaturizer is much lower than VANILA
- [ ] sample function should be in the model
- [ ] self.training_trace


2023-11-20
- [ ] All class to be a simple factory

2023-11-03
- [x] Finish `featurizer` part
- [x] Partially finish `Pipeline` part
- [ ] Better configuration support


2023-11-01

- [ ] Refer to https://github.com/DeepGraphLearning/torchdrug/blob/master/torchdrug/core/core.py#L317 for config writting
- [x] Refer to https://github.com/cxhernandez/molencoder/blob/master/molencoder/models.py and https://github.com/topazape/molecular-VAE/blob/master/sample.py for featurizer
- [x] Review the output shape of each layer in the ChemicalVAE model.