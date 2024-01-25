2023-11-23
- [x] Maybe loss should be calculated in the `forward` function of the model (however, some loss have parameters like beta)
- [x] model's `forward` function should input the `data` object, and output a `dict` object, which contains the `loss` and `output` of the model 
- [x] Multiple loss should be supported
- [x] SMILESFeaturizer is much lower than VANILA | @property problem
- [x] sample function should be in the model (however, in that case, the model should have access to the featurizer)
- [ ] self.training_trace
- [ ] vocab size should be inherited from the dataset
- [ ] using workdir as output dir
- [x] merge featurizer arguments into dataset?
- [ ] GraphVAE - https://github.com/JiaxuanYou/graph-generation
- [ ] JTVAE
- [ ] GraphAF
- [ ] Make `pipeline` be a context manager?


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