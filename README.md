# Deep FL Adversarial Attacks and Defenses
attacks and defenses on deep federated learning using and UAP and GANs.

## Usage

### Baseline Models
Viewable in `baseline.py`.

To create, train, and test a basic CNN baseline perform the following:
```python
basic_baseline = BasicBaseline() # initialize a basic CNN
basic_baseline.load_data() # load the FashionMNIST data
basic_baseline.configure_attack(attack) # configure attack 
basic_baseline.train(num_epochs, lr) # train the model for num_epochs epochs and at learning rate lr
basic_baseline.test() # test the accuracy of the model
```

To create, train, and test a federated CNN baseline perform the following:
```python
federated_baseline = FederatedBaseline(num_clients) # initialize a FL framework with num_clients clients training CNNs
federated_baseline.load_data() # load the FashionMNIST data
federated_baseline.configure_attack(attack, num_malicious) # configure num_malicious attackers using attack
federated_baseline.configure_defense(defense) # configure a defense model
federated_baseline.train(num_epochs, rounds, lr) # train the global model for num_epochs epochs, rounds rounds over the clients, and at learning rate lr
federated_baseline.test() # test the accuracy of the global model
```

### Attacks
viewable in `utils/attacks.py`.

for no attack:
```python
attack = NoAttack()
```

for random attack:
```python
attack = RandomAttack(num_classes) # where num_classes is the total number of classes in the data
```

for targeted attack:
```python
attack = TargetedAttack(target_label, class_label) # set training labels of target_label to class_label
```

for UAP attack:
```python
attack = UAPAttack(target_label)
```

for GAN attack:
```python
attack = GANAttack(client_model)
```
note that the GAN attack differs from the others as this does not affect the global model but instead aims to reconstruct user data. Thus its usage varies from the aforementioned attacks.

### Defenses
viewable in `utils/defenses.py`

for no defenese:
```python
defense = NoDefense()
```

for flipped labels defense:
```python
defense = FlippedLabelsDefense(num_classes) # where num_classes denotes source class comparisons
```


### Running baselines from bash
To run the basic baseline:
```
python3 baseline.py basic
```
To run the federated baseline:
```
python3 baseline.py federated
```


# Future work
1. finish GAN attack implementation 
2. create a `main.py` file to be able to run everything from the command line with flag arguments
3. reconfigure to work smoothly both on cpu and cuda

# References
client data reconstruction: https://github.com/Jaskiee/GAN-Attack-against-Federated-Deep-Learning

UAP attack: https://github.com/phibenz/uap_virtual_data.pytorch

label flipping defense: https://arxiv.org/pdf/2007.08432v2.pdf (code at https://github.com/git-disl/DataPoisoning_FL)
