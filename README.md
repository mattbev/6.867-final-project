# GAN Adversarial Attacks
Generative adversarial attacks on federate using GANs and UAP.

## Usage
There exist BasicBaseline and FederatedBaseline classes in `baseline.py`. The usage is very similar between the two.

### BasicBaseline
To create, train, and test a basic CNN baseline perform the following:
```
basic_baseline = BasicBaseline() # initialize a basic CNN
basic_baseline.load_data() # load the FashionMNIST data
basic_baseline.train(num_epochs, lr) # train the model for num_epochs epochs and at learning rate lr
basic_baseline.test() # test the accuracy of the model
```

### FederatedBaseline
To create, train, and test a federated CNN baseline perform the following:
```
federated_baseline = FederatedBaseline(num_clients) # initialize a FL framework with num_clients clients training CNNs
federated_baseline.load_data() # load the FashionMNIST data
federated_baseline.train(num_epochs, rounds, lr) # train the global model for num_epochs epochs, rounds rounds over the clients, and at learning rate lr
federated_baseline.test() # test the accuracy of the global model
```

# (internal) Strategy suggestion 

## Presentation
I think we can proceed as following, since we only have 10 minutes to present.
1 - We should aim at most 10 slides, spending 5 minutes motivating and explaining the problem and the rest showing our results. 
2 - The document on piazza said we could submit supplemental slides if helpfull, to be honest I think this should be our last priority
3 - We go as far as we can in the project and then write a last slide with future directions.

## Coding and building up the model
(The reference number 

### Setting up the federated learning enviroment and basic attacks
1 -  I think we should first set up a model of federated learning (using the MNIST dataset), where malicious users only train a GAN to reconstruct other
participants private data as in [3].  The original paper does not have a code, but I found a replication:
https://github.com/Jaskiee/GAN-Attack-against-Federated-Deep-Learning

2- Then we just add the label flipping strategy described in [10]

### Modifying the attack  and defenses

UAP attack -
1 - The next step is to implement the proposed attack, the original paper does not provides a code, but a pretty close implementation has
https://github.com/phibenz/uap_virtual_data.pytorch


Defense
2 -  I could not find codes for sniper and the GAN based defense, so we may try to code them. I  think the GAN one will be far harder to code, and we should focus on sniper.
Alternatively we could also try to focus on simplifying and implement the defense mechanism on (this paper also does not provide code) https://arxiv.org/pdf/2004.12571.pdf

However, I do find a paper that focus on defense against label flipping and has a code, so  I think we could start using that, this can also serve as a baseline for 
setting up the federated learning system (although there is no GAN attack) : https://arxiv.org/pdf/2007.08432v2.pdf ( code at https://github.com/git-disl/DataPoisoning_FL)

I just found this paper searchin on paperswithcode. We would depart a little from the strategy document but I think this is OK.

3 - I think if time becomes a problem we can work on coding attack and defense independently so it will be faster.

### Time permiting, other datasets or building a defense mechanism

## Dependencies
- pytorch
- numpy

