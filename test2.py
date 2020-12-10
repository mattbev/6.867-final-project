from baseline import BasicBaseline, FederatedBaseline
from utils.attacks import TargetedAttack, NoAttack
from utils.basics import load_model, save_model
from utils.defenses import FlippedLabelsDefense

def main(global_model, client_models):
    defense = FlippedLabelsDefense(num_classes=1)

    print(len(defense.run(global_model, client_models, plot_name="fig.png", verbose=True)))




if __name__ == "__main__":
    global_model = BasicBaseline().model
    load_model(global_model, "basic")

    client_model_1 = BasicBaseline().model
    load_model(client_model_1, "no_attack")

    client_model_2 = BasicBaseline().model
    load_model(client_model_2, "targeted_attack_7_4")

    client_model_3 = BasicBaseline().model
    load_model(client_model_3, "targeted_attack_7_4")

    client_models = [client_model_1, client_model_2, client_model_3]
    client_models.extend([load_model(BasicBaseline().model, "targeted_attack_7_4") for i in range(20)])
    main(global_model=global_model, client_models=client_models)

    # tool = FederatedBaseline(num_clients=5)
    # tool.load_data()
    # trainloader = tool._make_client_trainloaders()[0]

    # client = BasicBaseline()
    # client.set_trainloader(trainloader)
    # # client.configure_attack(TargetedAttack(target_label=7, target_class=4))
    # client.configure_attack(NoAttack())
    # client.train(
    #     num_epochs=2,
    #     lr=1e-3,
    #     verbose=True
    # )
    # save_model(client.model, "no_attack")