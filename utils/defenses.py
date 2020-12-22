import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fclusterdata


class NoDefense:
    def run(self, global_model, client_models, plot_name=None):
        return client_models

    def __repr__(self):
        return f"(attack=NoDefense)"


class FlippedLabelsDefense:
    def __init__(self, num_classes, layer_name="fc.weight"):
        self.num_classes = num_classes
        self.layer_name = layer_name


    def run(self, global_model, client_models, plot_name=None, verbose=False):
        """
        identifies malicious clients with flipping label attack
        Args:
            global_model (nn.Module): the global model
            client_models (list[nn.Module]): the client models
            plot_name (str, optional): the name of the figure to save. Defaults to None.
            verbose (boolean, optional): extended print output? Defaults to False.
        """    
        label_sets = []
        for source_class in range(self.num_classes):            
            param_diff = []
            global_params = list(global_model.state_dict()[self.layer_name])[source_class]
            for client in client_models:
                client_params = list(client.state_dict()[self.layer_name])[source_class]
                gradient = np.array([x for x in np.subtract(global_params, client_params)]).flatten()
                param_diff.append(gradient)


            scaler = StandardScaler()
            scaled_param_diff = scaler.fit_transform(param_diff)
            pca = PCA(2)
            dim_reduced_gradients = pca.fit_transform(scaled_param_diff)

            labels = fclusterdata(dim_reduced_gradients, t=2, criterion="maxclust")
            label_sets.append(labels)

            if verbose:
                print("Labels:", labels)
                print("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))
                print("Prescaled gradients: {}".format(str(param_diff)))
                print("Postscaled gradients: {}".format(str(scaled_param_diff)))
                print("PCA reduced gradients: {}".format(str(dim_reduced_gradients)))
                print("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))

        malicious_clients = np.any(np.array(label_sets) - 1, axis=0) # maps most common label to 1, and other to 2
        if plot_name:
            self.plot_gradients_2d(dim_reduced_gradients, malicious_clients, plot_name)

        return [client_models[i] for i in range(len(client_models)) if malicious_clients[i] == False]


    @staticmethod
    def plot_gradients_2d(gradients, labels, name="fig.png"):
        for i in range(len(gradients)):
            gradient = gradients[i]
            params = (("blue", "x") if labels[i] else ("orange", "."))
            color, marker = params
            plt.scatter(gradient[0], gradient[1], color=color, marker=marker, s=1000, linewidth=5)
        plt.savefig(f"figures/{name}")


    def __repr__(self):
        return f"(attack=FlippedLabelsDefense, num_classes={self.num_classes}, layer_name={self.layer_name})"