import random
import numpy as np

import torch
from main_old import Dataset, NonSymmetricDPP, Experiment, Args

from nonsymmetric_dpp_learning import compute_prediction_metrics
from utils import (logging, parse_cmdline_args)
from sklearn.utils import check_random_state
from results import Results

# control random-number generators
torch.manual_seed(1234)
random.seed(1446)
np.random.seed(13564)

# Set default for floating point to torch.float64
torch.set_default_tensor_type(torch.DoubleTensor)


class OrthogonalNDPP(NonSymmetricDPP):

    def __init__(self, **kwargs):
        super(OrthogonalNDPP, self).__init__(**kwargs)
        self.d_params = torch.nn.Parameter(torch.randn(self.num_nonsym_embedding_dims // 2))

    def get_sigmas(self):
        return torch.exp(self.d_params)

    def forward(self, _):
        V = self.get_v_embeddings().to(self.device)
        B = self.get_b_embeddings()
        B, _ = torch.qr(B)

        B = B.to(self.device)

        D = torch.zeros(self.num_nonsym_embedding_dims).to(self.device)
        D[::2] = self.get_sigmas()
        D = torch.diag_embed(
            D,
            offset=1,
        )[:-1, :-1]

        if self.ortho_v:
            V = V - B @ torch.linalg.solve(B.T @ B, B.T @ V)
            return V, B, D
        else:
            return V, B, D


class NewExperiment(Experiment):

    @classmethod
    def _build_model_object(cls, arguments, product_catalog, max_basket_size, seed):
        args = arguments.args
        model_cls = OrthogonalNDPP
        model_params = {
            param: getattr(args, param)
            for param in ["hidden_dims", "activation", "disable_gpu", "dropout", "noshare_v", "ortho_v", "num_threads"]
        }
        model_params["num_sym_embedding_dims"] = cls._compute_num_sym_embeddings(args)
        model_params["num_nonsym_embedding_dims"] = cls._compute_num_nonsym_embeddings(args)
        model_params["product_catalog"] = product_catalog
        features_setup = arguments.compute_features_setup(product_catalog)
        model_params["features_setup"] = features_setup
        model = model_cls(**model_params)

        if args.num_nonsym_embedding_dims == 0:
            model.disable_nonsym_embeddings = True
        else:
            model.disable_nonsym_embeddings = False

        logging.info("Built model:")
        print(model)
        return model


if __name__ == "__main__":
    arguments = Args.build_from_cli()
    args = arguments.args
    args.scores_file = args.scores_file.replace("scores", "scores-spectral")
    args_dict = arguments.args_dict
    num_val_baskets = args.num_val_baskets
    num_test_baskets = args.num_test_baskets
    seed = args.seed
    print(f"seed: {seed}")
    rng = check_random_state(seed)

    dataset = Dataset(args, seed, rng, num_val_baskets, num_test_baskets)

    model, ofile = NewExperiment.build(arguments, dataset)

    results_df = Experiment.run(model, arguments, dataset, store_inference_scores=True)
    res = Results(args.dataset_name, results_df)
    print(res)
