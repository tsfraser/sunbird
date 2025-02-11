from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from sunbird.covariance import CovarianceMatrix
from sunbird.data import data_readers
from sunbird.summaries import Bundle


class Inference(ABC):
    def __init__(
        self,
        theory_model: "Summary",
        observation: np.array,
        covariance_matrix: np.array,
        priors: Dict,
        fixed_parameters: Dict[str, float],
        select_filters: Dict,
        slice_filters: Dict,
        output_dir: Path,
        add_predicted_uncertainty: bool = False,
        device: str = "cpu",
    ):
        """Given an inference algorithm, a theory model, and a dataset, get posteriors on the
        parameters of interest. It assumes a gaussian likelihood.

        Args:
            theory_model (Summary): model used to predict the observable
            observation (np.array): observed data
            covariance_matrix (np.array): covariance matrix of the data
            priors (Dict): prior distributions for each parameter
            fixed_parameters (Dict[str, float]): dictionary of parameters that are fixed and their values
            select_filters (Dict, optional): filters to select values in coordinates. Defaults to None.
            slice_filters (Dict, optional): filters to slice values in coordinates. Defaults to None.
            output_dir (Path): directory where results will be stored
            device (str, optional): gpu or cpu. Defaults to "cpu".
        """
        self.theory_model = theory_model
        self.observation = observation
        self.covariance_matrix = covariance_matrix
        self.add_predicted_uncertainty = add_predicted_uncertainty
        if not self.add_predicted_uncertainty:
            self.inverse_covariance_matrix = self.invert_covariance(
                covariance_matrix=self.covariance_matrix,
            )
        self.priors = priors
        self.n_dim = len(self.priors)
        self.fixed_parameters = fixed_parameters
        self.device = device
        self.param_names = list(self.priors.keys())
        self.select_filters = select_filters
        self.slice_filters = slice_filters
        self.output_dir = Path(output_dir)

    @classmethod
    def from_config(
        cls,
        path_to_config: Path,
        device: str = "cpu",
    ) -> "Inference":
        """Read from config file to fit one of the abacus summit
        simulations

        Args:
            path_to_config (Path): path to configuration file
            device (str, optional): device to use to run model. Defaults to "cpu".

        Returns:
            Inference: inference object
        """
        with open(path_to_config, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config_dict(
            config=config,
            device=device,
        )

    @classmethod
    def from_config_dict(cls, config: Dict, device: str = "cpu"):
        """Use dictionary config to fit a given dataset

        Args:
            config (Dict): dictionary with configuration
            device (str, optional): device to use to run model. Defaults to "cpu".

        Returns:
            Inference: inference object
        """
        select_filters = config["select_filters"]
        slice_filters = config["slice_filters"]
        statistics = config["statistics"]
        observation, parameters = cls.get_observation_and_parameters(
            config["data"]["observation"],
            statistics=statistics,
            select_filters=select_filters,
            slice_filters=slice_filters,
        )
        fixed_parameters = {}
        for k in config["fixed_parameters"]:
            fixed_parameters[k] = parameters[k]
        covariance_config = config["data"]["covariance"]
        if "volume_scaling" not in covariance_config:
            if covariance_config["class"] == "AbacusSmall":
                raise ValueError(
                    "Volume scaling must be specified when using AbacusSmall covariance class."
                )
            else:
                covariance_config["volume_scaling"] = 1.0
        theory_model = cls.get_theory_model(
            config["theory_model"], statistics=config["statistics"]
        )
        covariance_matrix = cls.get_covariance_matrix(
            covariance_data_class=covariance_config["class"],
            covariance_dataset=covariance_config["dataset"],
            add_emulator_error=covariance_config["add_emulator_error_test_set"],
            add_simulation_error=covariance_config["add_simulation_error"],
            volume_scaling=covariance_config["volume_scaling"],
            statistics=config["statistics"],
            select_filters=select_filters,
            slice_filters=slice_filters,
            theory_model=theory_model,
        )
        parameters_to_fit = [
            p for p in theory_model.input_names if p not in fixed_parameters.keys()
        ]
        priors = cls.get_priors(config["priors"], parameters_to_fit)
        return cls(
            theory_model=theory_model,
            observation=observation,
            select_filters=select_filters,
            slice_filters=slice_filters,
            covariance_matrix=covariance_matrix,
            fixed_parameters=fixed_parameters,
            priors=priors,
            output_dir=config["inference"]["output_dir"],
            add_predicted_uncertainty=covariance_config["add_predicted_uncertainty"],
            device=device,
        )

    @classmethod
    def get_observation_and_parameters(
        cls,
        obs_config: Dict,
        statistics: List[str],
        select_filters: Optional[Dict] = None,
        slice_filters: Optional[Dict] = None,
    ) -> Tuple[np.array]:
        """Get observation and parameters for a given dataset

        Args:
            obs_config (Dict): dictionary with configuration for the dataset
            statistics (List[str]): list of statistics to constrain
            select_filters (Dict, optional): filters to select values in coordinates. Defaults to None.
            slice_filters (Dict, optional): filters to slice values in coordinates. Defaults to None.

        Returns:
            Tuple: observation and parameters
        """

        obs_class = getattr(data_readers, obs_config["class"])(
            select_filters=select_filters,
            slice_filters=slice_filters,
            statistics=statistics,
            **obs_config.get("args", {}),
        )
        observation = obs_class.get_observation(**obs_config.get("get_obs_args", {}))
        parameters = obs_class.get_parameters_for_observation(
            **obs_config.get("get_obs_args", {})
        )
        return observation, parameters

    @classmethod
    def get_covariance_matrix(
        cls,
        covariance_data_class: str,
        covariance_dataset: str,
        statistics: List[str],
        select_filters: Dict,
        slice_filters: Dict,
        add_emulator_error: bool = True,
        add_simulation_error: bool = True,
        apply_hartlap_correction: bool = True,
        volume_scaling: float = 1.0,
        theory_model=None,
    ) -> np.array:
        """Compute covariance matrix for a list of statistics

        Args:
            covariance_data_class (str): class to use to compute covariance matrix
            statistics (List[str]): list of statistics
            select_filters (Dict): filters to select values along a dimension
            slice_filters (Dict): filters to slice values along a dimension
            add_emulator_error (bool, optional): whether to add in the emulator error. Defaults to True.
            add_simulation_error (bool, optional): whether to add in the simulation error. Defaults to True.
            apply_hartlap_correction (bool, optional): whether to correct the covariance matrix with the Hartlap factor
            to ensure the inverse covariance matrix is an unbiased estimator.
            volume_scaling (float, optional): scaling factor to apply to the volume of the covariance matrix. Defaults to 1.0.

        Returns:
            np.array: covariance matrix
        """
        if isinstance(theory_model, Bundle):
            emulators = theory_model.all_summaries
        else:
            emulators = {
                statistics[0]: theory_model,
            }
        covariance = CovarianceMatrix(
            covariance_data_class=covariance_data_class,
            dataset=covariance_dataset,
            statistics=statistics,
            select_filters=select_filters,
            slice_filters=slice_filters,
            emulators=emulators,
        )
        cov_data = covariance.get_covariance_data(
            apply_hartlap_correction=apply_hartlap_correction,
            volume_scaling=volume_scaling,
        )
        if add_emulator_error:
            cov_data += covariance.get_covariance_emulator(
            )
        if add_simulation_error:
            cov_data += covariance.get_covariance_simulation(
                apply_hartlap_correction=apply_hartlap_correction,
            )
        return cov_data

    @classmethod
    def get_priors(
        cls, prior_config: Dict[str, Dict], parameters_to_fit: List[str]
    ) -> Dict:
        """Initialize priors for a given configuration and a list of parameters to fit

        Args:
            prior_config (Dict[str, Dict]): configuration of priors
            parameters_to_fit (List[str]): list of parameteters that are being fitted

        Returns:
            Dict: dictionary with initialized priors
        """
        distributions_module = importlib.import_module(prior_config.pop("stats_module"))
        prior_dict = {}
        for param in parameters_to_fit:
            config_for_param = prior_config[param]
            prior_dict[param] = cls.initialize_distribution(
                distributions_module, config_for_param
            )
        return prior_dict

    @classmethod
    def initialize_distribution(
        cls, distributions_module, dist_param: Dict[str, float]
    ):
        """Initialize a given prior distribution fromt he distributions_module

        Args:
            distributions_module : module form which to import distributions
            dist_param (Dict[str, float]): parameters of the distributions

        Returns:
            prior distirbution
        """
        if dist_param["distribution"] == "uniform":
            max_uniform = dist_param.pop("max")
            min_uniform = dist_param.pop("min")
            dist_param["loc"] = min_uniform
            dist_param["scale"] = max_uniform - min_uniform
        if dist_param["distribution"] == "norm":
            mean_gaussian = dist_param.pop("mean")
            dispersion_gaussian = dist_param.pop("dispersion")
            dist_param["loc"] = mean_gaussian
            dist_param["scale"] = dispersion_gaussian
        dist = getattr(distributions_module, dist_param.pop("distribution"))
        return dist(**dist_param)

    @classmethod
    def get_theory_model(
        cls,
        theory_config: Dict,
        statistics: List[str],
    ) -> "Summary":
        """Get theory model

        Args:
            theory_config (Dict): configuration for theory model, both module and class

        Returns:
            Summary: summary to fit
        """
        module = theory_config.pop("module")
        class_name = theory_config.pop("class")
        if "args" in theory_config:
            return getattr(importlib.import_module(module), class_name)(
                summaries=statistics,
                **theory_config.get("args", None),
            )
        else:
            return getattr(importlib.import_module(module), class_name)(
                summaries=statistics,
            )

    @abstractmethod
    def __call__(
        self,
    ):
        pass

    def invert_covariance(
        self,
        covariance_matrix: np.array,
    ) -> np.array:
        """invert covariance matrix

        Args:
            covariance_matrix (np.array): covariance matrix to invert

        Returns:
            np.array: inverse covariance
        """
        return np.linalg.inv(covariance_matrix)

    def get_loglikelihood_for_prediction(
        self,
        prediction: np.array,
        predicted_uncertainty: np.array,
    ) -> float:
        """Get gaussian loglikelihood for prediction

        Args:
            prediction (np.array): model prediction

        Returns:
            float: log likelihood
        """
        diff = prediction - self.observation
        if not self.add_predicted_uncertainty:
            return -0.5 * diff @ self.inverse_covariance_matrix @ diff
        covariance_matrix = self.covariance_matrix + np.diag(predicted_uncertainty**2)
        inverse_covariance_matrix = self.invert_covariance(covariance_matrix)
        return -0.5 * diff @ inverse_covariance_matrix @ diff

    def get_loglikelihood_for_prediction_vectorized(
        self,
        prediction: np.array,
        predicted_uncertainty: np.array,
    ) -> np.array:
        """Get vectorized loglikelihood prediction

        Args:
            prediction (np.array): prediciton in batches

        Returns:
            np.array: array of likelihoods
        """
        diff = prediction - self.observation
        if not self.add_predicted_uncertainty:
            right = np.einsum("ik,...k", self.inverse_covariance_matrix, diff)
            return -0.5 * np.einsum("ki,ji", diff, right)[:, 0]
        covariance_matrix = self.covariance_matrix + np.diag(predicted_uncertainty**2)
        inverse_covariance_matrix = self.invert_covariance(covariance_matrix)
        right = np.einsum("ik,...k", inverse_covariance_matrix, diff)
        return -0.5 * np.einsum("ki,ji", diff, right)[:, 0]



    def sample_parameters_from_prior(
        self,
    ):
        params = {}
        for param, dist in self.priors.items():
            params[param] = dist.rvs()
        for p, v in self.fixed_parameters.items():
            params[p] = v
        return params

    def sample_from_prior(
        self,
    ) -> Tuple:
        """Sample predictions from prior

        Returns:
            Tuple: tuple of parameters and theory model predictions
        """
        params = self.sample_parameters_from_prior()
        return params, self.theory_model(
            params, select_filters=self.select_filters, slice_filters=self.slice_filters
        )

    def get_model_prediction(
        self,
        parameters: np.array,
    ) -> np.array:
        """Get model prediction for a given set of input parameters

        Args:
            parameters (np.array): input parameters

        Returns:
            np.array: model prediction
        """
        params = dict(zip(list(self.priors.keys()), parameters))
        for i, fixed_param in enumerate(self.fixed_parameters.keys()):
            params[fixed_param] = self.fixed_parameters[fixed_param]
        return self.theory_model(
            params,
            select_filters=self.select_filters,
            slice_filters=self.slice_filters,
        )

    def get_model_prediction_vectorized(
        self,
        parameters: np.array,
    ) -> np.array:
        """get vectorized model predictions

        Args:
            parameters (np.array): input parameters

        Returns:
            np.array: model predictions in batches
        """
        params = {}
        for i, param in enumerate(self.priors.keys()):
            params[param] = parameters[:, i]
        for i, fixed_param in enumerate(self.fixed_parameters.keys()):
            params[fixed_param] = self.fixed_parameters[fixed_param] * np.ones(
                len(parameters)
            )
        out = self.theory_model.get_for_batch(
            params,
            s_min=self.s_min,
        )
        return out.reshape((len(parameters), -1))
