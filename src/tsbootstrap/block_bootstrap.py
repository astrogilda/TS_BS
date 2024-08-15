from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import Optional, Union

import numpy as np
from pydantic import Field

from tsbootstrap.base_bootstrap import BaseTimeSeriesBootstrap
from tsbootstrap.block_bootstrap_configs import (
    BartlettsBootstrapConfig,
    BaseBlockBootstrapConfig,
    BlackmanBootstrapConfig,
    CircularBlockBootstrapConfig,
    HammingBootstrapConfig,
    HanningBootstrapConfig,
    MovingBlockBootstrapConfig,
    NonOverlappingBlockBootstrapConfig,
    StationaryBlockBootstrapConfig,
    TukeyBootstrapConfig,
)
from tsbootstrap.block_generator import BlockGenerator
from tsbootstrap.block_length_sampler import BlockLengthSampler
from tsbootstrap.block_resampler import BlockResampler
from tsbootstrap.utils.types import DistributionTypes


class BlockBootstrap(BaseTimeSeriesBootstrap):
    """
    Block Bootstrap base class for time series data.

    Attributes
    ----------
    block_length : Optional[int]
        The length of the blocks to sample. If None, the block length is automatically set to the square root of the number of observations.
    block_length_distribution : Optional[str]
        The block length distribution function to use. If None, the block length distribution is not utilized.
    wrap_around_flag : bool
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool
        Whether to combine the block generation and sampling steps.
    block_weights : Optional[Union[np.ndarray, Callable]]
        The weights to use when sampling blocks.
    tapered_weights : Optional[Callable]
        The tapered weights to use when sampling blocks.
    overlap_length : Optional[int]
        The length of the overlap between blocks.
    min_block_length : Optional[int]
        The minimum length of the blocks.
    blocks : Optional[list[np.ndarray]]
        The generated blocks. Initialized as None.
    block_resampler : Optional[BlockResampler]
        The block resampler object. Initialized as None.

    Notes
    -----
    This class uses Pydantic for data validation. The `block_length`, `overlap_length`,
    and `min_block_length` fields must be greater than or equal to 1 if provided.

    The `blocks` and `block_resampler` attributes are not included in the initialization
    and are set during the bootstrap process.

    Raises
    ------
    ValueError
        If validation fails for any of the fields, e.g., if block_length is less than 1.
    """

    _tags = {"bootstrap_type": "block"}

    block_length: Optional[Integral] = Field(default=None, ge=1)
    block_length_distribution: Optional[DistributionTypes] = Field(
        default=None
    )
    wrap_around_flag: bool = Field(default=False)
    overlap_flag: bool = Field(default=False)
    combine_generation_and_sampling_flag: bool = Field(default=False)
    block_weights: Optional[Union[np.ndarray, Callable]] = Field(default=None)
    tapered_weights: Optional[Callable] = Field(default=None)
    overlap_length: Optional[Integral] = Field(default=None, ge=1)
    min_block_length: Optional[Integral] = Field(default=None, ge=1)

    blocks: Optional[list[np.ndarray]] = Field(default=None, init=False)
    block_resampler: Optional[BlockResampler] = Field(default=None, init=False)

    def _check_input_bb(self, X: np.ndarray, enforce_univariate=True) -> None:
        # type: ignore
        if self.block_length is not None and self.block_length > X.shape[0]:  # type: ignore
            raise ValueError(
                "block_length cannot be greater than the size of the input array X."
            )

    def _generate_blocks(self, X: np.ndarray) -> list[np.ndarray]:
        """Generates blocks of indices.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_features)
            The input samples.

        Returns
        -------
        blocks : list of arrays
            The generated blocks.

        """
        self._check_input_bb(X)
        block_length_sampler = BlockLengthSampler(
            avg_block_length=(
                self.block_length
                if self.block_length is not None
                else int(np.sqrt(X.shape[0]))
            ),  # type: ignore
            block_length_distribution=self.block_length_distribution,
            rng=self.rng,
        )

        block_generator = BlockGenerator(
            block_length_sampler=block_length_sampler,
            input_length=X.shape[0],  # type: ignore
            rng=self.rng,
            wrap_around_flag=self.wrap_around_flag,
            overlap_length=self.overlap_length,
            min_block_length=self.min_block_length,
        )

        blocks = block_generator.generate_blocks(
            overlap_flag=self.overlap_flag
        )

        return blocks

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y=None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generate a single bootstrap sample.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_features)
            The input samples.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing the indices and data of the generated blocks.
        """
        if self.combine_generation_and_sampling_flag or self.blocks is None:
            blocks = self._generate_blocks(X=X)

            block_resampler = BlockResampler(
                X=X,
                blocks=blocks,
                rng=self.rng,
                block_weights=self.block_weights,
                tapered_weights=self.tapered_weights,
            )
        else:
            blocks = self.blocks
            block_resampler = self.block_resampler

        (
            block_indices,
            block_data,
        ) = block_resampler.resample_block_indices_and_data()  # type: ignore

        if not self.combine_generation_and_sampling_flag:
            self.blocks = blocks
            self.block_resampler = block_resampler

        return block_indices, block_data


class BaseBlockBootstrap(BlockBootstrap):
    """
    Base class for block bootstrapping.

    Parameters
    ----------
    bootstrap_type : str, default="moving"
        The type of block bootstrap to use.
        Must be one of "nonoverlapping", "moving", "stationary", or "circular".
    kwargs
        Additional keyword arguments to pass to the BaseBlockBootstrapConfig class.
        See the documentation for BaseBlockBootstrapConfig for more information.
    """

    def __init__(
        self,
        bootstrap_type: str = "moving",
        **kwargs,
    ):
        # def __init__(
        #     self,
        #     n_bootstraps: Integral = 10,  # type: ignore
        #     block_length: Integral = None,
        #     block_length_distribution: str = None,
        #     wrap_around_flag: bool = False,
        #     overlap_flag: bool = False,
        #     combine_generation_and_sampling_flag: bool = False,
        #     block_weights=None,
        #     tapered_weights: Callable = None,
        #     overlap_length: Integral = None,
        #     min_block_length: Integral = None,
        #     rng=None,
        #     bootstrap_type: str = None,
        #     **kwargs,
        # ):
        self.bootstrap_type = bootstrap_type

        if hasattr(self, "config"):
            config = self.config
        else:
            config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                **kwargs,
            )
            # config = BaseBlockBootstrapConfig(
            #     n_bootstraps=n_bootstraps,
            #     block_length=block_length,
            #     block_length_distribution=block_length_distribution,
            #     wrap_around_flag=wrap_around_flag,
            #     overlap_flag=overlap_flag,
            #     combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            #     block_weights=block_weights,
            #     tapered_weights=tapered_weights,
            #     overlap_length=overlap_length,
            #     min_block_length=min_block_length,
            #     rng=rng,
            #     bootstrap_type=bootstrap_type,
            # )
            self.config = config

        super().__init__(
            # n_bootstraps=n_bootstraps,
            # block_length=block_length,
            # block_length_distribution=block_length_distribution,
            # wrap_around_flag=wrap_around_flag,
            # overlap_flag=overlap_flag,
            # combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            # block_weights=block_weights,
            # tapered_weights=tapered_weights,
            # overlap_length=overlap_length,
            # min_block_length=min_block_length,
            # rng=rng,
            **kwargs,
        )

        self.bootstrap_instance: Optional[BlockBootstrap] = None

        if config.bootstrap_type:
            bcls = BLOCK_BOOTSTRAP_TYPES_DICT[config.bootstrap_type]
            # self_params = self.get_params()
            # if "bootstrap_type" in self_params:
            #    self_params.pop("bootstrap_type")
            # bcls_params = bcls.get_param_names()
            # bcls_kwargs = {k: v for k, v in self_params.items() if k in bcls_params}
            # self.bootstrap_instance = bcls(**self_params)
            self.bootstrap_instance = bcls(**kwargs)

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        """
        Generate a single bootstrap sample using either the base BlockBootstrap method or the specified bootstrap_type.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_features)
            The input samples.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing the indices and data of the generated blocks.
        """
        if self.bootstrap_instance is None:
            # Generate samples using the base BlockBootstrap method
            (
                block_indices,
                block_data,
            ) = super()._generate_samples_single_bootstrap(X=X, y=y)
        else:
            # Generate samples using the specified bootstrap_type
            if hasattr(
                self.bootstrap_instance, "_generate_samples_single_bootstrap"
            ):
                (
                    block_indices,
                    block_data,
                ) = self.bootstrap_instance._generate_samples_single_bootstrap(
                    X=X, y=y
                )
            else:
                raise NotImplementedError(
                    f"The bootstrap class '{type(self.bootstrap_instance).__name__}' does not implement '_generate_samples_single_bootstrap' method."
                )

        return block_indices, block_data


class MovingBlockBootstrap(BlockBootstrap):
    r"""
    Moving Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to True, meaning that blocks can overlap.
    * `wrap_around_flag` is always set to False, meaning that the data will not
    wrap around when generating blocks.
    * `block_length_distribution` is always None, meaning that the block length
    distribution is not utilized.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Moving Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + \\lfloor U_i \\rfloor}

    where :math:`L` is the block length, :math:`U_i` is a uniform random variable on :math:`[0, 1]`, and :math:`\\lfloor \\cdot \\rfloor` is the floor function.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Optional[Integral] = None,
        block_length_distribution: Optional[str] = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Optional[Callable] = None,
        overlap_length: Optional[Integral] = None,
        min_block_length: Optional[Integral] = None,
        rng=None,
        **kwargs,
    ):
        self.config = MovingBlockBootstrapConfig(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )
        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
            **kwargs,
        )


class StationaryBlockBootstrap(BlockBootstrap):
    r"""
    Stationary Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to True, meaning that blocks can overlap.
    * `wrap_around_flag` is always set to False, meaning that the data will not
    wrap around when generating blocks.
    * `block_length_distribution` is always "geometric", meaning that the block
    length distribution is geometrically distributed.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Stationary Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + \\lfloor U_i \\rfloor}

    where :math:`L` is the block length, :math:`U_i` is a uniform random variable on :math:`[0, 1]`, and :math:`\\lfloor \\cdot \\rfloor` is the floor function.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Optional[Integral] = None,
        block_length_distribution: Optional[str] = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Optional[Callable] = None,
        overlap_length: Optional[Integral] = None,
        min_block_length: Optional[Integral] = None,
        rng=None,
        **kwargs,
    ):
        self.config = StationaryBlockBootstrapConfig(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
            **kwargs,
        )


class CircularBlockBootstrap(BlockBootstrap):
    r"""
    Circular Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to True, meaning that blocks can overlap.
    * `wrap_around_flag` is always set to True, meaning that the data will wrap
    around when generating blocks.
    * `block_length_distribution` is always None, meaning that the block length
    distribution is not utilized.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Circular Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + \\lfloor U_i \\rfloor}

    where :math:`L` is the block length, :math:`U_i` is a uniform random variable on :math:`[0, 1]`, and :math:`\\lfloor \\cdot \\rfloor` is the floor function.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Optional[Integral] = None,
        block_length_distribution: Optional[str] = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Optional[Callable] = None,
        overlap_length: Optional[Integral] = None,
        min_block_length: Optional[Integral] = None,
        rng=None,
        **kwargs,
    ):
        self.config = CircularBlockBootstrapConfig(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
            **kwargs,
        )


class NonOverlappingBlockBootstrap(BlockBootstrap):
    r"""
    Non-Overlapping Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to False, meaning that blocks cannot overlap.
    * `wrap_around_flag` is always set to False, meaning that the data will not
    wrap around when generating blocks.
    * `block_length_distribution` is always None, meaning that the block length
    distribution is not utilized.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Raises
    ------
    ValueError
        If block_length is not greater than 0.

    Notes
    -----
    The Non-Overlapping Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + i}

    where :math:`L` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Optional[Integral] = None,
        block_length_distribution: Optional[str] = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Optional[Callable] = None,
        overlap_length: Optional[Integral] = None,
        min_block_length: Optional[Integral] = None,
        rng=None,
        **kwargs,
    ):
        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
            **kwargs,
        )
        self.config = NonOverlappingBlockBootstrapConfig(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )


# Be cautious when using the default windowing functions from numpy, as they drop to 0 at the edges.This could be particularly problematic for smaller block_lengths. In the current implementation, we have clipped the min to 0.1, in block_resampler.py.


class BartlettsBootstrap(BaseBlockBootstrap):
    r"""Bartlett's Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Bartlett's window for tapered weights.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Bartlett window is defined as:

    .. math::
        w(n) = 1 - \\frac{|n - (N - 1) / 2|}{(N - 1) / 2}

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Window_function#Triangular_window
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Optional[Integral] = None,
        block_length_distribution: Optional[str] = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Optional[Callable] = None,
        overlap_length: Optional[Integral] = None,
        min_block_length: Optional[Integral] = None,
        bootstrap_type: str = "moving",
        rng=None,
        **kwargs,
    ):
        self.config = BartlettsBootstrapConfig(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            bootstrap_type=bootstrap_type,
            min_block_length=min_block_length,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            bootstrap_type=bootstrap_type,
            rng=rng,
            **kwargs,
        )


class HammingBootstrap(BaseBlockBootstrap):
    r"""
    Hamming Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Hamming window for tapered weights.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Hamming window is defined as:

    .. math::
        w(n) = 0.54 - 0.46 \\cos\\left(\\frac{2\\pi n}{N - 1}\\right)

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Optional[Integral] = None,
        block_length_distribution: Optional[str] = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Optional[Callable] = None,
        overlap_length: Optional[Integral] = None,
        min_block_length: Optional[Integral] = None,
        bootstrap_type: str = "moving",
        rng=None,
        **kwargs,
    ):
        self.config = HammingBootstrapConfig(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            bootstrap_type=bootstrap_type,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            bootstrap_type=bootstrap_type,
            rng=rng,
            **kwargs,
        )


class HanningBootstrap(BaseBlockBootstrap):
    r"""
    Hanning Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Hanning window for tapered weights.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    bootstrap_type : str, default="moving"
        The type of block bootstrap to use.
        Must be one of "nonoverlapping", "moving", "stationary", or "circular".
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Hanning window is defined as:

    .. math::
        w(n) = 0.5 - 0.5 \\cos\\left(\\frac{2\\pi n}{N - 1}\\right)

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Optional[Integral] = None,
        block_length_distribution: Optional[str] = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Optional[Callable] = None,
        overlap_length: Optional[Integral] = None,
        min_block_length: Optional[Integral] = None,
        bootstrap_type: str = "moving",
        rng=None,
        **kwargs,
    ):
        self.config = HanningBootstrapConfig(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            bootstrap_type=bootstrap_type,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            bootstrap_type=bootstrap_type,
            rng=rng,
            **kwargs,
        )


class BlackmanBootstrap(BaseBlockBootstrap):
    r"""
    Blackman Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Blackman window for tapered weights.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Blackman window is defined as:

    .. math::
        w(n) = 0.42 - 0.5 \\cos\\left(\\frac{2\\pi n}{N - 1}\\right) + 0.08 \\cos\\left(\\frac{4\\pi n}{N - 1}\\right)

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Window_function#Blackman_window
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Optional[Integral] = None,
        block_length_distribution: Optional[str] = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Optional[Callable] = None,
        overlap_length: Optional[Integral] = None,
        min_block_length: Optional[Integral] = None,
        bootstrap_type: str = "moving",
        rng=None,
        **kwargs,
    ):
        self.config = BlackmanBootstrapConfig(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            bootstrap_type=bootstrap_type,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            bootstrap_type=bootstrap_type,
            rng=rng,
            **kwargs,
        )


class TukeyBootstrap(BaseBlockBootstrap):
    r"""
    Tukey Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Tukey window for tapered weights.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Tukey window is defined as:

    .. math::
        w(n) = \\begin{cases}
            0.5\\left[1 + \\cos\\left(\\frac{2\\pi n}{\\alpha(N - 1)}\\right)\\right], & \\text{if } n < \\frac{\\alpha(N - 1)}{2}\\\\
            1, & \\text{if } \\frac{\\alpha(N - 1)}{2} \\leq n \\leq (N - 1)\\left(1 - \\frac{\\alpha}{2}\\right)\\\\
            0.5\\left[1 + \\cos\\left(\\frac{2\\pi n}{\\alpha(N - 1)}\\right)\\right], & \\text{if } n > (N - 1)\\left(1 - \\frac{\\alpha}{2}\\right)
        \\end{cases}

    where :math:`N` is the block length and :math:`\\alpha` is the parameter
    controlling the shape of the window.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Window_function#Tukey_window
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Optional[Integral] = None,
        block_length_distribution: Optional[str] = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Optional[Callable] = None,
        overlap_length: Optional[Integral] = None,
        min_block_length: Optional[Integral] = None,
        bootstrap_type: str = "moving",
        rng=None,
        **kwargs,
    ):
        self.config = TukeyBootstrapConfig(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            bootstrap_type=bootstrap_type,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            bootstrap_type=bootstrap_type,
            rng=rng,
            **kwargs,
        )


BLOCK_BOOTSTRAP_TYPES_DICT = {
    "nonoverlapping": NonOverlappingBlockBootstrap,
    "moving": MovingBlockBootstrap,
    "stationary": StationaryBlockBootstrap,
    "circular": CircularBlockBootstrap,
}
