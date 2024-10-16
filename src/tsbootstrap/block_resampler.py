from __future__ import annotations

import logging
import warnings
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.random import Generator, default_rng
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from tsbootstrap.utils.types import RngTypes
from tsbootstrap.utils.validate import (
    validate_block_indices,
    validate_rng,
)

# Initialize logger for this module using the module's name.
logger = logging.getLogger(__name__)


class BlockResampler(BaseModel):
    """
    Performs block resampling on time series data.

    This class facilitates the resampling of blocks of indices and their corresponding
    tapered weights to generate new blocks that collectively cover the entire input
    data array. It supports both weighted and tapered resampling, ensuring that the
    generated blocks adhere to specified constraints.

    Parameters
    ----------
    blocks : list[np.ndarray]
        A list of numpy arrays where each array represents the indices of a block in the time series.
    X : np.ndarray
        The input data array. Must be a 1D or 2D numpy array with at least two elements.
    block_weights : Optional[Union[Callable[[int], np.ndarray], np.ndarray]], optional
        An array of weights with length equal to `input_length` or a callable function to generate such weights.
        If None, default uniform weights are used.
    tapered_weights : Optional[Union[Callable[[int], list[np.ndarray]], np.ndarray]], optional
        An array of weights to apply to the data within the blocks or a callable to generate them.
        If None, default uniform weights are used.
    rng : Optional[RngTypes], optional
        Random number generator for reproducibility. If None, a new RNG instance is created.

    Examples
    --------
    >>> import numpy as np
    >>> from tsbootstrap.block_resampler import BlockResampler
    >>> blocks = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    >>> X = np.random.rand(6, 1)
    >>> block_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> tapered_weights = np.array([
    ...     [0.3, 0.4, 0.3],
    ...     [0.2, 0.5, 0.3]
    ... ])
    >>> resampler = BlockResampler(
    ...     blocks=blocks,
    ...     X=X,
    ...     block_weights=block_weights,
    ...     tapered_weights=tapered_weights
    ... )
    >>> new_blocks, new_tapered_weights = resampler.resample_blocks()
    >>> print(new_blocks)
    [array([...]), array([...]), ...]
    """

    # Configuration for Pydantic 2.0
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allows custom types like np.ndarray and Callable
        validate_assignment=True,  # Validates fields on assignment
    )

    # Define class attributes with Pydantic Fields
    blocks: List[np.ndarray] = Field(
        ..., description="List of numpy arrays representing block indices."
    )
    X: np.ndarray = Field(
        ..., description="Input data array (1D or 2D numpy array)."
    )
    block_weights: Optional[Union[Callable[[int], np.ndarray], np.ndarray]] = (
        Field(
            default=None,
            description=(
                "An array of weights with length equal to `input_length` or a callable function to generate such weights. "
                "If None, default uniform weights are used."
            ),
        )
    )
    tapered_weights: Optional[
        Union[Callable[[int], List[np.ndarray]], np.ndarray]
    ] = Field(
        default=None,
        description=(
            "An array of weights to apply to the data within the blocks or a callable to generate them. "
            "If None, default uniform weights are used."
        ),
    )
    rng: Optional[RngTypes] = Field(
        default_factory=lambda: default_rng(),
        description="Random number generator for reproducibility.",
    )

    # Private attributes for normalized weights
    _block_weights_normalized: Optional[np.ndarray] = PrivateAttr(default=None)
    _tapered_weights_normalized: Optional[List[np.ndarray]] = PrivateAttr(
        default=None
    )

    @field_validator("blocks", mode="before")
    @classmethod
    def validate_blocks(
        cls, v: List[np.ndarray], info: ValidationInfo
    ) -> List[np.ndarray]:
        """
        Validate the blocks to ensure they are a non-empty list of integer numpy arrays.

        Parameters
        ----------
        v : List[np.ndarray]
            The list of blocks to validate.
        info : ValidationInfo
            Provides context about the validation, including other fields.

        Returns
        -------
        List[np.ndarray]
            The validated list of blocks.

        Raises
        ------
        TypeError
            If blocks is not a list or contains non-numpy array elements.
        ValueError
            If blocks list is empty or contains arrays with invalid indices.
        """
        if not isinstance(v, list):
            error_msg = "'blocks' must be a list of numpy arrays."
            logger.error(error_msg)
            raise TypeError(error_msg)
        if not v:
            error_msg = "'blocks' list cannot be empty."
            logger.error(error_msg)
            raise ValueError(error_msg)
        for i, block in enumerate(v):
            if not isinstance(block, np.ndarray):
                error_msg = f"Block at index {i} must be a numpy array."
                logger.error(error_msg)
                raise TypeError(error_msg)
            if not issubclass(block.dtype.type, np.integer):
                error_msg = f"Block at index {i} must contain integers."
                logger.error(error_msg)
                raise TypeError(error_msg)
        logger.debug(f"Validated {len(v)} blocks.")
        return v

    @field_validator("X", mode="before")
    @classmethod
    def validate_X(cls, v: np.ndarray, info: ValidationInfo) -> np.ndarray:
        """
        Validate the input data array X.

        Parameters
        ----------
        v : np.ndarray
            The input data array to validate.
        info : ValidationInfo
            Provides context about the validation, including other fields.

        Returns
        -------
        np.ndarray
            The validated input data array.

        Raises
        ------
        TypeError
            If X is not a numpy array.
        ValueError
            If X has fewer than two elements or is not 1D or 2D.
        """
        if not isinstance(v, np.ndarray):
            error_msg = "'X' must be a numpy array."
            logger.error(error_msg)
            raise TypeError(error_msg)
        if v.size < 2:
            error_msg = "'X' must have at least two elements."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if v.ndim == 1:
            warnings.warn(
                "Input 'X' is a 1D array. It will be reshaped to a 2D array.",
                stacklevel=2,
            )
            v = v.reshape(-1, 1)
            logger.debug("Reshaped 'X' from 1D to 2D array.")
        elif v.ndim > 2:
            error_msg = "'X' must be a 1D or 2D numpy array."
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"Validated input data array 'X' with shape {v.shape}.")
        return v

    @field_validator("block_weights", mode="after")
    @classmethod
    def validate_block_weights(
        cls,
        v: Optional[Union[Callable[[int], np.ndarray], np.ndarray]],
        info: ValidationInfo,
    ) -> Optional[Union[Callable[[int], np.ndarray], np.ndarray]]:
        """
        Validate the block_weights parameter.

        Parameters
        ----------
        v : Optional[Union[Callable[[int], np.ndarray], np.ndarray]]
            The block_weights to validate.
        info : ValidationInfo
            Provides context about the validation, including other fields.

        Returns
        -------
        Optional[Union[Callable[[int], np.ndarray], np.ndarray]]
            The validated block_weights.

        Raises
        ------
        TypeError
            If block_weights is neither a callable nor a numpy array.
        ValueError
            If block_weights array does not match the input_length.
        """
        X: np.ndarray = info.data.get("X")
        input_length = X.shape[0] if X is not None else None
        if input_length is None:
            error_msg = "'X' must be set before 'block_weights'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if v is None:
            logger.debug("No 'block_weights' provided. Using uniform weights.")
            return None

        if isinstance(v, np.ndarray):
            if v.shape[0] != input_length:
                error_msg = f"'block_weights' array length ({v.shape[0]}) must match 'input_length' ({
                        input_length})."
                logger.error(error_msg)
                raise ValueError(error_msg)
            if not np.all(v >= 0):
                error_msg = "'block_weights' must contain non-negative values."
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.debug("'block_weights' validated as numpy array.")
        elif not callable(v):
            error_msg = "'block_weights' must be a callable or a numpy array."
            logger.error(error_msg)
            raise TypeError(error_msg)
        logger.debug("'block_weights' validated.")
        return v

    @field_validator("tapered_weights", mode="after")
    @classmethod
    def validate_tapered_weights(
        cls,
        v: Optional[Union[Callable[[int], List[np.ndarray]], np.ndarray]],
        info: ValidationInfo,
    ) -> Optional[Union[Callable[[int], List[np.ndarray]], np.ndarray]]:
        """
        Validate the tapered_weights parameter.

        Parameters
        ----------
        v : Optional[Union[Callable[[int], List[np.ndarray]], np.ndarray]]
            The tapered_weights to validate.
        info : ValidationInfo
            Provides context about the validation, including other fields.

        Returns
        -------
        Optional[Union[Callable[[int], List[np.ndarray]], np.ndarray]]
            The validated tapered_weights.

        Raises
        ------
        TypeError
            If tapered_weights is neither a callable nor a numpy array.
        ValueError
            If tapered_weights array does not match the number of blocks.
        """
        blocks: List[np.ndarray] = info.data.get("blocks", [])
        if v is None:
            logger.debug(
                "No 'tapered_weights' provided. Using uniform weights."
            )
            return None
        if isinstance(v, np.ndarray):
            if v.shape[0] != len(blocks):
                error_msg = f"'tapered_weights' array length ({v.shape[0]}) must match the number of blocks ({
                        len(blocks)})."
                logger.error(error_msg)
                raise ValueError(error_msg)
            # Each entry in tapered_weights should be an array matching the block's length
            for i, weights in enumerate(v):
                if not isinstance(weights, np.ndarray):
                    error_msg = f"'tapered_weights[{
                        i}]' must be a numpy array."
                    logger.error(error_msg)
                    raise TypeError(error_msg)
                if len(weights) != len(blocks[i]):
                    error_msg = f"Length of 'tapered_weights[{i}]' ({len(weights)}) must match the length of block {
                            i} ({len(blocks[i])})."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if not np.all(weights >= 0):
                    error_msg = f"'tapered_weights[{
                        i}]' must contain non-negative values."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            logger.debug("'tapered_weights' validated as numpy array.")
        elif not callable(v):
            error_msg = (
                "'tapered_weights' must be a callable or a numpy array."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
        logger.debug("'tapered_weights' validated.")
        return v

    @field_validator("rng", mode="before")
    @classmethod
    def validate_rng_field(
        cls, v: Optional[RngTypes], info: ValidationInfo
    ) -> Generator:
        """
        Validate and set the random number generator.

        Parameters
        ----------
        v : Optional[RngTypes]
            The RNG to validate.
        info : ValidationInfo
            Provides context about the validation, including other fields.

        Returns
        -------
        Generator
            The validated RNG.

        Raises
        ------
        TypeError
            If rng is not a numpy Generator or an integer.
        ValueError
            If rng is an integer but it is not a non-negative integer.
        """
        rng = validate_rng(v, allow_seed=True)
        logger.debug(f"Random number generator set: {rng}")
        return rng

    @model_validator(mode="after")
    def check_consistency(self) -> BlockResampler:
        """
        Perform inter-field validation to ensure consistency among fields.

        This validator runs after all field validators have processed their respective fields,
        ensuring that interdependent fields maintain logical consistency.

        Returns
        -------
        BlockResampler
            The validated BlockResampler instance.

        Raises
        ------
        ValueError
            If any of the consistency checks fail.
        """
        blocks: List[np.ndarray] = self.blocks
        X_shape = self.X.shape[0]

        if not blocks:
            error_msg = "'blocks' list cannot be empty."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Ensure all block indices are within the range of X
        for i, block in enumerate(blocks):
            if np.any(block < 0) or np.any(block >= X_shape):
                error_msg = f"Block indices in block {
                    i} must be within the range of the input data array 'X'."
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Prepare normalized weights
        self._block_weights_normalized = (
            self._prepare_block_weights_normalized()
        )
        self._tapered_weights_normalized = (
            self._prepare_tapered_weights_normalized()
        )

        logger.debug("All inter-field consistency checks passed.")
        return self

    def _prepare_block_weights_normalized(self) -> np.ndarray:
        """
        Prepare the normalized block_weights array.

        Returns
        -------
        np.ndarray
            Normalized block_weights array.

        Raises
        ------
        TypeError
            If block_weights is neither a callable nor a numpy array.
        ValueError
            If block_weights cannot be normalized.
        """
        if self.block_weights is None:
            weights = np.full(len(self.X), 1.0)
            logger.debug("Using uniform block_weights.")
        elif isinstance(self.block_weights, np.ndarray):
            weights = self.block_weights
            logger.debug("Using provided block_weights as numpy array.")
        elif callable(self.block_weights):
            weights = self.block_weights(len(self.X))
            if not isinstance(weights, np.ndarray):
                error_msg = (
                    "'block_weights' callable must return a numpy array."
                )
                logger.error(error_msg)
                raise TypeError(error_msg)
            if weights.shape[0] != len(self.X):
                error_msg = f"'block_weights' callable must return an array of length {
                        len(self.X)}."
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.debug("Using block_weights generated by callable.")
        else:
            error_msg = (
                "'block_weights' must be a callable or a numpy array or None."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Normalize weights
        total_weight = weights.sum()
        if total_weight == 0:
            error_msg = "'block_weights' sum must be greater than zero."
            logger.error(error_msg)
            raise ValueError(error_msg)
        normalized_weights = weights / total_weight
        logger.debug(f"'block_weights' normalized: {normalized_weights}")
        return normalized_weights

    def _prepare_tapered_weights_normalized(self) -> List[np.ndarray]:
        """
        Prepare the normalized tapered_weights array.

        Returns
        -------
        List[np.ndarray]
            Normalized tapered_weights for each block.

        Raises
        ------
        TypeError
            If tapered_weights is neither a callable nor a numpy array.
        ValueError
            If tapered_weights cannot be normalized.
        """
        if self.tapered_weights is None:
            tapered_weights_list = [
                np.ones(len(block)) for block in self.blocks
            ]
            logger.debug("Using uniform tapered_weights for each block.")
        elif isinstance(self.tapered_weights, np.ndarray):
            tapered_weights_list = []
            for i, block in enumerate(self.blocks):
                weights = self.tapered_weights[i]
                if not isinstance(weights, np.ndarray):
                    error_msg = f"'tapered_weights[{
                        i}]' must be a numpy array."
                    logger.error(error_msg)
                    raise TypeError(error_msg)
                if len(weights) != len(block):
                    error_msg = f"Length of 'tapered_weights[{i}]' ({len(weights)}) must match the length of block {
                            i} ({len(block)})."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if not np.all(weights >= 0):
                    error_msg = f"'tapered_weights[{
                        i}]' must contain non-negative values."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                tapered_weights_list.append(weights)
            logger.debug("Using provided tapered_weights as numpy array.")
        elif callable(self.tapered_weights):
            tapered_weights_list = self.tapered_weights(len(self.blocks))
            if not isinstance(tapered_weights_list, list):
                error_msg = "'tapered_weights' callable must return a list of numpy arrays."
                logger.error(error_msg)
                raise TypeError(error_msg)
            if len(tapered_weights_list) != len(self.blocks):
                error_msg = f"'tapered_weights' callable must return a list of length {
                        len(self.blocks)}."
                logger.error(error_msg)
                raise ValueError(error_msg)
            for i, weights in enumerate(tapered_weights_list):
                if not isinstance(weights, np.ndarray):
                    error_msg = f"'tapered_weights[{
                        i}]' must be a numpy array."
                    logger.error(error_msg)
                    raise TypeError(error_msg)
                if len(weights) != len(self.blocks[i]):
                    error_msg = f"Length of 'tapered_weights[{i}]' ({len(weights)}) must match the length of block {
                            i} ({len(self.blocks[i])})."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if not np.all(weights >= 0):
                    error_msg = f"'tapered_weights[{
                        i}]' must contain non-negative values."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            logger.debug("Using tapered_weights generated by callable.")
        else:
            error_msg = "'tapered_weights' must be a callable or a numpy array or None."
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Normalize weights within each block
        normalized_tapered_weights = []
        for i, weights in enumerate(tapered_weights_list):
            total_weight = weights.sum()
            if total_weight == 0:
                error_msg = f"Sum of 'tapered_weights[{
                    i}]' must be greater than zero."
                logger.error(error_msg)
                raise ValueError(error_msg)
            normalized_weights = weights / total_weight
            normalized_tapered_weights.append(normalized_weights)
            logger.debug(f"'tapered_weights[{i}]' normalized.")

        return normalized_tapered_weights

    def resample_blocks(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Resample blocks and their corresponding tapered_weights with replacement to create a new list of blocks and tapered_weights.

        The resampling process continues until the total length of the resampled blocks equals the length of the input data array 'X'.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing the newly generated list of blocks and their corresponding tapered_weights with total length equal to 'X'.

        Raises
        ------
        ValueError
            If the resampling process cannot cover the entire input data array 'X' due to weight constraints.
        """
        n = self.X.shape[0]
        logger.debug(f"Starting resampling process to cover {n} elements.")
        new_blocks: List[np.ndarray] = []
        new_tapered_weights: List[np.ndarray] = []
        total_samples = 0

        while total_samples < n:
            if self._block_weights_normalized.sum() == 0:
                error_msg = "Sum of block_weights is zero. Cannot proceed with resampling."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Sample a starting index based on block_weights_normalized
            sampled_index = self.rng.choice(
                len(self.X), p=self._block_weights_normalized
            )
            logger.debug(f"Sampled starting index: {sampled_index}")

            # Find the block that starts with the sampled_index
            matching_blocks = [
                i
                for i, block in enumerate(self.blocks)
                if block[0] == sampled_index
            ]
            if not matching_blocks:
                logger.warning(
                    f"No block starts at index {
                               sampled_index}. Skipping."
                )
                continue  # Skip if no block starts at the sampled index

            block_idx = matching_blocks[0]
            selected_block = self.blocks[block_idx]
            selected_tapered_weight = self._tapered_weights_normalized[
                block_idx
            ]

            logger.debug(
                f"Selected block {block_idx} with length {
                         len(selected_block)}."
            )

            remaining = n - total_samples
            block_length = len(selected_block)

            if block_length > remaining:
                logger.debug(
                    f"Truncating block from length {
                             block_length} to {remaining}."
                )
                adjusted_block = selected_block[:remaining]
                adjusted_tapered_weight = selected_tapered_weight[:remaining]
            else:
                adjusted_block = selected_block
                adjusted_tapered_weight = selected_tapered_weight

            new_blocks.append(adjusted_block)
            new_tapered_weights.append(adjusted_tapered_weight)
            total_samples += len(adjusted_block)

            logger.debug(
                f"Added block of length {
                         len(adjusted_block)}. Total samples covered: {total_samples}."
            )

            if total_samples >= n:
                logger.info("Resampling completed successfully.")
                break

        # Validate that the new blocks cover the entire input data array
        validate_block_indices(new_blocks, n)
        logger.info("All resampled blocks validated successfully.")
        return new_blocks, new_tapered_weights

    def resample_block_indices_and_data(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate block indices and corresponding data for the input data array 'X'.

        This method performs the resampling of blocks and applies the corresponding
        tapered_weights to the data within each block.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing a list of resampled block indices and a list of the corresponding
            data blocks after applying tapered_weights.

        Raises
        ------
        ValueError
            If the resampling process cannot cover the entire input data array 'X'.
        """
        logger.info("Starting resample_block_indices_and_data process.")
        resampled_blocks, resampled_tapered_weights = self.resample_blocks()
        block_data: List[np.ndarray] = []

        for i, block in enumerate(resampled_blocks):
            taper = resampled_tapered_weights[i]
            if self.X.ndim > 1:
                data_block = self.X[block] * taper[:, np.newaxis]
            else:
                data_block = self.X[block] * taper
            block_data.append(data_block)
            logger.debug(f"Processed block {i}: shape {data_block.shape}.")

        logger.info(
            "resample_block_indices_and_data process completed successfully."
        )
        return resampled_blocks, block_data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockResampler):
            return False
        blocks_equal = all(
            np.array_equal(a, b) for a, b in zip(self.blocks, other.blocks)
        )
        X_equal = np.array_equal(self.X, other.X)
        if isinstance(
            self._block_weights_normalized, np.ndarray
        ) and isinstance(other._block_weights_normalized, np.ndarray):
            block_weights_equal = np.array_equal(
                self._block_weights_normalized, other._block_weights_normalized
            )
        else:
            block_weights_equal = (
                self._block_weights_normalized
                == other._block_weights_normalized
            )
        tapered_weights_equal = all(
            np.array_equal(a, b)
            for a, b in zip(
                self._tapered_weights_normalized,
                other._tapered_weights_normalized,
            )
        )
        rng_equal = self.rng == other.rng
        return (
            blocks_equal
            and X_equal
            and block_weights_equal
            and tapered_weights_equal
            and rng_equal
        )

    # Optional: Remove __repr__ and __str__ if Pydantic's defaults are sufficient
