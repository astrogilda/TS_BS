from __future__ import annotations

import logging
import warnings
from typing import Callable, Optional, Union

import numpy as np
from numpy.random import Generator, default_rng
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
)

from tsbootstrap.utils.types import RngTypes
from tsbootstrap.utils.validate import (
    validate_block_indices,
    validate_rng,
)

# Initialize logger
logger = logging.getLogger(__name__)


class BlockResampler(BaseModel):
    """
    Performs block resampling on time series data.

    Parameters
    ----------
    blocks : list[np.ndarray]
        A list of numpy arrays where each array contains indices representing a block in the input data.
    X : np.ndarray
        The input data array to be resampled. Can be a 1D or 2D numpy array.
    block_weights : Optional[Union[Callable[[int], np.ndarray], np.ndarray]]
        An array of weights with length equal to `input_length` or a callable function to generate such weights.
        If None, default uniform weights are used.
    tapered_weights : Optional[Union[Callable[[int], list[np.ndarray]], list[np.ndarray]]]
        A list of weight arrays to apply to the data within the blocks or a callable to generate them.
        Each array corresponds to a block. If None, default uniform weights are used.
    rng : RngTypes
        Random number generator for reproducibility. Can be a numpy Generator or an integer seed.

    Attributes
    ----------
    _block_weights_normalized : Optional[np.ndarray]
        Private attribute to store normalized block_weights.
    _tapered_weights_normalized : Optional[list[np.ndarray]]
        Private attribute to store normalized tapered_weights.
    """

    # Configuration for Pydantic 2.0
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allows custom types like np.ndarray and Callable
        validate_assignment=True,  # Validates fields on assignment
    )

    # Define class attributes with Pydantic Fields
    blocks: list[np.ndarray] = Field(
        ..., description="list of numpy arrays representing block indices."
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
        Union[Callable[[int], list[np.ndarray]], list[np.ndarray]]
    ] = Field(
        default=None,
        description=(
            "A list of weight arrays to apply to the data within the blocks or a callable to generate them. "
            "Each array corresponds to a block. If None, default uniform weights are used."
        ),
    )
    rng: RngTypes = Field(  # type: ignore
        default_factory=lambda: default_rng(),
        description="Random number generator for reproducibility.",
    )

    # Private attributes for normalized weights
    _block_weights_normalized: Optional[np.ndarray] = PrivateAttr(default=None)
    _tapered_weights_normalized: Optional[list[np.ndarray]] = PrivateAttr(
        default=None
    )

    @field_validator("blocks", mode="before")
    @classmethod
    def validate_blocks(
        cls, v: list[np.ndarray], info: ValidationInfo
    ) -> list[np.ndarray]:
        """
        Validate the 'blocks' field before assignment.

        Parameters
        ----------
        v : list[np.ndarray]
            The blocks to validate.
        info : ValidationInfo
            Provides context about the validation, including other fields.

        Returns
        -------
        list[np.ndarray]
            The validated blocks.

        Raises
        ------
        TypeError
            If 'blocks' is not a list of numpy arrays.
        ValueError
            If 'blocks' list is empty or contains invalid data.
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
        Validate the 'X' field before assignment.

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
            If 'X' is not a numpy array.
        ValueError
            If 'X' has insufficient dimensions or invalid shape.
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
        Validate the 'block_weights' field after assignment.

        Parameters
        ----------
        v : Optional[Union[Callable[[int], np.ndarray], np.ndarray]]
            The block weights to validate.
        info : ValidationInfo
            Provides context about the validation, including other fields.

        Returns
        -------
        Optional[Union[Callable[[int], np.ndarray], np.ndarray]]
            The validated block_weights.

        Raises
        ------
        TypeError
            If 'block_weights' is neither a callable nor a numpy array.
        ValueError
            If 'block_weights' has invalid length or contains negative values.
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
        v: Optional[
            Union[Callable[[int], list[np.ndarray]], list[np.ndarray]]
        ],
        info: ValidationInfo,
    ) -> Optional[Union[Callable[[int], list[np.ndarray]], list[np.ndarray]]]:
        """
        Validate the 'tapered_weights' field after assignment.

        Parameters
        ----------
        v : Optional[Union[Callable[[int], list[np.ndarray]], list[np.ndarray]]]
            The tapered weights to validate.
        info : ValidationInfo
            Provides context about the validation, including other fields.

        Returns
        -------
        Optional[Union[Callable[[int], list[np.ndarray]], list[np.ndarray]]]
            The validated tapered_weights.

        Raises
        ------
        TypeError
            If 'tapered_weights' is neither a callable nor a list of numpy arrays.
        ValueError
            If 'tapered_weights' has invalid length or contains negative values.
        """
        blocks: list[np.ndarray] = info.data.get("blocks", [])
        if v is None:
            logger.debug(
                "No 'tapered_weights' provided. Using uniform weights."
            )
            return None
        if isinstance(v, list):
            if len(v) != len(blocks):
                error_msg = f"'tapered_weights' list length ({len(
                    v)}) must match the number of blocks ({len(blocks)})."
                logger.error(error_msg)
                raise ValueError(error_msg)
            for i, weights in enumerate(v):
                if not isinstance(weights, np.ndarray):
                    error_msg = f"'tapered_weights[{
                        i}]' must be a numpy array."
                    logger.error(error_msg)
                    raise TypeError(error_msg)
                if len(weights) != len(blocks[i]):
                    error_msg = f"Length of 'tapered_weights[{i}]' ({len(
                        weights)}) must match the length of block {i} ({len(blocks[i])})."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if not np.all(weights >= 0):
                    error_msg = f"'tapered_weights[{
                        i}]' must contain non-negative values."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            logger.debug(
                "'tapered_weights' validated as list of numpy arrays."
            )
        elif callable(v):
            tapered_weights_list = v(len(blocks))
            if not isinstance(tapered_weights_list, list):
                error_msg = "'tapered_weights' callable must return a list of numpy arrays."
                logger.error(error_msg)
                raise TypeError(error_msg)
            if len(tapered_weights_list) != len(blocks):
                error_msg = f"'tapered_weights' callable must return a list of length {
                    len(blocks)}."
                logger.error(error_msg)
                raise ValueError(error_msg)
            for i, weights in enumerate(tapered_weights_list):
                if not isinstance(weights, np.ndarray):
                    error_msg = f"'tapered_weights[{
                        i}]' must be a numpy array."
                    logger.error(error_msg)
                    raise TypeError(error_msg)
                if len(weights) != len(blocks[i]):
                    error_msg = f"Length of 'tapered_weights[{i}]' ({len(
                        weights)}) must match the length of block {i} ({len(blocks[i])})."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if not np.all(weights >= 0):
                    error_msg = f"'tapered_weights[{
                        i}]' must contain non-negative values."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            logger.debug(
                "'tapered_weights' validated as list of numpy arrays from callable."
            )
        else:
            error_msg = "'tapered_weights' must be a callable or a list of numpy arrays."
            logger.error(error_msg)
            raise TypeError(error_msg)
        return v

    @field_validator("rng", mode="before")
    @classmethod
    def validate_rng_field(
        cls, v: RngTypes, info: ValidationInfo
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
        blocks: list[np.ndarray] = self.blocks
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

        This method processes the `block_weights` attribute by normalizing it so that the sum equals 1.
        It handles cases where `block_weights` is provided as a callable, a numpy array, or left as None.

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
            # Default to uniform weights if block_weights is not provided
            weights = np.full(len(self.X), 1.0)
            logger.debug("Using uniform block_weights.")
        elif isinstance(self.block_weights, np.ndarray):
            # Use the provided numpy array as block_weights
            weights = self.block_weights
            logger.debug("Using provided block_weights as numpy array.")
        elif callable(self.block_weights):
            # Generate block_weights using the provided callable
            weights = self.block_weights(len(self.X))
            if not isinstance(weights, np.ndarray):
                error_msg = (
                    "'block_weights' callable must return a numpy array."
                )
                logger.error(error_msg)
                # Changed from TypeError to ValueError
                raise ValueError(error_msg)
            if weights.shape[0] != len(self.X):
                error_msg = f"'block_weights' callable must return an array of length {
                    len(self.X)}."
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.debug("Using block_weights generated by callable.")
        else:
            # Invalid type for block_weights
            error_msg = (
                "'block_weights' must be a callable or a numpy array or None."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)  # Changed from TypeError to ValueError

        # Normalize weights so that their sum equals 1
        total_weight = weights.sum()
        if total_weight == 0:
            error_msg = "'block_weights' sum must be greater than zero."
            logger.error(error_msg)
            raise ValueError(error_msg)
        normalized_weights = weights / total_weight
        logger.debug(f"'block_weights' normalized: {normalized_weights}")
        return normalized_weights

    def _prepare_tapered_weights_normalized(self) -> list[np.ndarray]:
        """
        Prepare the normalized tapered_weights array.

        This method processes the `tapered_weights` attribute by normalizing each array within
        the list so that the sum of weights for each block equals 1. It handles cases where
        `tapered_weights` is provided as a callable, a list of numpy arrays, or left as None.

        Returns
        -------
        list[np.ndarray]
            Normalized tapered_weights for each block.

        Raises
        ------
        TypeError
            If tapered_weights is neither a callable nor a list of numpy arrays.
        ValueError
            If tapered_weights cannot be normalized.
        """
        if self.tapered_weights is None:
            # Default to uniform weights for each block if tapered_weights is not provided
            tapered_weights_list = [
                np.ones(len(block)) for block in self.blocks
            ]
            logger.debug("Using uniform tapered_weights for each block.")
        elif isinstance(self.tapered_weights, list):
            # Use the provided list of numpy arrays as tapered_weights
            tapered_weights_list = self.tapered_weights
            logger.debug(
                "Using provided tapered_weights as list of numpy arrays."
            )
        elif callable(self.tapered_weights):
            # Generate tapered_weights using the provided callable
            tapered_weights_list = self.tapered_weights(len(self.blocks))
            if not isinstance(tapered_weights_list, list):
                error_msg = "'tapered_weights' callable must return a list of numpy arrays."
                logger.error(error_msg)
                raise ValueError(error_msg)
            if len(tapered_weights_list) != len(self.blocks):
                error_msg = f"'tapered_weights' callable must return a list of length {
                    len(self.blocks)}."
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.debug("Using tapered_weights generated by callable.")
        else:
            # Invalid type for tapered_weights
            error_msg = "'tapered_weights' must be a callable or a list of numpy arrays."
            logger.error(error_msg)
            raise ValueError(error_msg)  # Changed from TypeError to ValueError

        # Validate and normalize each tapered_weights array
        normalized_tapered_weights = []
        for i, weights in enumerate(tapered_weights_list):
            if not isinstance(weights, np.ndarray):
                error_msg = f"'tapered_weights[{i}]' must be a numpy array."
                logger.error(error_msg)
                raise TypeError(error_msg)
            if len(weights) != len(self.blocks[i]):
                error_msg = f"Length of 'tapered_weights[{i}]' ({len(
                    weights)}) must match the length of block {i} ({len(self.blocks[i])})."
                logger.error(error_msg)
                raise ValueError(error_msg)
            if not np.all(weights >= 0):
                error_msg = f"'tapered_weights[{
                    i}]' must contain non-negative values."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Normalize weights within the block so that they sum to 1
            total_taper_weight = weights.sum()
            if total_taper_weight == 0:
                error_msg = f"Sum of 'tapered_weights[{
                    i}]' must be greater than zero."
                logger.error(error_msg)
                raise ValueError(error_msg)
            normalized_weights = weights / total_taper_weight
            normalized_tapered_weights.append(normalized_weights)
            logger.debug(f"'tapered_weights[{i}]' normalized.")

        return normalized_tapered_weights

    @computed_field
    def block_weights_normalized(self) -> Optional[np.ndarray]:
        """
        Computed field to expose normalized block_weights.

        Returns
        -------
        Optional[np.ndarray]
            The normalized block_weights array.
        """
        return self._block_weights_normalized

    @computed_field
    def tapered_weights_normalized(self) -> Optional[list[np.ndarray]]:
        """
        Computed field to expose normalized tapered_weights.

        Returns
        -------
        Optional[list[np.ndarray]]
            The list of normalized tapered_weights arrays.
        """
        return self._tapered_weights_normalized

    def resample_blocks(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Resample blocks and their corresponding tapered_weights with replacement to create a new list of blocks and tapered_weights.

        The resampling process continues until the total length of the resampled blocks equals the length of the input data array 'X'.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            A tuple containing the newly generated list of blocks and their corresponding tapered_weights with total length equal to 'X'.

        Raises
        ------
        ValueError
            If the resampling process cannot cover the entire input data array 'X' due to weight constraints.
        """
        n = self.X.shape[0]
        logger.debug(f"Starting resampling process to cover {n} elements.")
        new_blocks: list[np.ndarray] = []
        new_tapered_weights: list[np.ndarray] = []
        total_samples = 0

        while total_samples < n:
            # Ensure that there are remaining weights to sample from
            if self._block_weights_normalized.sum() == 0:
                error_msg = "Sum of block_weights is zero. Cannot proceed with resampling."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Sample a starting index based on block_weights_normalized probabilities
            sampled_index = self.rng.choice(
                len(self.X), p=self._block_weights_normalized
            )
            logger.debug(f"Sampled starting index: {sampled_index}")

            # Identify blocks that start with the sampled index
            matching_blocks = [
                i
                for i, block in enumerate(self.blocks)
                if block[0] == sampled_index
            ]
            if not matching_blocks:
                # If no block starts at the sampled index, log a warning and skip
                logger.warning(
                    f"No block starts at index {sampled_index}. Skipping."
                )
                continue  # Skip to the next iteration

            # Select the first matching block
            block_idx = matching_blocks[0]
            selected_block = self.blocks[block_idx]
            selected_tapered_weight = self._tapered_weights_normalized[
                block_idx
            ]

            logger.debug(
                f"Selected block {block_idx} with length {
                    len(selected_block)}."
            )

            # Determine how much of the block can be added without exceeding 'n'
            remaining = n - total_samples
            block_length = len(selected_block)

            if block_length > remaining:
                # Truncate the block and its weights if it exceeds the remaining length
                logger.debug(
                    f"Truncating block from length {
                        block_length} to {remaining}."
                )
                adjusted_block = selected_block[:remaining]
                adjusted_tapered_weight = selected_tapered_weight[:remaining]
            else:
                # Use the entire block
                adjusted_block = selected_block
                adjusted_tapered_weight = selected_tapered_weight

            # Append the adjusted block and its tapered weights to the new lists
            new_blocks.append(adjusted_block)
            new_tapered_weights.append(adjusted_tapered_weight)
            total_samples += len(adjusted_block)

            logger.debug(
                f"Added block of length {len(adjusted_block)}. Total samples covered: {
                    total_samples}."
            )

            # If we've covered the entire data array, exit the loop
            if total_samples >= n:
                logger.info("Resampling completed successfully.")
                break

        # Validate that the new blocks cover the entire input data array
        validate_block_indices(new_blocks, n)
        logger.info("All resampled blocks validated successfully.")
        return new_blocks, new_tapered_weights

    def resample_block_indices_and_data(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generate block indices and corresponding data for the input data array 'X'.

        This method performs the resampling of blocks and applies the corresponding
        tapered_weights to the data within each block.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            A tuple containing a list of resampled block indices and a list of the corresponding
            data blocks after applying tapered_weights.

        Raises
        ------
        ValueError
            If the resampling process cannot cover the entire input data array 'X'.
        """
        logger.info("Starting resample_block_indices_and_data process.")
        # Resample blocks and get the corresponding tapered_weights
        resampled_blocks, resampled_tapered_weights = self.resample_blocks()
        block_data: list[np.ndarray] = []

        for i, block in enumerate(resampled_blocks):
            taper = resampled_tapered_weights[i]
            if self.X.ndim > 1:
                # Apply tapered weights to each feature dimension
                data_block = self.X[block] * taper[:, np.newaxis]
            else:
                # Apply tapered weights directly
                data_block = self.X[block] * taper
            block_data.append(data_block)
            logger.debug(f"Processed block {i}: shape {data_block.shape}.")

        logger.info(
            "resample_block_indices_and_data process completed successfully."
        )
        return resampled_blocks, block_data

    def __eq__(self, other: object) -> bool:
        """
        Check equality between two BlockResampler instances.

        Parameters
        ----------
        other : object
            The other object to compare against.

        Returns
        -------
        bool
            True if all relevant attributes are equal, False otherwise.
        """
        if not isinstance(other, BlockResampler):
            return False

        # Compare blocks by checking each corresponding pair
        blocks_equal = all(
            np.array_equal(a, b) for a, b in zip(self.blocks, other.blocks)
        )

        # Compare input data arrays
        X_equal = np.array_equal(self.X, other.X)

        # Compare normalized block_weights
        if isinstance(
            self.block_weights_normalized, np.ndarray
        ) and isinstance(other.block_weights_normalized, np.ndarray):
            block_weights_equal = np.array_equal(
                self.block_weights_normalized, other.block_weights_normalized
            )
        else:
            # Handle cases where normalized weights are None or different types
            block_weights_equal = (
                self.block_weights_normalized == other.block_weights_normalized
            )

        # Compare normalized tapered_weights for each block
        tapered_weights_equal = all(
            np.array_equal(a, b)
            for a, b in zip(
                self.tapered_weights_normalized,
                other.tapered_weights_normalized,
            )
        )

        # Compare random number generators
        rng_equal = self.rng == other.rng

        return (
            blocks_equal
            and X_equal
            and block_weights_equal
            and tapered_weights_equal
            and rng_equal
        )
