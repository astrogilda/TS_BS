import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from tsbootstrap import BlockResampler
from tsbootstrap.utils.odds_and_ends import check_generator

# Hypothesis strategy for generating random seeds
rng_strategy = st.integers(0, 10**6)


def block_generator(
    input_length: int,
    wrap_around_flag: bool,
    overlap_length: int,
    min_block_length: int,
    avg_block_length: int,
    overlap_flag: bool,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Generates blocks and an input data array X for testing.

    Parameters
    ----------
    input_length : int
        The total length of the input data array.
    wrap_around_flag : bool
        Whether to allow blocks to wrap around the input array.
    overlap_length : int
        The maximum number of overlapping elements between consecutive blocks.
    min_block_length : int
        The minimum length of a block.
    avg_block_length : int
        The average length of a block.
    overlap_flag : bool
        Whether blocks can overlap.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray]
        A tuple containing a list of block indices and the input data array X.
    """
    from tsbootstrap.block_generator import BlockGenerator, BlockLengthSampler

    # Initialize the block length sampler with the average block length
    block_length_sampler = BlockLengthSampler(
        avg_block_length=avg_block_length
    )

    # Initialize the random number generator
    rng = np.random.default_rng()

    # Initialize the block generator with the provided parameters
    block_generator_instance = BlockGenerator(
        block_length_sampler=block_length_sampler,
        input_length=input_length,
        wrap_around_flag=wrap_around_flag,
        rng=rng,
        overlap_length=overlap_length,
        min_block_length=min_block_length,
    )

    # Generate blocks based on the overlap flag
    blocks = block_generator_instance.generate_blocks(
        overlap_flag=overlap_flag
    )

    # Generate a random input data array X
    X = np.random.uniform(low=0, high=1e6, size=input_length).reshape(-1, 1)

    return blocks, X


# Hypothesis strategy for generating valid blocks and X
valid_block_indices_and_X = st.builds(
    block_generator,
    input_length=st.integers(min_value=50, max_value=100),
    wrap_around_flag=st.booleans(),
    # Adjusted for feasibility
    overlap_length=st.integers(min_value=1, max_value=3),
    # Adjusted for feasibility
    min_block_length=st.integers(min_value=2, max_value=5),
    # Adjusted for feasibility
    avg_block_length=st.integers(min_value=5, max_value=20),
    overlap_flag=st.booleans(),
)


def block_weights_func(size: int) -> np.ndarray:
    """
    Generates random block weights.

    Parameters
    ----------
    size : int
        The size of the block_weights array.

    Returns
    -------
    np.ndarray
        An array of random weights.
    """
    return np.random.uniform(low=0, high=1e6, size=size)


def tapered_weights_callable(
    num_blocks: int, blocks: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Generates tapered weights for each block.

    Parameters
    ----------
    num_blocks : int
        The number of blocks.
    blocks : list[np.ndarray]
        The list of blocks for which tapered_weights are generated.

    Returns
    -------
    list[np.ndarray]
        A list of tapered_weights arrays corresponding to each block.
    """
    return [
        np.random.uniform(low=0, high=1e6, size=len(block)) for block in blocks
    ]


def check_list_of_arrays_equality(
    list1: list[np.ndarray], list2: list[np.ndarray], equal: bool = True
) -> None:
    """
    Check if two lists of NumPy arrays are equal or not equal, based on the `equal` parameter.

    Parameters
    ----------
    list1 : list[np.ndarray]
        The first list of arrays.
    list2 : list[np.ndarray]
        The second list of arrays.
    equal : bool, optional
        If True, asserts that the lists are equal. If False, asserts that they are not all equal.

    Raises
    ------
    AssertionError
        If the assertion fails.
    """
    if equal:
        # Ensure both lists have the same length
        assert len(list1) == len(list2), "Lists are not of the same length"
        # Check each corresponding pair of arrays
        for i, (array1, array2) in enumerate(zip(list1, list2)):
            np.testing.assert_array_equal(
                array1, array2, err_msg=f"Arrays at index {i} are not equal"
            )
    else:
        # If lists have different lengths, they are considered not equal
        if len(list1) != len(list2):
            return
        # Check if at least one pair of arrays is not equal
        mismatch = False
        for _, (array1, array2) in enumerate(zip(list1, list2)):
            try:
                np.testing.assert_array_equal(array1, array2)
            except AssertionError:
                mismatch = True
                break
        assert mismatch, "All arrays are unexpectedly equal"


def unique_first_indices(blocks: list[np.ndarray]) -> list[np.ndarray]:
    """
    Return a list of blocks with unique first indices.

    This helps in minimizing redundancy during testing.

    Parameters
    ----------
    blocks : list[np.ndarray]
        The list of blocks to filter.

    Returns
    -------
    list[np.ndarray]
        A list of blocks with unique first indices.
    """
    seen_first_indices = set()
    unique_blocks = []
    for block in blocks:
        if block[0] not in seen_first_indices:
            unique_blocks.append(block)
            seen_first_indices.add(block[0])
    return unique_blocks


class TestInit:
    """Test the __init__ method of BlockResampler."""

    class TestPassingCases:
        """Test cases where BlockResampler should initialize correctly."""

        @settings(deadline=None)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_init(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test initialization of BlockResampler with various block_weights and tapered_weights.
            """
            blocks, X = block_indices_and_X
            rng = np.random.default_rng(random_seed)

            # Define tapered_weights as either None or a callable returning a list of arrays
            tapered_weights_choice = np.random.choice([0, 1])
            if tapered_weights_choice == 0:
                tapered_weights = None
            else:
                # Define a callable that captures the current blocks via closure
                def tapered_weights_callable_wrapper_func(
                    num_blocks: int,
                ) -> list[np.ndarray]:
                    return tapered_weights_callable(num_blocks, blocks)

                tapered_weights = tapered_weights_callable_wrapper_func

            # Define block_weights as either None, an array, or a callable
            block_weights_choice = np.random.choice([0, 1, 2])
            if block_weights_choice == 0:
                block_weights = None
            elif block_weights_choice == 1:
                block_weights = block_weights_func(int(X.shape[0]))
            else:
                # Define a callable for block_weights
                def block_weights_callable_wrapper_func(
                    size: int,
                ) -> np.ndarray:
                    return block_weights_func(size)

                block_weights = block_weights_callable_wrapper_func

            # Ensure blocks have unique first indices to avoid ambiguity
            blocks = unique_first_indices(blocks)

            # Initialize BlockResampler
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=block_weights,
                tapered_weights=tapered_weights,
                rng=rng,
            )

            # Assertions to verify correct initialization

            # 1. Compare blocks using np.testing.assert_array_equal for each pair
            assert len(br.blocks) == len(blocks), "Number of blocks mismatch."
            for original_block, br_block in zip(blocks, br.blocks):
                np.testing.assert_array_equal(
                    original_block, br_block, err_msg="Blocks do not match."
                )

            # 2. Compare X
            np.testing.assert_array_equal(
                br.X, X, err_msg="'X' does not match."
            )

            # 3. Compare RNG
            assert br.rng == check_generator(
                rng
            ), "Random number generators do not match."

            # 4. Verify block_weights_normalized
            if block_weights is None:
                expected_weights = np.ones(len(X)) / len(X)
                np.testing.assert_array_almost_equal(
                    br.block_weights_normalized,
                    expected_weights,
                    err_msg="'block_weights_normalized' normalization failed for None.",
                )
            elif isinstance(block_weights, np.ndarray):
                normalized_weights = block_weights / block_weights.sum()
                np.testing.assert_array_almost_equal(
                    br.block_weights_normalized,
                    normalized_weights,
                    err_msg="'block_weights_normalized' normalization failed for ndarray.",
                )
            else:
                # If block_weights is a callable, verify the normalized weights
                assert isinstance(
                    br.block_weights_normalized, np.ndarray
                ), "Normalized block_weights is not a numpy array."
                assert np.isclose(
                    br.block_weights_normalized.sum(), 1.0
                ), "Normalized block_weights do not sum to 1."
                assert len(br.block_weights_normalized) == len(
                    X
                ), "Length of normalized block_weights does not match 'X'."

            # 5. Verify tapered_weights_normalized
            if tapered_weights is None:
                # Should be list of uniform weights for each block
                assert len(br.tapered_weights_normalized) == len(
                    blocks
                ), "Number of tapered_weights_normalized does not match number of blocks."
                for i, (block, taper) in enumerate(
                    zip(blocks, br.tapered_weights_normalized)
                ):
                    expected_taper = np.ones(len(block)) / len(block)
                    np.testing.assert_array_almost_equal(
                        taper,
                        expected_taper,
                        err_msg=f"'tapered_weights_normalized[{
                            i}]' normalization failed for None.",
                    )
            else:
                # tapered_weights is a callable returning list of arrays
                assert isinstance(
                    br.tapered_weights_normalized, list
                ), "'tapered_weights_normalized' is not a list."
                assert len(br.tapered_weights_normalized) == len(
                    blocks
                ), "Number of tapered_weights_normalized does not match number of blocks."
                for i, taper in enumerate(br.tapered_weights_normalized):
                    assert isinstance(
                        taper, np.ndarray
                    ), f"'tapered_weights_normalized[{
                            i}]' is not a numpy array."
                    assert len(taper) == len(
                        blocks[i]
                    ), f"Length of 'tapered_weights_normalized[{
                            i}]' does not match length of block {i}."
                    assert np.isclose(
                        taper.sum(), 1.0
                    ), f"'tapered_weights_normalized[{i}]' do not sum to 1."
                    assert (
                        np.max(taper) <= 1.0
                    ), f"'tapered_weights_normalized[{
                            i}]' contain values greater than 1."

            # Optional: Further assertions can be added as needed

    class TestFailingCases:
        """Test cases where BlockResampler should raise exceptions."""

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_wrong_block_weights_length(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test initializing BlockResampler with block_weights of incorrect length.
            """
            blocks, X = block_indices_and_X
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            # Assigning block_weights with incorrect length should raise ValidationError
            with pytest.raises(
                ValidationError,
                match="'block_weights' array length .* must match 'input_length' .*",
            ):
                br.block_weights = np.arange(len(X) + 1)

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_wrong_block_weights_negative_values(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test initializing BlockResampler with block_weights containing negative values.
            """
            blocks, X = block_indices_and_X
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            # Assigning block_weights with negative values should raise ValidationError
            with pytest.raises(
                ValidationError,
                match="'block_weights' must contain non-negative values.",
            ):
                br.block_weights = -np.ones(len(X))

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_wrong_block_weights_non_callable_return_type(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test initializing BlockResampler with a block_weights callable that does not return a numpy array.
            """
            blocks, X = block_indices_and_X
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            # Assigning a callable that does not return np.ndarray should raise ValidationError
            with pytest.raises(
                ValidationError,
                match="'block_weights' callable must return a numpy array.",
            ):

                def invalid_block_weights_callable(size: int) -> list:
                    return [1] * size

                br.block_weights = invalid_block_weights_callable

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_wrong_block_weights_non_callable_type(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test initializing BlockResampler with block_weights being of incorrect type (not callable or ndarray).
            """
            blocks, X = block_indices_and_X
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            # Assigning block_weights with non-callable, non-ndarray type should raise ValidationError
            with pytest.raises(ValidationError) as exc_info:
                br.block_weights = "invalid_weights"

            # Extract the errors from the ValidationError
            errors = exc_info.value.errors()

            # Assert that there are exactly 2 errors
            assert (
                len(errors) == 2
            ), f"Expected 2 validation errors, got {
                len(errors)}"

            # Define the expected error messages
            expected_errors = [
                {
                    "loc": ("block_weights", "callable"),
                    "msg": "Input should be callable",
                    "type": "callable_type",
                },
                {
                    "loc": ("block_weights", "is-instance[ndarray]"),
                    "msg": "Input should be an instance of ndarray",
                    "type": "is_instance_of",
                },
            ]

            # Iterate over expected errors and verify each one is present
            for expected_error in expected_errors:
                # Check if any error in the actual errors matches the expected error
                match_found = False
                for actual_error in errors:
                    if (
                        actual_error["loc"] == expected_error["loc"]
                        and actual_error["msg"] == expected_error["msg"]
                        and actual_error["type"] == expected_error["type"]
                    ):
                        match_found = True
                        break
                assert (
                    match_found
                ), f"Expected error {expected_error} not found in actual errors {errors}"


class TestResampleBlocks:
    """Test the resample_blocks method of BlockResampler."""

    class TestPassingCases:
        """Test cases where resample_blocks should work correctly."""

        @settings(deadline=1000)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_resample_blocks_valid_inputs(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that the 'resample_blocks' method works correctly with valid inputs.
            """
            blocks, X = block_indices_and_X
            # Ensure blocks have unique first indices to avoid ambiguity
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            # Initialize BlockResampler without specifying weights
            br = BlockResampler(blocks=blocks, X=X, rng=rng)
            new_blocks, new_tapered_weights = br.resample_blocks()

            # Check that the total length of the new blocks equals the length of X
            total_length = sum(len(block) for block in new_blocks)
            assert total_length == len(
                X
            ), f"Total length {
                total_length} does not match X length {len(X)}"

            # Check that the lengths of new_blocks and new_tapered_weights are equal
            assert len(new_blocks) == len(
                new_tapered_weights
            ), "Mismatch in lengths of new_blocks and new_tapered_weights"

            # If multiple blocks exist, ensure different resamplings produce different results
            if len(blocks) > 1:
                # Resample again with the same RNG and expect different results
                new_blocks_2, new_tapered_weights_2 = br.resample_blocks()
                check_list_of_arrays_equality(
                    new_blocks, new_blocks_2, equal=False
                )

                # Resample with a different RNG and expect different results
                rng2 = np.random.default_rng((random_seed + 1) * 2)
                br_new_rng = BlockResampler(blocks=blocks, X=X, rng=rng2)
                new_blocks_3, new_tapered_weights_3 = (
                    br_new_rng.resample_blocks()
                )
                check_list_of_arrays_equality(
                    new_blocks, new_blocks_3, equal=False
                )

                # Resample again with the original RNG and expect same results as the first resampling
                br_same_rng = BlockResampler(blocks=blocks, X=X, rng=rng)
                new_blocks_4, new_tapered_weights_4 = (
                    br_same_rng.resample_blocks()
                )
                check_list_of_arrays_equality(new_blocks, new_blocks_4)

    class TestFailingCases:
        """Test cases where resample_blocks should raise exceptions."""

        # Currently empty, but can be populated with tests that ensure proper exceptions are raised


class TestGenerateBlockIndicesAndData:
    """Test the resample_block_indices_and_data method of BlockResampler."""

    class TestPassingCases:
        """Test cases where resample_block_indices_and_data should work correctly."""

        @settings(deadline=None)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_valid_inputs(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that the 'resample_block_indices_and_data' method works correctly with valid inputs.
            """
            blocks, X = block_indices_and_X
            # Ensure blocks have unique first indices to avoid ambiguity
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            # Define tapered_weights as either None or a callable returning a list of arrays
            tapered_weights_choice = np.random.choice([0, 1])
            if tapered_weights_choice == 0:
                tapered_weights = None
            else:
                # Define a callable that captures the current blocks via closure
                def tapered_weights_callable_wrapper_func(
                    num_blocks: int,
                ) -> list[np.ndarray]:
                    return tapered_weights_callable(num_blocks, blocks)

                tapered_weights = tapered_weights_callable_wrapper_func

            # Initialize BlockResampler
            br = BlockResampler(
                blocks=blocks,
                X=X,
                rng=rng,
                tapered_weights=tapered_weights,
            )

            # Perform resampling
            new_blocks, block_data = br.resample_block_indices_and_data()

            # Check that the total length of the new blocks equals the length of X
            total_length_blocks = sum(len(block) for block in new_blocks)
            total_length_data = sum(
                len(data_block) for data_block in block_data
            )
            assert total_length_blocks == len(
                X
            ), f"Total blocks length {
                total_length_blocks} does not match X length {len(X)}"
            assert total_length_data == len(
                X
            ), f"Total data length {
                total_length_data} does not match X length {len(X)}"

            # Check that the lengths of new_blocks and block_data are equal
            assert len(new_blocks) == len(
                block_data
            ), "Mismatch in lengths of new_blocks and block_data"

            # Check that each data_block has the correct shape
            for i, block in enumerate(new_blocks):
                if X.ndim > 1:
                    expected_shape = (len(block), X.shape[1])
                else:
                    expected_shape = (len(block),)
                assert (
                    block_data[i].shape == expected_shape
                ), f"Data block {i} shape {
                    block_data[i].shape} does not match expected {expected_shape}"

            # If multiple blocks exist, ensure different resamplings produce different results
            if len(blocks) > 1:
                # Resample again with the same RNG and expect different results
                new_blocks_2, block_data_2 = (
                    br.resample_block_indices_and_data()
                )
                check_list_of_arrays_equality(
                    new_blocks, new_blocks_2, equal=False
                )

                # Resample with a different RNG and expect different results
                rng2 = np.random.default_rng((random_seed + 1) * 2)
                br_new_rng = BlockResampler(
                    blocks=blocks,
                    X=X,
                    rng=rng2,
                    tapered_weights=tapered_weights,
                )
                new_blocks_3, block_data_3 = (
                    br_new_rng.resample_block_indices_and_data()
                )
                check_list_of_arrays_equality(
                    new_blocks, new_blocks_3, equal=False
                )

                # Resample again with the original RNG and expect same results as the first resampling
                br_same_rng = BlockResampler(
                    blocks=blocks,
                    X=X,
                    rng=rng,
                    tapered_weights=tapered_weights,
                )
                new_blocks_4, block_data_4 = (
                    br_same_rng.resample_block_indices_and_data()
                )
                check_list_of_arrays_equality(new_blocks, new_blocks_4)
