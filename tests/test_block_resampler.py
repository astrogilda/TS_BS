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
        def test_init_with_valid_inputs(
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
                block_weights = block_weights_func(len(blocks))
            else:
                # Define a callable for block_weights
                def block_weights_callable_wrapper_func(
                    num_blocks: int,
                ) -> np.ndarray:
                    return block_weights_func(num_blocks)

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
                expected_weights = np.ones(len(blocks)) / len(blocks)
                np.testing.assert_almost_equal(
                    br.block_weights_normalized,
                    expected_weights,
                    err_msg="'block_weights_normalized' normalization failed for None.",
                )
            elif isinstance(block_weights, np.ndarray):
                normalized_weights = block_weights / block_weights.sum()
                np.testing.assert_almost_equal(
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
                    blocks
                ), "Length of normalized block_weights does not match number of blocks."

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
                    np.testing.assert_almost_equal(
                        taper,
                        expected_taper,
                        err_msg=f"'tapered_weights_normalized[{
                            i}]' normalization failed for None.",
                    )
            else:
                # tapered_weights is a callable or a list of arrays
                assert isinstance(
                    br.tapered_weights_normalized, list
                ), "'tapered_weights_normalized' is not a list."
                assert len(br.tapered_weights_normalized) == len(
                    blocks
                ), "Number of tapered_weights_normalized does not match number of blocks."
                for i, taper in enumerate(br.tapered_weights_normalized):
                    assert isinstance(
                        taper, np.ndarray
                    ), f"'tapered_weights_normalized[{i}]' is not a numpy array."
                    assert len(taper) == len(
                        blocks[i]
                    ), f"Length of 'tapered_weights_normalized[{i}]' does not match length of block {i}."
                    assert np.isclose(
                        taper.sum(), 1.0
                    ), f"'tapered_weights_normalized[{i}]' do not sum to 1."
                    assert (
                        np.max(taper) <= 1.0
                    ), f"'tapered_weights_normalized[{i}]' contain values greater than 1."

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
                match="'block_weights' array length .* must match number of blocks .*",
            ):
                br.block_weights = np.arange(len(blocks) + 1)

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
                # match="'block_weights' must contain non-negative values .*",
            ):
                br.block_weights = -np.ones(len(blocks))

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
                # match="Error in 'block_weights' callable .*",
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
            # Attempting to initialize with invalid block_weights type should raise ValidationError
            with pytest.raises(ValidationError) as exc_info:
                BlockResampler(
                    blocks=blocks,
                    X=X,
                    block_weights="invalid_weights",
                    tapered_weights=None,
                    rng=None,
                )

            # Extract the errors from the ValidationError
            errors = exc_info.value.errors()

            # Assert that there is at least one relevant error
            assert (
                len(errors) > 0
            ), "Expected validation errors for invalid block_weights type."

            # Check that at least one error message matches the expected pattern
            error_messages = [error["msg"] for error in errors]
            assert any(
                "callable" in msg or "ndarray" in msg for msg in error_messages
            ), "Expected errors related to 'callable' or 'ndarray' types."

        @settings(deadline=None)
        @given(
            valid_block_indices_and_X,
            st.sampled_from(
                [1, -1]
            ),  # Generate length_delta as either +1 or -1
        )
        def test_init_wrong_tapered_weights_length(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            length_delta: int,
        ) -> None:
            """
            Test initializing BlockResampler with tapered_weights of incorrect length.

            Parameters
            ----------
            block_indices_and_X : tuple[list[np.ndarray], np.ndarray]
                A tuple containing a list of block indices and the input data array X.
            length_delta : int
                The amount to adjust the length of tapered_weights (either +1 or -1).
            """
            blocks, X = block_indices_and_X
            len_blocks = len(blocks)

            # Generate correct tapered_weights
            correct_tapered_weights = [
                np.random.uniform(low=0, high=1e6, size=len(block))
                for block in blocks
            ]

            # Generate incorrect_tapered_weights by adjusting the length
            new_length = len_blocks + length_delta

            # Handle edge cases where new_length might be invalid
            if new_length < 1:
                pytest.skip(
                    "Cannot have tapered_weights with length less than 1."
                )
            elif new_length == len_blocks:
                pytest.skip(
                    "tapered_weights length matches number of blocks; skipping."
                )

            else:
                if new_length > len_blocks:
                    # Add extra tapered_weights to make the list longer
                    extra_tapered_weights = [
                        np.random.uniform(low=0, high=1e6, size=1)
                        for _ in range(new_length - len_blocks)
                    ]
                    incorrect_tapered_weights = (
                        correct_tapered_weights + extra_tapered_weights
                    )
                else:
                    # Remove tapered_weights to make the list shorter
                    incorrect_tapered_weights = correct_tapered_weights[
                        :new_length
                    ]

                # Perform the test only if new_length is different from len_blocks
                if new_length != len_blocks:
                    with pytest.raises(
                        ValidationError,
                        match="'tapered_weights' list length .* must match number of blocks .*",
                    ):
                        BlockResampler(
                            blocks=blocks,
                            X=X,
                            block_weights=None,
                            tapered_weights=incorrect_tapered_weights,
                            rng=None,
                        )
                else:
                    pytest.skip(
                        "tapered_weights length matches number of blocks; skipping."
                    )

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_wrong_tapered_weights_negative_values(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test initializing BlockResampler with tapered_weights containing negative values.
            """
            blocks, X = block_indices_and_X
            # Create tapered_weights with negative values for the first block
            tapered_weights = [
                np.array([-1.0, 0.8, 0.6]),
                np.array([1.0, 0.9, 0.7]),
                np.array([1.0, 0.85, 0.65]),
            ]
            with pytest.raises(
                ValidationError,
                # match="tapered_weights' must contain non-negative values .*",
            ):
                BlockResampler(
                    blocks=blocks,
                    X=X,
                    block_weights=None,
                    tapered_weights=tapered_weights,
                    rng=None,
                )

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_wrong_tapered_weights_non_callable_return_type(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test initializing BlockResampler with a tapered_weights callable that does not return a list of numpy arrays.
            """
            blocks, X = block_indices_and_X
            with pytest.raises(
                ValidationError,
                # match="Error in 'tapered_weights' callable .*",
            ):

                def invalid_tapered_weights_callable(
                    num_blocks: int,
                ) -> np.ndarray:
                    return np.array([1] * 5)

                BlockResampler(
                    blocks=blocks,
                    X=X,
                    block_weights=None,
                    tapered_weights=invalid_tapered_weights_callable,
                    rng=None,
                )

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_wrong_tapered_weights_non_callable_type(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test initializing BlockResampler with tapered_weights being of incorrect type (not callable or list of ndarray).
            """
            blocks, X = block_indices_and_X
            # Attempting to initialize with invalid tapered_weights type should raise ValidationError
            with pytest.raises(ValidationError) as exc_info:
                BlockResampler(
                    blocks=blocks,
                    X=X,
                    block_weights=None,
                    tapered_weights="invalid_tapered_weights",
                    rng=None,
                )

            # Extract the errors from the ValidationError
            errors = exc_info.value.errors()

            # Assert that there is at least one relevant error
            assert (
                len(errors) > 0
            ), "Expected validation errors for invalid tapered_weights type."

            # Check that at least one error message matches the expected pattern
            error_messages = [error["msg"] for error in errors]
            assert any(
                "callable" in msg or "list" in msg for msg in error_messages
            ), "Expected errors related to 'callable' or 'list' types."


class TestResampleBlocks:
    """Test the resample_blocks method of BlockResampler."""

    class TestPassingCases:
        """Test cases where resample_blocks should work correctly."""

        @settings(deadline=None)
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
                block_weights = block_weights_func(len(blocks))
            else:
                # Define a callable for block_weights
                def block_weights_callable_wrapper_func(
                    num_blocks: int,
                ) -> np.ndarray:
                    return block_weights_func(num_blocks)

                block_weights = block_weights_callable_wrapper_func

            # Initialize BlockResampler
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=block_weights,
                tapered_weights=tapered_weights,
                rng=rng,
            )

            # Perform resampling
            new_blocks, new_tapered_weights = br.resample_blocks()

            # Check that the total length of the new blocks equals the length of X
            total_length = sum(len(block) for block in new_blocks)
            assert total_length == len(
                X
            ), f"Total length {total_length} does not match X length {len(X)}"

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
                br_new_rng = BlockResampler(
                    blocks=blocks,
                    X=X,
                    block_weights=block_weights,
                    tapered_weights=tapered_weights,
                    rng=rng2,
                )
                new_blocks_3, new_tapered_weights_3 = (
                    br_new_rng.resample_blocks()
                )
                check_list_of_arrays_equality(
                    new_blocks, new_blocks_3, equal=False
                )

                # Resample again with the original RNG and expect same results as the first resampling
                br_same_rng = BlockResampler(
                    blocks=blocks,
                    X=X,
                    block_weights=block_weights,
                    tapered_weights=tapered_weights,
                    rng=rng,
                )
                new_blocks_4, new_tapered_weights_4 = (
                    br_same_rng.resample_blocks()
                )
                check_list_of_arrays_equality(new_blocks, new_blocks_4)

    class TestFailingCases:
        """Test cases where resample_blocks should raise exceptions."""

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_resample_blocks_zero_sum_weights(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test that resample_blocks raises an error when block_weights sum to zero.
            """
            blocks, X = block_indices_and_X
            # Assign block_weights to sum to zero
            block_weights = np.zeros(len(blocks))
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=block_weights,
                tapered_weights=None,
                rng=None,
            )
            with pytest.raises(
                ValueError,
                match="Sum of block_weights is zero. Cannot proceed with resampling.",
            ):
                br.resample_blocks()

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_resample_blocks_invalid_block_indices(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test that resample_blocks raises an error when block indices are out of range.
            """
            blocks, X = block_indices_and_X
            # Introduce an invalid block index
            if blocks:
                blocks_with_invalid = blocks.copy()
                blocks_with_invalid[0] = np.append(
                    blocks_with_invalid[0], X.shape[0] + 1
                )
                with pytest.raises(
                    ValueError,
                    match="must be within the range of the input data array 'X'",
                ):
                    BlockResampler(
                        blocks=blocks_with_invalid,
                        X=X,
                        block_weights=None,
                        tapered_weights=None,
                        rng=None,
                    ).resample_blocks()

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_resample_blocks_empty_blocks(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test that resample_blocks raises an error when blocks list is empty.
            """
            blocks, X = block_indices_and_X
            with pytest.raises(
                ValueError, match="'blocks' list cannot be empty."
            ):
                BlockResampler(
                    blocks=[],
                    X=X,
                    block_weights=None,
                    tapered_weights=None,
                    rng=None,
                ).resample_blocks()


class TestResampleBlockIndicesAndData:
    """Test the resample_block_indices_and_data method of BlockResampler."""

    class TestPassingCases:
        """Test cases where resample_block_indices_and_data should work correctly."""

        @settings(deadline=None)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_resample_block_indices_and_data_valid_inputs(
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

            # Define block_weights as either None, an array, or a callable
            block_weights_choice = np.random.choice([0, 1, 2])
            if block_weights_choice == 0:
                block_weights = None
            elif block_weights_choice == 1:
                block_weights = block_weights_func(len(blocks))
            else:
                # Define a callable for block_weights
                def block_weights_callable_wrapper_func(
                    num_blocks: int,
                ) -> np.ndarray:
                    return block_weights_func(num_blocks)

                block_weights = block_weights_callable_wrapper_func

            # Initialize BlockResampler
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=block_weights,
                tapered_weights=tapered_weights,
                rng=rng,
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
            ), f"Total blocks length {total_length_blocks} does not match X length {len(X)}"
            assert total_length_data == len(
                X
            ), f"Total data length {total_length_data} does not match X length {len(X)}"

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
                ), f"Data block {i} shape {block_data[i].shape} does not match expected {expected_shape}"

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
                    block_weights=block_weights,
                    tapered_weights=tapered_weights,
                    rng=rng2,
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
                    block_weights=block_weights,
                    tapered_weights=tapered_weights,
                    rng=rng,
                )
                new_blocks_4, block_data_4 = (
                    br_same_rng.resample_block_indices_and_data()
                )
                check_list_of_arrays_equality(new_blocks, new_blocks_4)

    class TestFailingCases:
        """Test cases where resample_block_indices_and_data should raise exceptions."""

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_resample_block_indices_and_data_zero_sum_weights(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test that resample_block_indices_and_data raises an error when block_weights sum to zero.
            """
            blocks, X = block_indices_and_X
            # Assign block_weights to sum to zero
            block_weights = np.zeros(len(blocks))
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=block_weights,
                tapered_weights=None,
                rng=None,
            )
            with pytest.raises(
                ValueError,
                match="Sum of block_weights is zero. Cannot proceed with resampling.",
            ):
                br.resample_block_indices_and_data()

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_resample_block_indices_and_data_invalid_block_indices(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test that resample_block_indices_and_data raises an error when block indices are out of range.
            """
            blocks, X = block_indices_and_X
            # Introduce an invalid block index
            if blocks:
                blocks_with_invalid = blocks.copy()
                blocks_with_invalid[0] = np.append(
                    blocks_with_invalid[0], X.shape[0] + 1
                )
                with pytest.raises(
                    ValueError,
                    match="must be within the range of the input data array 'X'",
                ):
                    BlockResampler(
                        blocks=blocks_with_invalid,
                        X=X,
                        block_weights=None,
                        tapered_weights=None,
                        rng=None,
                    ).resample_block_indices_and_data()

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_resample_block_indices_and_data_empty_blocks(
            self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
        ) -> None:
            """
            Test that resample_block_indices_and_data raises an error when blocks list is empty.
            """
            blocks, X = block_indices_and_X
            with pytest.raises(
                ValueError, match="'blocks' list cannot be empty."
            ):
                BlockResampler(
                    blocks=[],
                    X=X,
                    block_weights=None,
                    tapered_weights=None,
                    rng=None,
                ).resample_block_indices_and_data()


class TestEquality:
    """Test the __eq__ method of BlockResampler."""

    class TestPassingCases:
        """Test cases where BlockResampler instances should be equal."""

        @given(valid_block_indices_and_X, rng_strategy)
        def test_equality_same_parameters(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that two BlockResampler instances with the same parameters are equal.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            br1 = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=None,
                rng=rng,
            )
            br2 = BlockResampler(
                blocks=blocks.copy(),
                X=X.copy(),
                block_weights=None,
                tapered_weights=None,
                rng=check_generator(rng),
            )

            assert (
                br1 == br2
            ), "BlockResampler instances with identical parameters should be equal."

    class TestFailingCases:
        """Test cases where BlockResampler instances should not be equal."""

        @given(valid_block_indices_and_X, rng_strategy)
        def test_equality_different_parameters(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that two BlockResampler instances with different parameters are not equal.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng1 = np.random.default_rng(random_seed)
            rng2 = np.random.default_rng(random_seed + 1)

            br1 = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=None,
                rng=rng1,
            )
            br2 = BlockResampler(
                blocks=blocks.copy(),
                X=X.copy(),
                block_weights=None,
                tapered_weights=None,
                rng=rng2,
            )

            assert (
                br1 != br2
            ), "BlockResampler instances with different RNGs should not be equal."


class TestReprAndStr:
    """Test the __repr__ and __str__ methods of BlockResampler."""

    class TestPassingCases:
        """Test cases where __repr__ and __str__ should work correctly."""

        @given(valid_block_indices_and_X, rng_strategy)
        def test_repr(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test the __repr__ method of BlockResampler.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=None,
                rng=rng,
            )

            repr_str = repr(br)
            assert (
                "BlockResampler(blocks=" in repr_str
            ), "__repr__ does not contain expected string."
            assert "X=" in repr_str, "__repr__ does not contain 'X='."
            assert (
                "block_weights=" in repr_str
            ), "__repr__ does not contain 'block_weights='."
            assert (
                "tapered_weights=" in repr_str
            ), "__repr__ does not contain 'tapered_weights='."
            assert "rng=" in repr_str, "__repr__ does not contain 'rng='."

        @given(valid_block_indices_and_X, rng_strategy)
        def test_str(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test the __str__ method of BlockResampler.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=None,
                rng=rng,
            )

            str_repr = str(br)
            assert (
                "BlockResampler with" in str_repr
            ), "__str__ does not contain expected string."
            assert (
                f"{len(
                blocks)} blocks"
                in str_repr
            ), "__str__ does not correctly state number of blocks."
            assert (
                f"input data of shape {
                X.shape}"
                in str_repr
            ), "__str__ does not correctly state shape of X."
            assert (
                "block_weights=Uniform" in str_repr
            ), "__str__ does not correctly indicate uniform block_weights."
            assert (
                "tapered_weights=Uniform" in str_repr
            ), "__str__ does not correctly indicate uniform tapered_weights."
            assert (
                f"and RNG {
                br.rng}"
                in str_repr
            ), "__str__ does not correctly display RNG."

    class TestFailingCases:
        """Test cases where __repr__ and __str__ should not have issues, typically not applicable."""

        # Generally, __repr__ and __str__ should not fail with valid instances.
        # However, we can test that they handle unusual but valid inputs.

        @given(valid_block_indices_and_X, rng_strategy)
        def test_repr_with_unusual_blocks(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test the __repr__ method with blocks containing large indices.
            """
            blocks, X = block_indices_and_X
            # Modify blocks to have large indices within range
            blocks = [block + 1000 for block in blocks]
            rng = np.random.default_rng(random_seed)

            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=None,
                rng=rng,
            )

            repr_str = repr(br)
            assert (
                "BlockResampler(blocks=" in repr_str
            ), "__repr__ does not contain expected string."
            assert "X=" in repr_str, "__repr__ does not contain 'X='."

        @given(valid_block_indices_and_X, rng_strategy)
        def test_str_with_no_tapered_weights(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test the __str__ method when tapered_weights are not provided.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=None,
                rng=rng,
            )

            str_repr = str(br)
            assert (
                "tapered_weights=Uniform" in str_repr
            ), "__str__ does not correctly indicate uniform tapered_weights."


class TestWeightNormalization:
    """Test the normalization of block_weights and tapered_weights."""

    class TestPassingCases:
        """Test cases where weight normalization should work correctly."""

        @given(valid_block_indices_and_X, rng_strategy)
        def test_block_weights_normalization(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that block_weights_normalized sums to 1 and maintains the correct proportions.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            # Define random block_weights
            block_weights = np.random.uniform(low=0, high=10, size=len(blocks))

            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=block_weights,
                tapered_weights=None,
                rng=rng,
            )

            # Check that normalized weights sum to 1
            assert np.isclose(
                br.block_weights_normalized.sum(), 1.0
            ), "Normalized block_weights do not sum to 1."

            # Check that normalization preserves proportions
            expected_normalized = block_weights / block_weights.sum()
            np.testing.assert_almost_equal(
                br.block_weights_normalized,
                expected_normalized,
                err_msg="block_weights_normalized do not preserve original proportions.",
            )

        @given(valid_block_indices_and_X, rng_strategy)
        def test_tapered_weights_normalization(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that tapered_weights_normalized sums to 1 for each block and maintains the correct proportions.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            # Define random tapered_weights
            tapered_weights = [
                np.random.uniform(low=0, high=10, size=len(block))
                for block in blocks
            ]

            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=tapered_weights,
                rng=rng,
            )

            # Check that each tapered_weights_normalized sum to 1
            for i, taper in enumerate(br.tapered_weights_normalized):
                assert np.isclose(
                    taper.sum(), 1.0
                ), f"tapered_weights_normalized[{
                    i}] do not sum to 1."

                # Check that normalization preserves proportions
                expected_normalized = (
                    tapered_weights[i] / tapered_weights[i].sum()
                )
                np.testing.assert_almost_equal(
                    taper,
                    expected_normalized,
                    err_msg=f"tapered_weights_normalized[{
                        i}] do not preserve original proportions.",
                )

    class TestFailingCases:
        """Test cases where weight normalization should fail."""

        @given(valid_block_indices_and_X, rng_strategy)
        def test_block_weights_normalization_zero_sum(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that block_weights_normalized raises an error when block_weights sum to zero.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            # Define block_weights that sum to zero
            block_weights = np.zeros(len(blocks))

            with pytest.raises(
                ValueError,
                match="'block_weights' sum must be greater than zero.",
            ):
                BlockResampler(
                    blocks=blocks,
                    X=X,
                    block_weights=block_weights,
                    tapered_weights=None,
                    rng=rng,
                )

        @given(valid_block_indices_and_X, rng_strategy)
        def test_tapered_weights_normalization_zero_sum(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that tapered_weights_normalized raises an error when tapered_weights sum to zero for a block.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            # Define tapered_weights with zero sum for the first block
            tapered_weights = [
                np.zeros(len(blocks[0])),
                np.random.uniform(low=0, high=10, size=len(blocks[1])),
                np.random.uniform(low=0, high=10, size=len(blocks[2])),
            ]

            with pytest.raises(
                ValueError,
                match=r"Sum of 'tapered_weights\[0\]' must be greater than zero.",
            ):
                BlockResampler(
                    blocks=blocks,
                    X=X,
                    block_weights=None,
                    tapered_weights=tapered_weights,
                    rng=rng,
                )


class TestEdgeCases:
    """Test edge cases for BlockResampler."""

    class TestPassingCases:
        """Test cases where BlockResampler handles edge cases correctly."""

        def test_single_block_exact_fit(self):
            """
            Test with a single block that exactly fits X without truncation.
            """
            blocks = [np.array([0, 1, 2, 3, 4])]
            X = np.arange(5).reshape(-1, 1)
            tapered_weights = [np.ones(5)]
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=np.array([1.0]),
                tapered_weights=tapered_weights,
                rng=42,
            )
            new_blocks, new_tapered_weights = br.resample_blocks()
            assert (
                len(new_blocks) == 1
            ), "There should be exactly one resampled block."
            assert np.array_equal(
                new_blocks[0], blocks[0]
            ), "Resampled block does not match the single block."
            assert (
                len(new_tapered_weights) == 1
            ), "There should be exactly one tapered_weights array."
            assert np.array_equal(
                new_tapered_weights[0], tapered_weights[0]
            ), "Resampled tapered_weights do not match."

        def test_multiple_blocks_exact_fit(self):
            """
            Test with multiple blocks that exactly fit X without needing truncation.
            """
            blocks = [
                np.array([0, 1]),
                np.array([2, 3]),
                np.array([4, 5]),
            ]
            X = np.arange(6).reshape(-1, 1)
            block_weights = np.array([1.0, 1.0, 1.0])
            tapered_weights = [
                np.array([0.5, 0.5]),
                np.array([0.6, 0.4]),
                np.array([0.7, 0.3]),
            ]
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=block_weights,
                tapered_weights=tapered_weights,
                rng=42,
            )
            new_blocks, new_tapered_weights = br.resample_blocks()
            total_length = sum(len(block) for block in new_blocks)
            assert total_length == len(
                X
            ), "Total resampled block length does not match length of X."
            # Since weights are uniform and blocks fit exactly, expect all blocks to be present
            assert set(map(tuple, new_blocks)) == set(
                map(tuple, blocks)
            ), "Blocks should exactly fit X without resampling."

        def test_resample_blocks_with_truncated_block(self):
            """
            Test resampling where a block needs to be truncated to fit X.
            """
            blocks = [
                np.array([0, 1, 2, 3]),
                np.array([4, 5, 6]),
                np.array([7, 8, 9]),
            ]
            X = np.arange(10).reshape(-1, 1)
            block_weights = np.array([0.3, 0.3, 0.4])
            tapered_weights = [
                np.array([1.0, 0.8, 0.6, 0.4]),
                np.array([1.0, 0.9, 0.7]),
                np.array([1.0, 0.85, 0.65]),
            ]
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=block_weights,
                tapered_weights=tapered_weights,
                rng=42,
            )
            new_blocks, new_tapered_weights = br.resample_blocks()
            total_length = sum(len(block) for block in new_blocks)
            assert total_length == len(
                X
            ), "Total resampled block length does not match length of X."
            # Check that no block exceeds its original length
            for original_block, new_block in zip(blocks, new_blocks):
                assert len(new_block) <= len(
                    original_block
                ), "Resampled block exceeds original block length."

    class TestFailingCases:
        """Test cases where BlockResampler fails on edge cases."""

        def test_single_block_truncated(self):
            """
            Test with a single block that needs to be truncated to fit X.
            """
            blocks = [np.array([0, 1, 2, 3, 4, 5])]
            X = np.arange(4).reshape(-1, 1)  # Smaller than block length
            tapered_weights = [np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])]
            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=np.array([1.0]),
                tapered_weights=tapered_weights,
                rng=42,
            )
            new_blocks, new_tapered_weights = br.resample_blocks()
            assert (
                len(new_blocks) == 1
            ), "There should be exactly one resampled block."
            assert len(new_blocks[0]) == len(
                X
            ), "Resampled block should be truncated to fit X."
            assert np.array_equal(
                new_blocks[0], blocks[0][:4]
            ), "Resampled block is not correctly truncated."
            assert (
                len(new_tapered_weights) == 1
            ), "There should be exactly one tapered_weights array."
            assert np.array_equal(
                new_tapered_weights[0], tapered_weights[0][:4]
            ), "Resampled tapered_weights are not correctly truncated."


class TestWeightCallable:
    """Test scenarios where block_weights and tapered_weights are callables."""

    class TestPassingCases:
        """Test cases where weight callables should work correctly."""

        @given(valid_block_indices_and_X, rng_strategy)
        def test_block_weights_callable(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that block_weights provided as a callable are correctly generated and normalized.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            def weights_callable(num_blocks: int) -> np.ndarray:
                return np.array([0.1, 0.2, 0.7])

            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=weights_callable,
                tapered_weights=None,
                rng=rng,
            )

            expected_normalized = (
                weights_callable(len(blocks))
                / weights_callable(len(blocks)).sum()
            )
            np.testing.assert_almost_equal(
                br.block_weights_normalized,
                expected_normalized,
                err_msg="block_weights_normalized do not match expected values from callable.",
            )

        @given(valid_block_indices_and_X, rng_strategy)
        def test_tapered_weights_callable(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that tapered_weights provided as a callable are correctly generated and normalized.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            def tapered_callable(num_blocks: int) -> list[np.ndarray]:
                return [
                    np.array([1.0, 0.8, 0.6]),
                    np.array([1.0, 0.9, 0.7]),
                    np.array([1.0, 0.85, 0.65]),
                ]

            br = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=tapered_callable,
                rng=rng,
            )

            # Verify each tapered_weights_normalized
            for i, taper in enumerate(br.tapered_weights_normalized):
                expected_normalized = (
                    tapered_callable(len(blocks))[i]
                    / tapered_callable(len(blocks))[i].sum()
                )
                np.testing.assert_almost_equal(
                    taper,
                    expected_normalized,
                    err_msg=f"tapered_weights_normalized[{
                        i}] do not match expected values from callable.",
                )

    class TestFailingCases:
        """Test cases where weight callables should fail."""

        @given(valid_block_indices_and_X, rng_strategy)
        def test_block_weights_callable_invalid_return_type(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that providing a block_weights callable that returns a non-numpy array raises an error.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            def invalid_weights_callable(num_blocks: int) -> list:
                return [1] * num_blocks

            with pytest.raises(
                ValueError
            ):  # , match="Error in 'block_weights' callable .*"):
                BlockResampler(
                    blocks=blocks,
                    X=X,
                    block_weights=invalid_weights_callable,
                    tapered_weights=None,
                    rng=rng,
                )

        @given(valid_block_indices_and_X, rng_strategy)
        def test_tapered_weights_callable_invalid_return_type(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that providing a tapered_weights callable that returns a non-list raises an error.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            def invalid_tapered_callable(num_blocks: int) -> np.ndarray:
                return np.array([1] * 5)

            with pytest.raises(
                ValueError
            ):  # , match="Error in 'tapered_weights' callable .*"):
                BlockResampler(
                    blocks=blocks,
                    X=X,
                    block_weights=None,
                    tapered_weights=invalid_tapered_callable,
                    rng=rng,
                )


class TestRandomness:
    """Test the randomness and reproducibility of resampling."""

    class TestPassingCases:
        """Test cases where randomness should be reproducible."""

        @given(valid_block_indices_and_X, rng_strategy)
        def test_resample_blocks_reproducibility(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that resampling is reproducible with the same RNG seed.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)

            br1 = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=None,
                rng=rng,
            )
            br2 = BlockResampler(
                blocks=blocks.copy(),
                X=X.copy(),
                block_weights=None,
                tapered_weights=None,
                rng=check_generator(rng),
            )

            new_blocks1, new_tapered_weights1 = br1.resample_blocks()
            new_blocks2, new_tapered_weights2 = br2.resample_blocks()

            # Blocks should be identical
            check_list_of_arrays_equality(new_blocks1, new_blocks2, equal=True)

            # Tapered weights should be identical
            check_list_of_arrays_equality(
                new_tapered_weights1, new_tapered_weights2, equal=True
            )

        @given(valid_block_indices_and_X, rng_strategy)
        def test_resample_blocks_non_reproducibility(
            self,
            block_indices_and_X: tuple[list[np.ndarray], np.ndarray],
            random_seed: int,
        ) -> None:
            """
            Test that resampling produces different results with different RNG seeds.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng1 = np.random.default_rng(random_seed)
            rng2 = np.random.default_rng(random_seed + 1)

            br1 = BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=None,
                rng=rng1,
            )
            br2 = BlockResampler(
                blocks=blocks.copy(),
                X=X.copy(),
                block_weights=None,
                tapered_weights=None,
                rng=rng2,
            )

            new_blocks1, new_tapered_weights1 = br1.resample_blocks()
            new_blocks2, new_tapered_weights2 = br2.resample_blocks()

            # Blocks should not be identical
            check_list_of_arrays_equality(
                new_blocks1, new_blocks2, equal=False
            )

            # Tapered weights should not be identical
            check_list_of_arrays_equality(
                new_tapered_weights1, new_tapered_weights2, equal=False
            )


class TestExceptionMessages:
    """Test that exception messages are clear and informative."""

    @given(valid_block_indices_and_X)
    def test_exception_message_invalid_block_weights_type(
        self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
    ) -> None:
        """
        Test that providing an invalid type for block_weights raises an appropriate error message.
        """
        blocks, X = block_indices_and_X
        with pytest.raises(
            ValidationError,
            match="'block_weights' must be a callable or a numpy array .*",
        ):
            BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=123,  # Invalid type
                tapered_weights=None,
                rng=None,
            )

    @given(valid_block_indices_and_X)
    def test_exception_message_invalid_tapered_weights_type(
        self, block_indices_and_X: tuple[list[np.ndarray], np.ndarray]
    ) -> None:
        """
        Test that providing an invalid type for tapered_weights raises an appropriate error message.
        """
        blocks, X = block_indices_and_X
        with pytest.raises(
            ValidationError,
            match="'tapered_weights' must be a callable or a list of numpy arrays .*",
        ):
            BlockResampler(
                blocks=blocks,
                X=X,
                block_weights=None,
                tapered_weights=456,  # Invalid type
                rng=None,
            )
