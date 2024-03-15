"""Automated tests based on the skbase test suite template."""
from skbase.testing import BaseFixtureGenerator as _BaseFixtureGenerator
from skbase.testing import TestAllObjects as _TestAllObjects

from tsbootstrap.registry import OBJECT_TAG_LIST, all_objects
from tsbootstrap.tests.scenarios.scenarios_getter import retrieve_scenarios
from tsbootstrap.tests.test_switch import run_test_for_class

# whether to test only estimators from modules that are changed w.r.t. main
# default is False, can be set to True by pytest --only_changed_modules True flag
ONLY_CHANGED_MODULES = False

# objects temporarily excluded due to known bugs
TEMPORARY_EXCLUDED_OBJECTS = ["StationaryBlockBootstrap"]  # see bug #73


class PackageConfig:
    """Contains package config variables for test classes."""

    # class variables which can be overridden by descendants
    # ------------------------------------------------------

    # package to search for objects
    # expected type: str, package/module name, relative to python environment root
    package_name = "tsbootstrap"

    # list of object types (class names) to exclude
    # expected type: list of str, str are class names
    exclude_objects = ["ClassName"] + TEMPORARY_EXCLUDED_OBJECTS
    # exclude classes from extension templates
    # exclude classes with known bugs

    # list of valid tags
    # expected type: list of str, str are tag names
    valid_tags = OBJECT_TAG_LIST


class BaseFixtureGenerator(_BaseFixtureGenerator):
    """Fixture generator for base testing functionality in sktime.

    Test classes inheriting from this and not overriding pytest_generate_tests
        will have estimator and scenario fixtures parametrized out of the box.

    Descendants can override:
        estimator_type_filter: str, class variable; None or scitype string
            e.g., "forecaster", "transformer", "classifier", see BASE_CLASS_SCITYPE_LIST
            which estimators are being retrieved and tested
        fixture_sequence: list of str
            sequence of fixture variable names in conditional fixture generation
        _generate_[variable]: object methods, all (test_name: str, **kwargs) -> list
            generating list of fixtures for fixture variable with name [variable]
                to be used in test with name test_name
            can optionally use values for fixtures earlier in fixture_sequence,
                these must be input as kwargs in a call
        is_excluded: static method (test_name: str, est: class) -> bool
            whether test with name test_name should be excluded for estimator est
                should be used only for encoding general rules, not individual skips
                individual skips should go on the EXCLUDED_TESTS list in _config
            requires _generate_estimator_class and _generate_estimator_instance as is
        _excluded_scenario: static method (test_name: str, scenario) -> bool
            whether scenario should be skipped in test with test_name test_name
            requires _generate_estimator_scenario as is

    Fixtures parametrized
    ---------------------
    object_class: estimator inheriting from BaseObject
        ranges over estimator classes not excluded by EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
    object_instance: instance of estimator inheriting from BaseObject
        ranges over estimator classes not excluded by EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
        instances are generated by create_test_instance class method of estimator_class
    scenario: instance of TestScenario
        ranges over all scenarios returned by retrieve_scenarios
        applicable for estimator_class or estimator_instance
    """

    # overrides object retrieval in scikit-base
    def _all_objects(self):
        """Retrieve list of all object classes of type self.object_type_filter."""
        obj_list = all_objects(
            object_types=getattr(self, "object_type_filter", None),
            return_names=False,
            exclude_objects=self.exclude_objects,
        )

        # run_test_for_class selects the estimators to run
        # based on whether they have changed, and whether they have all dependencies
        # internally, uses the ONLY_CHANGED_MODULES flag,
        # and checks the python env against python_dependencies tag
        obj_list = [obj for obj in obj_list if run_test_for_class(obj)]

        def scitype(obj):
            type_tag = obj.get_class_tag("object_type", "object")
            return type_tag

        # exclude config objects and sampler objects
        excluded_types = ["config", "sampler"]
        obj_list = [obj for obj in obj_list if scitype(obj) not in excluded_types]

        return obj_list

    # which sequence the conditional fixtures are generated in
    fixture_sequence = [
        "object_class",
        "object_instance",
        "scenario",
    ]

    def _generate_scenario(self, test_name, **kwargs):
        """Return estimator test scenario.

        Fixtures parametrized
        ---------------------
        scenario: instance of TestScenario
            ranges over all scenarios returned by retrieve_scenarios
        """
        if "object_class" in kwargs.keys():
            obj = kwargs["object_class"]
        elif "object_instance" in kwargs.keys():
            obj = kwargs["object_instance"]
        else:
            return []

        scenarios = retrieve_scenarios(obj)
        scenarios = [s for s in scenarios if not self._excluded_scenario(test_name, s)]
        scenario_names = [type(scen).__name__ for scen in scenarios]

        return scenarios, scenario_names

    @staticmethod
    def _excluded_scenario(test_name, scenario):
        """Skip list generator for scenarios to skip in test_name.

        Arguments
        ---------
        test_name : str, name of test
        scenario : instance of TestScenario, to be used in test

        Returns
        -------
        bool, whether scenario should be skipped in test_name
        """
        # for now, all scenarios are enabled
        # if not scenario.get_tag("is_enabled", False, raise_error=False):
        #     return True

        return False


class TestAllObjects(PackageConfig, BaseFixtureGenerator, _TestAllObjects):
    """Generic tests for all objects in the mini package."""

    # override test_constructor to allow for kwargs
    def test_constructor(self, object_class):
        """Check that the constructor has sklearn compatible signature and behaviour.

        Overrides the test_constructor method from _TestAllObjects,
        in order to allow for the constructor to have kwargs.
        """
        try:
            # dispatch for remaining test logic
            super().test_constructor(object_class)
        except AssertionError as e:
            if not "constructor __init__ of" in str(e):
                raise e
